from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import copy
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from model.decode import generic_decode

from utils.image import gaussian_radius, draw_umich_gaussian

from model.ConvGRU import ConvGRU

from einops import rearrange as rearr, repeat

from spatial_correlation_sampler import spatial_correlation_sample


class LocalWalk(nn.Module):
    def __init__(self, topk=0, radius=0.2, temp=0.1, pad_value=0,
            broadcast_val=False, corr_module=False):
        super(LocalWalk, self).__init__()

        self.topk = topk
        self.radius = radius
        self.vals = {}
        self.idxmaps = {}
        self.temp = temp
        self.pad_value = pad_value

        self.broadcast_val = broadcast_val

        self.corr_module = corr_module

    def get_identity_label(self, keys):
        '''
        returns 1 x H*W x H x W as reshaped H*W x H*W identity matrix
        '''
        B, C, H, W = keys.shape
        name = f"{H}_{W}"
        if name not in self.vals:
            vals = self.distance_field(H, W).flatten(0, 1)
            vals = (vals == 0).float()
            vals = repeat(vals, 'n h w -> b n h w', b=B if not self.broadcast_val else 1)
            self.vals[name] = vals.to(keys.device)
            print('created vals')

        return self.vals[name]

    def knn(self, A):
        if self.pad_value == 0 or self.topk > 0:
            with torch.no_grad():
                mask = (A == self.pad_value)
                if self.topk > 0:
                  mask |= (A < A.topk(k=self.topk, dim=-1)[0].min(-1, keepdim=True)[0])
            A[mask] = -10

        return A

    def distance_field(self, H, W, p=2):
        gx, gy = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
        D = ( (gx[None, None, :, :] - gx[:, :, None, None]).abs()**p + (gy[None, None, :, :] - gy[:, :, None, None]).abs()**p ).float() #** (1/p)
        return D

    def make_scatter_map(self, keys, kH, kW):
        B, C, H, W = keys.shape
        name = f"{H}_{W}_{kH}_{kW}"
        if name not in self.idxmaps:
            idx_map = torch.arange(H*W).view(H, W)[None, None].float()
            idx_map = torch.nn.functional.unfold(idx_map, kernel_size=(kH, kW), stride=1, padding=(kH//2, kW//2))
            idx_map = rearr(idx_map, 'b n hw -> b hw n')
            idx_map = idx_map.clamp(min=0).long()
            self.idxmaps[name] = idx_map.to(keys.device)
            print('created idx map')

        return self.idxmaps[name]

    def forward(self, query, keys, val=None):
        '''
        assumes q, k, v: B D N
        '''

        B, C, H, W = keys.shape
        kW = kH = int(H * self.radius) * 2 + 1

        val = self.make_scatter_map(keys, kH, kW)
        out = self.get_identity_label(keys) * 0
        out = repeat(out, '1 n h w -> b (h w) n', b=B) if out.shape[0] == 1 else \
                rearr(out, 'b n h w -> b (h w) n')

        if self.corr_module:
            att = spatial_correlation_sample(query,
                               keys,
                               kernel_size=1,
                               patch_size=kH,
                               stride=1,
                               padding=0,
                               dilation=1,
                               dilation_patch=1) / self.temp
            att = rearr(att, 'b p1 p2 h w -> b h w (p1 p2)')

        A = self.knn(att)

        A = torch.exp(rearr(A, 'b h w n -> b (h w) n'))
        out.scatter_add_(2, val.to(A.device).expand_as(A), A)
        val = rearr(out, 'b (h w) n -> b n h w', h=H)

        return val

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class ParallelModel(nn.Module):
  def __init__(self, conv, head1, head2):
    super(ParallelModel, self).__init__()

    self.conv = conv
    self.head1 = head1
    self.head2 = head2

  def forward(self, x):
    hidden = F.relu(self.conv(x))

    return self.head1(hidden), self.head2(hidden)

class BaseModel(nn.Module):
    def __init__(self, heads, head_convs, num_stacks, last_channel, opt=None):
        super(BaseModel, self).__init__()
        if opt is not None and opt.head_kernel != 3:
          print('Using head kernel:', opt.head_kernel)
          head_kernel = opt.head_kernel
        else:
          head_kernel = 3
        self.num_stacks = num_stacks
        self.opt = opt

        self.local_rw = LocalWalk(radius=opt.local_rw_r, temp=opt.rw_temp, topk=0, pad_value=-1,
            broadcast_val=False, corr_module=True)

        if opt.is_recurrent:
          in_channel = last_channel
          if opt.pre_hm:
              in_channel += 1
          self.conv_gru = ConvGRU(in_channel, last_channel, (opt.gru_filter_size, opt.gru_filter_size), opt.num_gru_layers, batch_first=True, nl=opt.nl)

        self.rw_head = None
        if opt.rw_head_depth > 0:
          self.rw_head = nn.Sequential()
          for i in range(opt.rw_head_depth):
            self.rw_head.add_module('rw_conv%d' % i, nn.Conv2d(last_channel, last_channel, 
                    kernel_size=1, stride=1, padding=0, bias=True))
            self.rw_head.add_module('rw_relu%d' % i, nn.ReLU(True))

        self.heads = heads
        for head in self.heads:
          if 'tracking' in head and not opt.sup_reg:
            continue
          classes = self.heads[head]
          head_conv = head_convs[head]
          if len(head_conv) > 0:
            out = nn.Conv2d(head_conv[-1], classes, 
                  kernel_size=1, stride=1, padding=0, bias=True)
            conv = nn.Conv2d(last_channel, head_conv[0],
                            kernel_size=head_kernel, 
                            padding=head_kernel // 2, bias=True)
            convs = [conv]
            for k in range(1, len(head_conv)):
              convs.append(nn.Conv2d(head_conv[k - 1], head_conv[k], 
                            kernel_size=1, bias=True))
            if len(convs) == 1:
              fc = nn.Sequential(conv, nn.ReLU(inplace=True), out)
            elif len(convs) == 2:
              fc = nn.Sequential(
                convs[0], nn.ReLU(inplace=True), 
                convs[1], nn.ReLU(inplace=True), out)
            elif len(convs) == 3:
              fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), 
                  convs[2], nn.ReLU(inplace=True), out)
            elif len(convs) == 4:
              fc = nn.Sequential(
                  convs[0], nn.ReLU(inplace=True), 
                  convs[1], nn.ReLU(inplace=True), 
                  convs[2], nn.ReLU(inplace=True), 
                  convs[3], nn.ReLU(inplace=True), out)
            if 'hm' in head or 'visibility' in head:
              if not isinstance(fc, ParallelModel):
                fc[-1].bias.data.fill_(opt.prior_bias)
              else:
                fc.head1.bias.data.fill_(opt.prior_bias)
                fc.head2.bias.data.fill_(opt.prior_bias)
            else:
              fill_fc_weights(fc)
          else:
            fc = nn.Conv2d(last_channel, classes, 
                kernel_size=1, stride=1, padding=0, bias=True)
            if 'hm' in head:
              fc.bias.data.fill_(opt.prior_bias)
            else:
              fill_fc_weights(fc)
          self.__setattr__(head, fc)

    def img2feats(self, x):
      raise NotImplementedError

    def freeze_backbone(self):
      raise NotImplementedError

    def freeze_gru(self):
      for parameter in self.conv_gru.parameters():
          parameter.requires_grad = False    
    
    def imgpre2feats(self, x, pre_img=None, pre_hm=None):
      raise NotImplementedError

    def step(self, x, h, prev_feat=None):
      feats = self.imgpre2feats(x, None, torch.zeros(1))

      batch_size = int(len(feats[0]))
      inp = feats[0].view(batch_size, 1, feats[0].size(1), feats[0].size(2), feats[0].size(3))

      curr_step = inp[:, 0, :, :, :]
      if self.opt.pre_hm:
        hm = nn.functional.interpolate(x[0]['pre_hm'], size=(inp.size(3), inp.size(4)), mode="bilinear")
        curr_step = torch.cat((curr_step, hm), 1).unsqueeze(1)
      else:
        curr_step = curr_step.unsqueeze(1)
      intermediate_outputs, layer_reset_list, layer_update_list, last_output = self.conv_gru(curr_step, h)
      h = last_output
      feats = last_output[-1:][0]

      A = None
      curr_feat = None
      if self.opt.random_walk:
        if self.opt.pool_kernel > 0:
          pad = math.floor(self.opt.pool_kernel / 2)
          stride = pad + 1
          curr_feat = F.avg_pool2d(feats, self.opt.pool_kernel, stride, pad)
        else:
          curr_feat = feats
        if self.rw_head is not None:
          curr_feat = self.rw_head(curr_feat)
        curr_feat = F.normalize(curr_feat, p=2, dim=1)

        if prev_feat is not None:
          A = self.local_rw.forward(query=prev_feat, keys=curr_feat)
          A = rearr(A, 'b n h w -> b (h w) n')
          A /= A.sum(-1).unsqueeze(-1)

      out = []
      z = self.apply_heads(feats, {})
      out.append(z)

      return out, h, A, curr_feat

    def apply_heads(self, feature, z):
      for head in self.heads:
        if 'tracking' in head and not self.opt.sup_reg:
          continue
        z[head] = self.__getattr__(head)(feature)

      return z

    def forward(self, x, pre_img=None, pre_hm=None, batch_size=1):
      if (pre_hm is not None) or (pre_img is not None):
        feats = self.imgpre2feats(x, pre_img, pre_hm)
      else:
        feats = self.img2feats(x)
      
      out = []
      if self.opt.model_output_list:
        for s in range(self.num_stacks):
          z = []
          for head in sorted(self.heads):
            z.append(self.__getattr__(head)(feats[s]))
          out.append(z)
      else:
        for s in range(self.num_stacks): # zero for GRU model
          if self.opt.is_recurrent:

            input_len = int(len(feats[s]) / batch_size)
            inp = feats[s].view(batch_size, input_len, feats[s].size(1), feats[s].size(2), feats[s].size(3))
            feat_res = [feats[s].size(2), feats[s].size(3)]
            hm = torch.zeros(batch_size, 1, inp.size(3), inp.size(4)).cuda()
            h = None
            pre_hms = []
            prev_sim_map = None
            masked_sim_maps = []
            prev_feat = None
            # process a batch of frames one by one
            for i in range(inp.size(1)):
              curr_step = inp[:, i, :, :, :]
              if self.opt.pre_hm:
                curr_step = torch.cat((curr_step, hm), 1)
              intermediate_outputs, layer_reset_list, layer_update_list, last_output = self.conv_gru(curr_step.unsqueeze(1), h)
              h = last_output
              feats[s] = last_output[-1:][0]

              if self.opt.random_walk and 'rw_labels' in x[i]:
                # Downsample the memory state
                if self.opt.pool_kernel > 1:
                  pad = math.floor(self.opt.pool_kernel / 2)
                  stride = pad + 1
                  curr_feat = F.avg_pool2d(feats[s], self.opt.pool_kernel, stride, pad)
                else:
                  curr_feat = feats[s]
                
                if self.rw_head is not None:
                  curr_feat = self.rw_head(curr_feat)
                
                # L2 normalization of the node states
                curr_feat = F.normalize(curr_feat, dim=1, p=2)

                # Start Random Walk from the second frame 
                if prev_feat is not None:
                  # Compute edge weights
                  A = self.local_rw.forward(query=prev_feat, keys=curr_feat)
                  A = rearr(A, 'b n h w -> b (h w) n')

                  # Get indices of the active trajectories (the object was already seen in the past)
                  valid_inds = x[i]['valid_inds'][..., None].float().to(A.device)
                  #Get node indices of all trajectories
                  rw_inds = x[i]['rw_inds']
                  iidx = torch.arange(rw_inds.shape[0])[:, None]
                  if prev_sim_map is not None:
                    # Make a step of the Random Walk (only for active trajectories)
                    A = torch.matmul(prev_sim_map, A) * valid_inds + I * (1 - valid_inds)
                  else:
                    # In the second frame intitalize trajectories that are visible in the first frame
                    A = A[iidx, rw_inds]
                    I = torch.eye(A.shape[-1], dtype=A.dtype, device=A.device)[None].repeat(A.shape[0], 1, 1)
                    I = I[iidx, rw_inds]
                    A = A * valid_inds + I * (1 - valid_inds)

                  # Set walker state to 0 around the ground truht object centers to remove hard negatives
                  heatmaps = x[i]['rw_labels']
                  gt_centers = heatmaps.argmax(-1)
                  heatmaps = heatmaps.scatter(-1, gt_centers.unsqueeze(2), 0)
                  masks = torch.pow(1 - heatmaps, 4)
                  A_masked = A.clone()
                  A_masked *= masks
                  A_masked /= A_masked.sum(-1).unsqueeze(-1)
                  A /= A.sum(-1).unsqueeze(-1)
                  A = A * valid_inds + I * (1 - valid_inds)
                  masked_sim_maps.append(A_masked)

                  prev_sim_map = A
                  
                prev_feat = curr_feat

              z = {}
              z = self.apply_heads(feats[s], z)
              out.append(z)

              pre_hms.append(hm)
              if self.opt.predicted_hm:
                dets = generic_decode(z, K=self.opt.K, opt=self.opt, do_sigmoid=True)
                hm = self.generate_hm_from_det(dets, inp.size(3), inp.size(4))
                hm = torch.from_numpy(hm).cuda()
              elif self.opt.pre_hm and i < (inp.size(1) - 1):
                if type(x) == list:
                  hm = nn.functional.interpolate(x[i + 1]['pre_hm'], size=(inp.size(3), inp.size(4)), mode="bilinear")
                else:
                  hm = nn.functional.interpolate(pre_hm, size=(inp.size(3), inp.size(4)), mode="bilinear")
          else:
            z = self.apply_heads(feats[s], {})
          
            out.append(z)
            pre_hms = [pre_hm]

      return out, pre_hms, x, masked_sim_maps

    def _ignore_region(self, region, ignore_val=1):
      np.maximum(region, ignore_val, out=region)

    def generate_hm_from_det(self, dets, hm_h, hm_w):
      pre_hms = np.zeros((len(dets['bboxes']), 1, hm_h, hm_w), dtype=np.float32)

      for b in range(len(dets['bboxes'])):
        frame_dets = dets['bboxes'][b]
        frame_scores = dets['scores'][b]
        frame_visibilities = None
        if self.opt.visibility:
          frame_visibilities = dets['visibility'][b]
        count_vis = 0
        for i, bbox in enumerate(frame_dets):
          if frame_scores[i] < self.opt.pre_thresh:
            continue
          if frame_visibilities is not None and frame_visibilities[i] >= self.opt.visibility_thresh_eval:
            continue
          h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
          max_rad = 1
          if (h > 0 and w > 0):
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius)) 
            max_rad = max(max_rad, radius)
            ct = np.array(
              [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            conf = 1
            
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(pre_hms[b][0], ct_int, radius, k=conf)
            count_vis += 1
        # print("%d out of %d visualized in batch %d" % (count_vis, len(frame_dets), b))

      return pre_hms
