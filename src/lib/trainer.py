from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time
import torch
import torch.nn.functional as F
import numpy as np
import random
from progress.bar import Bar

from torch.nn.parallel import DistributedDataParallel as DDP
from utils.utils import AverageMeter

from model.losses import FastFocalLoss, RegWeightedL1Loss
from model.losses import BinRotLoss, WeightedBCELoss, WeightedBCELossNoLogits
from model.utils import _sigmoid
import torch.nn as nn

class GenericLoss(torch.nn.Module):
  def __init__(self, opt):
    super(GenericLoss, self).__init__()
    self.crit = FastFocalLoss(opt=opt)
    self.crit_reg = RegWeightedL1Loss()
    if 'rot' in opt.heads:
      self.crit_rot = BinRotLoss()
    if 'nuscenes_att' in opt.heads:
      self.crit_nuscenes_att = WeightedBCELoss()
    self.crit_gates = WeightedBCELossNoLogits()
    self.crit_new_classifier = WeightedBCELoss()
    if opt.random_walk:
      self.rw_loss_gamma = 2

    self.opt = opt

  def _sigmoid_output(self, output):
    if 'hm' in output:
      output['hm'] = _sigmoid(output['hm'])
    if 'visibility' in output:
      output['visibility'] = _sigmoid(output['visibility'])
    if 'hm_hp' in output:
      output['hm_hp'] = _sigmoid(output['hm_hp'])
    if 'dep' in output:
      output['dep'] = 1. / (output['dep'].sigmoid() + 1e-6) - 1.
    return output

  def forward(self, outputs, batch, input_len=None, 
          frame_ind=1, rw_labels=None, rw_mask=None, rw_inds=None, rw_mask_indices=None, rw_mask_weights=None, sim_map=None, prev_sim_map=None):
    opt = self.opt
    losses = {head: 0 for head in opt.heads}
    acc = None

    if opt.random_walk:
      losses['random_walk'] = torch.zeros(1).cuda()
    if opt.sup_centeroverlap:
      losses['overlap'] = torch.zeros(1).cuda()

    per_batch_hm_loss = torch.zeros(len(batch['hm']), 1)
    rw_loss1 = None
    rw_loss2 = None

    for s in range(opt.num_stacks):
      output = outputs[s]
      output = self._sigmoid_output(output)

      zero_neg = False

      if sim_map is not None:
        values = sim_map
        num_pos = rw_mask.sum()
        classification_labels = rw_labels.argmax(-1)
        fix_weights = (1 - values) ** self.rw_loss_gamma
        logits = torch.log(values + 1e-20)
        logits_focal = fix_weights * logits
        if self.opt.sup_centeroverlap:
          mask_sum = rw_mask_weights.sum()
          if mask_sum != 0:
            selected_vals = values.gather(-1, rw_mask_indices.long())
            over_loss = self.opt.overlapmult * (selected_vals * rw_mask_weights).sum() / mask_sum
            losses['overlap'] += over_loss
        rw_loss = F.nll_loss(logits_focal.flatten(0, 1), classification_labels.flatten(), reduction='none')
        rw_loss = rw_loss.view(logits.shape[0], -1) * rw_mask

        acc = (torch.argmax(logits, dim=-1) == torch.argmax(rw_labels, dim=-1)).float() * rw_mask
        acc = acc.sum() / (num_pos + 1e-4)

        rw_loss1 = rw_loss[0].sum() / (rw_mask[0].sum() + 1e-4)
        rw_loss2 = rw_loss[1].sum() / (rw_mask[1].sum() + 1e-4)
        rw_loss = rw_loss.sum() / (num_pos + 1e-4)
        losses['random_walk'] += rw_loss

      if 'hm' in output:
        tot_hm_loss, per_batch_hm_loss = self.crit(
          output['hm'], batch['hm'], batch['ind'], 
          batch['mask'], batch['cat'], zero_neg)
        losses['hm'] += tot_hm_loss / opt.num_stacks

      if 'visibility' in output:
        tot_vis_loss, per_batch_vis_loss = self.crit(
          output['visibility'], batch['visibility'], batch['visibility_ind'], 
          batch['visibility_mask'], batch['visibility_cat'])
        losses['visibility'] += tot_vis_loss / opt.num_stacks
      
      regression_heads = [
        'reg', 'wh', 'ltrb', 'ltrb_amodal', 'hps', 
        'dep', 'dim', 'amodel_offset', 'velocity']
      if self.opt.sup_reg:
        regression_heads += ["tracking"]

      for head in regression_heads:
        if head in output:
          losses[head] += self.crit_reg(
            output[head], batch[head + '_mask'],
            batch['ind'], batch[head]) / opt.num_stacks
      
      if 'hm_hp' in output:
        losses['hm_hp'] += self.crit(
          output['hm_hp'], batch['hm_hp'], batch['hp_ind'], 
          batch['hm_hp_mask'], batch['joint']) / opt.num_stacks
        if 'hp_offset' in output:
          losses['hp_offset'] += self.crit_reg(
            output['hp_offset'], batch['hp_offset_mask'],
            batch['hp_ind'], batch['hp_offset']) / opt.num_stacks
        
      if 'rot' in output:
        losses['rot'] += self.crit_rot(
          output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'],
          batch['rotres']) / opt.num_stacks

      if 'nuscenes_att' in output:
        losses['nuscenes_att'] += self.crit_nuscenes_att(
          output['nuscenes_att'], batch['nuscenes_att_mask'],
          batch['ind'], batch['nuscenes_att']) / opt.num_stacks

    losses['tot'] = 0
    if 'random_walk' in losses:
      losses['tot'] += opt.rw_weight * losses['random_walk']
    if 'overlap' in losses:
      losses['tot'] += losses['overlap']
    for head in opt.heads:
      losses['tot'] += opt.weights[head] * losses[head]
    
    return losses['tot'], losses, per_batch_hm_loss, acc, [rw_loss1, rw_loss2]

class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss

  def forward(self, batch, batch_size=1, stream=False, pre_gru_state=None, eval_mode=False):
    """ Forward function

    Parameters
    ----------
    batch: list or dict
      Dict: legacy input format from CenterNet. To be deprecated.
      List: A list of dictionary with input and annotations.

    batch_size: int
      Batch size

    stream: bool
      Whether the model is evaluated in steam mode.

    pre_gru_state: torch.Tensor
      Previous ConvGRU state vector.

    eval_mode: bool
      whether it is used for evaluation.
    """

    per_batch_mean_hm_loss = torch.zeros(batch_size, 1).cuda()
    pre_batch_rw_losses = []

    if type(batch) != list:
      pre_img = batch['pre_img'] if 'pre_img' in batch else None
      pre_hm = batch['pre_hm'] if 'pre_hm' in batch else None
      outputs, pre_hm, _ = self.model(batch['image'], pre_img, pre_hm, batch_size)
      loss, loss_stats = self.loss([outputs[-1]], batch)
    else:
      if eval_mode:
        pre_img = []
        pre_hm = []
        for i in range(len(batch) - 1):
          pre_img.append(batch[i]['image'])
          pre_hm.append(batch[i]['pre_hm'])
      else:
        pre_img = torch.zeros(5, 5)
        pre_hm = None

      sim_maps = None
      if stream and eval_mode:
        outputs, output_gru_state, _, _ = self.model.step(batch, pre_gru_state)
      else:
        outputs, pre_hm, batch, sim_maps = self.model(batch, pre_img, pre_hm, batch_size)
      loss = None
      stats = []

      rw_labels = None
      rw_mask = None
      rw_inds = None
      rw_mask_indices = None
      rw_mask_weights = None
      sim_map = None
      past_sim_map = None
      acc = 0
      if not eval_mode:
        for i in range(len(batch)):
          if i > 0:
            if len(sim_maps) > 0:
              rw_labels = batch[i]['rw_labels']
              rw_mask = batch[i]['rw_mask']
              rw_inds = batch[i]['rw_inds']
              rw_mask_indices = batch[i]['rw_mask_indices']
              rw_mask_weights = batch[i]['rw_mask_weights']
              sim_map = sim_maps[i - 1]
          loss_step, loss_stats, per_batch_hm_loss, acc_step, per_batch_rw = self.loss([outputs[i]], batch[i], len(batch), i, rw_labels, rw_mask, rw_inds, rw_mask_indices, rw_mask_weights, sim_map, past_sim_map)
          pre_batch_rw_losses.append(per_batch_rw)
          per_batch_mean_hm_loss += per_batch_hm_loss
          stats.append(loss_stats)
          past_sim_map = sim_map
          if loss is None:
            loss = loss_step
          else:
            loss += loss_step
          if acc_step is not None:
            acc += acc_step
        loss /= len(batch)
        acc /= len(batch)
        per_batch_mean_hm_loss /= len(batch)
      else:
          loss_step, loss_stats, _, _, _ = self.loss([copy.deepcopy(outputs[-1])], batch[-1])
          stats.append(loss_stats)
          loss = loss_step

    if stream:
      return outputs, loss, loss_stats, batch, output_gru_state
    return outputs, loss, loss_stats, pre_hm, batch, per_batch_mean_hm_loss, sim_maps, acc, pre_batch_rw_losses

def get_losses(opt):
  loss_order = ['hm', 'wh', 'reg', 'ltrb', 'hps', 'hm_hp', \
      'hp_offset', 'dep', 'dim', 'rot', 'amodel_offset', \
      'ltrb_amodal', 'nuscenes_att', 'velocity', 'visibility']
  if opt.sup_reg:
    loss_order += ['tracking']
  loss_states = ['tot'] + [k for k in loss_order if k in opt.heads]
  if opt.random_walk:
    loss_states += ['random_walk']    
  if opt.sup_centeroverlap:
    loss_states += ['overlap']
  loss = GenericLoss(opt)
  return loss_states, loss

class Trainer(object):
  def __init__(
    self, opt, model, optimizer=None):
    self.opt = opt
    self.optimizer = optimizer
    self.loss_stats, self.loss = get_losses(opt)
    self.model_with_loss = ModleWithLoss(model, self.loss)

  def set_device(self, gpus, chunk_sizes, device):
    self.rank = device
    self.model_with_loss = DDP(self.model_with_loss, device_ids=[device], find_unused_parameters=True)

  def run_epoch(self, phase, epoch, data_loader):
    model_with_loss = self.model_with_loss
    if phase == 'train':
      model_with_loss.train()
    else:
      if len(self.opt.gpus) > 1:
        model_with_loss = self.model_with_loss.module
      model_with_loss.eval()
      torch.cuda.empty_cache()

    opt = self.opt
    results = {}
    data_time, batch_time = AverageMeter(), AverageMeter()
    avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
    if opt.random_walk:
      avg_loss_stats['acc'] = AverageMeter()
    len_data = len(data_loader)
    num_iters = len_data
    if phase == 'train':
      num_iters = len_data if opt.num_iters < 0 else opt.num_iters
    else:
      num_iters = len_data if opt.num_val_iters < 0 else opt.num_val_iters

    if self.rank == 0:
      bar = Bar('{}/{}'.format(opt.task, opt.exp_id), max=num_iters)
    end = time.time()
    vids = []
    step = int(round(len_data / opt.num_vis_vids))
    real_iter = iter(data_loader)
    for iter_id, batch in enumerate(data_loader):
      if iter_id >= num_iters:
        break

      data_time.update(time.time() - end)

      if type(batch) == list:
        batch_size = batch[0]['image'].size(0)
        for i in range(len(batch)):
          for k in batch[i]:
            if k != 'meta' and k!= 'gt_det' and k!= 'image_path':
              batch[i][k] = batch[i][k].to(device=opt.device, non_blocking=True)     
      else:
        for k in batch:
          if k != 'meta' and k!= 'gt_det' and k!= 'image_path':
            batch[k] = batch[k].to(device=opt.device, non_blocking=True)

      outputs, loss, loss_stats, pre_hms, batch, per_batch_mean_hm_loss, sim_maps, acc, pre_batch_rw_losses = model_with_loss(batch, batch_size)  
      output = outputs[-1]
      prev_output = outputs[0]
        
      loss = loss.mean()
      if phase == 'train':
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

      batch_time.update(time.time() - end)
      end = time.time()

      if self.rank == 0:
        Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
          epoch, iter_id, num_iters, phase=phase,
          total=bar.elapsed_td, eta=bar.eta_td)
        for l in avg_loss_stats:
          if l == 'acc':
            continue
          avg_loss_stats[l].update(
            loss_stats[l].mean().item(), batch_size)
          Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
        if acc is not None and 'acc' in avg_loss_stats and type(acc) is not float:
          avg_loss_stats['acc'].update(acc.mean().item(), batch_size)
          Bar.suffix = Bar.suffix + '|Acc {:.4f} '.format(avg_loss_stats['acc'].avg)  
        Bar.suffix = Bar.suffix + '|Data {dt.val:.3f}s({dt.avg:.3f}s) ' \
          '|Net {bt.avg:.3f}s'.format(dt=data_time, bt=batch_time)
        if opt.print_iter > 0: # If not using progress bar
          if iter_id % opt.print_iter == 0:
            print('{}/{}| {}'.format(opt.task, opt.exp_id, Bar.suffix)) 
        else:
          bar.next()
      
      del output, loss, loss_stats
    
    if self.rank == 0:
      bar.finish()
    
    ret = {k: v.avg for k, v in avg_loss_stats.items()}
    if self.rank == 0:
      ret['time'] = bar.elapsed_td.total_seconds() / 60.
      ret['vids'] = vids

    return ret, results
  
  def val(self, epoch, data_loader):
    return self.run_epoch('val', epoch, data_loader)

  def train(self, epoch, data_loader):
    return self.run_epoch('train', epoch, data_loader)
