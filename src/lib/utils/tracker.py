import numpy as np
import cv2
from sklearn.utils.linear_assignment_ import linear_assignment
from numba import jit
import copy
import math
import torch.nn.functional as F
import torch
from .image import transform_preds_with_trans
from scipy.spatial.distance import cdist
from utils.image import gaussian_radius, draw_umich_gaussian

class Tracker(object):
  def __init__(self, opt):
    self.opt = opt
    self.alpha = 0.9
    self.frame_ind = 0
    self.reset()

  def init_track(self, results):
    for item in results:
      if item['score'] > self.opt.new_thresh:
        self.id_count += 1
        item['active'] = 1
        item['age'] = 1
        item['tracking_id'] = self.id_count
        if not ('ct' in item):
          bbox = item['bbox']
          item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        item['prev_ct'] = item['ct']
        item['v'] = [0, 0]
        self.tracks.append(item)

  def reset(self):
    self.full_tracks = {}
    self.id_count = 0
    self.tracks = []
    self.affinity = None
    self.frame_ind = 0

  def init_walk(self, feat_res, rw_downsample, track):
    best_ct = track['orig_ct']

    ct_int = [int(x / (rw_downsample)) for x in best_ct]
    slot_ind = ct_int[1] * (feat_res[1] // rw_downsample) + ct_int[0]
    walk = self.affinity[slot_ind]

    return walk

  def step(self, results, affinity=None, debugger=None, feat_res=None, raw_input_res=None, postproc_input_res=None, trans=None):
    N = len(results)
    M = len(self.tracks)
    self.affinity = affinity

    dets = np.array(
      [det['ct'] + det['tracking'] for det in results], np.float32) # N x 2
    dets_original = np.array([det['ct'] for det in results], np.float32) # N x 2
    track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
      (track['bbox'][3] - track['bbox'][1])) \
      for track in self.tracks], np.float32) # M
    track_cat = np.array([track['class'] for track in self.tracks], np.int32) # M
    item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * \
      (item['bbox'][3] - item['bbox'][1])) \
      for item in results], np.float32) # N
    item_cat = np.array([item['class'] for item in results], np.int32) # N
    tracks = np.array(
      [pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2

    dist = (((tracks.reshape(1, -1, 2) - \
              dets.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M

    invalid = ((dist > track_size.reshape(1, M)) + \
      (dist > item_size.reshape(N, 1)) + \
      (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
    dist = dist + invalid * 1e18
    
    if self.opt.hungarian:
      item_score = np.array([item['score'] for item in results], np.float32) # N
      dist[dist > 1e18] = 1e18
      matched_indices = linear_assignment(dist)
    else:
      matched_indices = greedy_assignment(copy.deepcopy(dist))
    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])]
    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])]
    
    if self.opt.hungarian:
      matches = []
      for m in matched_indices:
        if dist[m[0], m[1]] > 1e16:
          unmatched_dets.append(m[0])
          unmatched_tracks.append(m[1])
        else:
          matches.append(m)
      matches = np.array(matches).reshape(-1, 2)
    else:
      matches = matched_indices

    ret = []
    for m in matches:
      track = results[m[0]]
      track['tracking_id'] = self.tracks[m[1]]['tracking_id']
      track['age'] = 1
      track['crw_score'] = 1
      track['active'] = 1
      track['alive'] = self.tracks[m[1]]['alive'] + 1
      track['prev_ct'] = self.tracks[m[1]]['ct']      
      track['v'] = [x - y  for x,y in zip(track['ct'], track['prev_ct'])]
      track['walk'] = None
      ret.append(track)
    
    # Never used
    rw_downsample = math.floor(self.opt.pool_kernel / 2) + 1
    out_hm_all = None
    for i in unmatched_tracks:
      track = self.tracks[i]
      if track['age'] < self.opt.max_age:
        bbox = track['bbox']
        ct = track['orig_ct']
        
        if self.opt.random_walk:
          x_thresh = feat_res[1] * self.opt.border_thresh
          y_thresh = feat_res[0] * self.opt.border_thresh

          # Initialize the walk or make a step if already initialized
          if track['walk'] is None:
            track['walk'] = self.init_walk(feat_res, rw_downsample, track)
          else:
            track['walk']  = torch.matmul(track['walk'], affinity)

          heatmap = track['walk']
          heatmap = heatmap.view(feat_res[0] // rw_downsample, feat_res[1] // rw_downsample)

          rw_score = heatmap.max().item()
          track['crw_score'] = rw_score
          # Terminate trajectories with low confidence
          if rw_score < self.opt.rw_score_thresh:
            continue

          heatmap = heatmap.unsqueeze(0).unsqueeze(0)
          heatmap = F.interpolate(heatmap, size=feat_res, mode='bilinear')

          rw_center = torch.argmax(heatmap.squeeze())
          rw_ct_x = rw_center % feat_res[1]
          rw_ct_y = rw_center // feat_res[1]

          # Terminate trajectories that go out of view
          if (rw_ct_x <= x_thresh) or (rw_ct_x >= (feat_res[1] - x_thresh)) or (rw_ct_y <= y_thresh) or (rw_ct_y >= (feat_res[0] - y_thresh)):
            continue

          rw_ct = transform_preds_with_trans(np.array([rw_ct_x, rw_ct_y]).reshape(1, 2), trans).reshape(2)

          # Match updated Random Walk hypothesis with remaining detections
          if len(unmatched_dets) > 0:
            unmatched_centers = np.array([results[j]['ct'] for j in unmatched_dets], np.float32)
            all_centers = np.array([results[j]['ct'] for j in range(len(results))], np.float32)
            dist_occl = ((unmatched_centers - rw_ct) ** 2).sum(axis=1)
            dist_all = ((all_centers - rw_ct) ** 2).sum(axis=1)
            j = dist_occl.argmin()
            f = dist_all.argmin()
            new_track = results[unmatched_dets[j]]
            temp_track = results[f]
            if math.sqrt(dist_occl[j]) <= self.opt.disocl_match_thresh and new_track['class'] == track['class']:
              track['age'] = 1
              track['alive'] = 1
              track['ct'] = new_track['ct']
              track['bbox'] = new_track['bbox']
              track['v'] = [0, 0]
              track['walk'] = None
              del unmatched_dets[j]

          # If no match found, save Random Walk maximul likelihood hypothesis 
          if track['walk'] is not None:
            out_hm = heatmap.squeeze().detach().cpu().numpy()
            out_hm = cv2.resize(out_hm, (postproc_input_res[1], postproc_input_res[0]))
            out_hm = np.expand_dims(out_hm, axis=2)
            if postproc_input_res[0] > raw_input_res[0]:
              pad_h = postproc_input_res[0] - raw_input_res[0]
              pad_w = postproc_input_res[1] - raw_input_res[1]
              if pad_w > 0:
                out_hm = out_hm[int(round(pad_h / 2)): -int(round(pad_h / 2)), int(round(pad_w / 2)): -int(round(pad_w / 2)), :]
              else:
                out_hm = out_hm[int(round(pad_h / 2)): -int(round(pad_h / 2)), :, :]

            if out_hm_all is None:
              out_hm_all = out_hm
            else:
              out_hm_all += out_hm

            track['ct'] = [rw_ct[0], rw_ct[1]]
            b_w  = bbox[2] - bbox[0]
            b_h  = bbox[3] - bbox[1]
            track['bbox'] = [rw_ct[0] - b_w / 2, rw_ct[1] - b_h / 2, rw_ct[0] + b_w / 2, rw_ct[1] + b_h / 2]
            track['age'] += 1
            track['active'] = 1 # 0
            track['alive'] = 0
        else:
          track['age'] += 1
          track['active'] = 1 # 0
          track['alive'] = 0

          ct = track['ct']
          # v = track['v']
          v = [0, 0]
          track['bbox'] = [
            bbox[0] + v[0], bbox[1] + v[1],
            bbox[2] + v[0], bbox[3] + v[1]]
          track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
          
        ret.append(track)


    # Create tracks for all un-matched detections
    for i in unmatched_dets:
      track = results[i]
      if track['score'] > self.opt.new_thresh:
        self.id_count += 1
        track['tracking_id'] = self.id_count
        track['age'] = 1
        track['active'] =  1
        track['alive'] = 1
        track['crw_score'] = 1
        track['prev_ct'] = track['ct']
        track['v'] = [0, 0]
        track['walk'] = None
        ret.append(track)

    self.tracks = ret
    return copy.deepcopy(ret), out_hm_all

  @staticmethod
  def tlwh_to_xyah(tlwh):
    """Convert bounding box to format `(center x, center y, aspect ratio,
    height)`, where the aspect ratio is `width / height`.
    """
    ret = np.asarray(tlwh).copy()
    ret[:2] += ret[2:] / 2
    ret[2] /= ret[3]
    return ret

  @staticmethod
  def tlbr_to_tlwh(tlbr):
    ret = np.asarray(tlbr).copy()
    ret[2:] -= ret[:2]
    return ret

def greedy_assignment(dist, v_dist=None):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)
