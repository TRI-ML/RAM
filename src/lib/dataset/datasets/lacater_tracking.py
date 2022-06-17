from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
import numpy as np
import torch
import json
import cv2
import os
import math

from ..video_dataset import VideoDataset

class LACATERTracking(VideoDataset):
  num_categories = 2
  dataset_folder = 'la_cater'
  default_resolution = [224, 320]
  class_name = ['Object', 'Snitch']
  cat_ids = {1:1, 2:2}
  max_objs = 16
  def __init__(self, opt, split, ann_path=None, img_dir=None, rank=None):
    data_dir = os.path.join(opt.data_dir, self.dataset_folder)
    if ann_path is None:
      img_dir = data_dir
      ann_file_ = split
      self.datase_path = img_dir + '/' + split + '_data'
      img_dir += '/' + split + '_data/frames/'
      ann_path = os.path.join(
        data_dir, 'annotations', '{}_data.json'.format(
          ann_file_))
    self.images = None
    super(LACATERTracking, self).__init__(opt, split, ann_path, img_dir)

    self.box_size_thresh = [0, 0]

    self.alpha_in_degree = False
    self.crop = 0
    self.depth_scale = 1
    self.dep_mask = 0
    self.dim_mask = 1
    self.rot_mask = 0
    self.amodel_offset_mask = 0
    self.ignore_amodal = True
    self.num_samples = len(self.images)
    self.exp_id = opt.exp_id
    self.max_vids = opt.max_vids
    if opt.const_v_over_occl:
      self.const_v_over_occl = True

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def save_results(self, results, save_dir):
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)
    save_dir += '/results'
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)

    vid_count = 0
    for video in self.coco.dataset['videos']:
      formattted_results = []
      other_boxes = []
      visibility = []
      if self.max_vids != 0 and vid_count == self.max_vids:
        break
      vid_count += 1
      video_id = video['id']
      images = self.video_to_images[video_id]
      
      first_snitch = None
      for image_info in images:
        img_id = image_info['id']
        if not (img_id in results):
          continue
        snitch_box = None
        snitch_score = -1
        visible = True
        other_boxes_frame = []
        for i in range(len(results[img_id])):
          item = results[img_id][i]
          category_id = item['class']
          bbox = [item['bbox'][0].item(), item['bbox'][1].item(), item['bbox'][2].item(), item['bbox'][3].item()]

          if category_id == 1:
            if item['age'] == 1:
              other_boxes_frame.append(bbox)
            continue

          if first_snitch is None:
            first_snitch = bbox

          if snitch_box is None:
            snitch_box = bbox
            snitch_score = item['score'].item()
            if item['age'] > 1:
              visible = False
          elif item['score'] >= snitch_score:
            snitch_box = bbox
            snitch_score = item['score'].item()
            if item['age'] > 1:
              visible = False

        if snitch_box is not None:
          formattted_results.append(snitch_box)
        elif len(formattted_results) > 0:
          visible = False
          formattted_results.append(formattted_results[-1])
        else:
          formattted_results.append([])
        visibility.append(visible)
        other_boxes.append(other_boxes_frame)

      for i in range(len(formattted_results)):
        if len(formattted_results[i]) == 0:
          formattted_results[i] = first_snitch
      print(save_dir + '/%s_bb.json' % video['file_name'])
      json.dump(formattted_results, open(save_dir + '/%s_bb.json' % video['file_name'], 'w'))
      json.dump(other_boxes, open(save_dir + '/%s_otherbb.json' % video['file_name'], 'w'))
      json.dump(visibility, open(save_dir + '/%s_visibility.json' % video['file_name'], 'w'))

  def run_eval(self, results, save_dir, write_to_file=False, dataset_version=None):
    self.save_results(results, save_dir)

    command = 'python lib/utils/post_process_lacter.py --exp %s' % self.exp_id
    os.system(command)

    command = 'python laceter_analysis.py ' + \
              '--exp %s' % self.exp_id + \
              ' --dataset_path %s' % self.datase_path + \
              ' --max_vids %d' % self.max_vids

    if write_to_file:
      print("Writing to file")
      command += ' > ../exp/tracking/{}/eval_out.txt'.format(self.exp_id)
    print(command)
    os.system(command)

  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    return float("{:.2f}".format(x))
