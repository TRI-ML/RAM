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

from ..datasets.lacater_tracking import LACATERTracking

class LACATERMovingTracking(LACATERTracking):
  num_categories = 2
  dataset_folder = 'la_cater_moving'
  default_resolution = [224, 320]
  class_name = ['Object', 'Snitch']
  cat_ids = {1:1, 2:2}
  max_objs = 16
  def __init__(self, opt, split, rank=None):
    data_dir = os.path.join(opt.data_dir, self.dataset_folder)
    img_dir = data_dir
    ann_file_ = split
    self.datase_path = img_dir + '/' + split + '_data'
    img_dir += '/' + split + '_data/frames/'
    ann_path = os.path.join(
      data_dir, 'annotations', '{}_data.json'.format(
        ann_file_))
    self.images = None
    super(LACATERMovingTracking, self).__init__(opt, split, ann_path, img_dir)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def __len__(self):
    return self.num_samples

  def _to_float(self, x):
    return float("{:.2f}".format(x))
