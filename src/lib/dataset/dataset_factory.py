from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os

from .datasets.kitti_tracking import KITTITracking
from .datasets.pd_tracking import PDTracking
from .datasets.custom_dataset import CustomDataset
from .datasets.nuscenes_tracking import nuScenesTracking
from .datasets.lacater_tracking import LACATERTracking
from .datasets.lacater_moving_tracking import LACATERMovingTracking

dataset_factory = {
  'custom': CustomDataset,
  'kitti_tracking': KITTITracking,
  'pd_tracking': PDTracking,
  'nuscenes_tracking': nuScenesTracking,
  'la_cater': LACATERTracking,
  'la_cater_moving': LACATERMovingTracking
}


def get_dataset(dataset):
  return dataset_factory[dataset]
