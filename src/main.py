from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import glob
import subprocess
import json

import torch
import torch.utils.data
from opts import opts
from model.model import create_model, load_model, save_model
from logger import Logger
from dataset.dataset_factory import get_dataset
from dataset.datasets.pd_tracking import PDTracking
from trainer import Trainer
from dataset.joint_loader import JointLoader

import torch.multiprocessing as mp
import torch.distributed as dist


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'

    # initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def get_optimizer(opt, model):
  if opt.optim == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
  elif opt.optim == 'sgd':
    print('Using SGD')
    optimizer = torch.optim.SGD(
      model.parameters(), opt.lr, momentum=0.9, weight_decay=0.0001)
  else:
    assert 0, opt.optim
  return optimizer

def launch_test(opt):
  if opt.dataset != 'joint':
    dataset = opt.dataset
  else:
    dataset = opt.dataset1
  dataset_version = opt.dataset_version
  if dataset == 'kitti_tracking':
    dataset_version = 'val_half'
  if opt.random_walk:
    if dataset == 'kitti_tracking' or dataset == 'pd_tracking':
      command = "python test.py tracking --exp_id %s --dataset %s --dataset_version %s --track_thresh 0.4 --resume --is_recurrent --debug 4 --gru_filter_size %d --input_len %d --write_to_file --num_gru_layers %d --debug 4 --random_walk --rw_head_depth 2 --pool_kernel 3 --max_age 16 --rw_score_thresh 0.05 --local_rw_r 0.2 --new_thresh 0.5" % \
      (opt.exp_id, dataset, dataset_version, opt.gru_filter_size, opt.input_len, opt.num_gru_layers)
    else:
      command = "python test.py tracking --exp_id %s --dataset %s --dataset_version %s --track_thresh 0.4 --resume --is_recurrent --debug 4 --gru_filter_size %d --input_len %d --num_gru_layers %d --debug 4 --random_walk --rw_head_depth 2 --pool_kernel %d --max_age 300 --rw_score_thresh 0.005 --local_rw_r %f --new_thresh 0.5 --max_vids 100" % \
      (opt.exp_id, dataset, dataset_version, opt.gru_filter_size, opt.input_len, opt.num_gru_layers, opt.pool_kernel, opt.local_rw_r)
  else:
    command = "python test.py tracking --exp_id %s --dataset %s --dataset_version %s --track_thresh 0.4 --resume --is_recurrent --debug 4 --gru_filter_size %d --input_len %d --write_to_file --num_gru_layers %d --test_with_loss --flip 0 --visibility_thresh_eval %f --debug 4" % \
    (opt.exp_id, dataset, dataset_version, opt.gru_filter_size, opt.input_len, opt.num_gru_layers, opt.visibility_thresh_eval)
  if dataset == 'kitti_tracking':
     command += ' --max_out_age 4'
  if opt.input_len > 3:
    command += ' --stream_test'
  if opt.pre_hm:
    command += ' --pre_hm'
  if opt.predicted_hm:
    command += ' --predicted_hm'
  if opt.no_color_aug:
    command += ' --no_color_aug'
  if opt.visibility:
    command += ' --visibility'
  if opt.ltrb_amodal:
    command += ' --ltrb_amodal'
  if opt.sup_reg:
    command += ' --sup_reg'
  print("Running eval: %s" % command)

  os.system(command)

def train(rank, opt, Dataset):
  setup(rank, len(opt.gpus))

  logger = Logger(opt)

  print('Creating model...')
  model = create_model(opt.arch, opt.heads, opt.head_conv, opt=opt)
  if opt.freeze_backbone:
    model.freeze_backbone()
  if opt.freeze_gru:
    model.freeze_gru()
  torch.cuda.set_device(rank)
  model.cuda(rank)

  optimizer = get_optimizer(opt, model)
  start_epoch = 0
  if opt.load_model != '':
    model, optimizer, start_epoch = load_model(
      model, opt.load_model, opt, optimizer)
  
  print('Setting up train data...')
  train_sampler = None
  if opt.dataset != 'joint':
    train_dataset = Dataset(opt, 'train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	train_dataset,
    	num_replicas=len(opt.gpus),
    	rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, sampler=train_sampler
    )
  else:
    dataset1 = Dataset(opt, 'train')
    dataset1.input_len = 2
    batch_size = opt.batch_size
    if opt.dataset1 == 'nuscenes_tracking':
      batch_size = 4
    if opt.dataset1 == 'kitti_tracking':
      dataset1.input_len = 12
      
    if opt.dataset1 == 'nuscenes_tracking':
      dataset1.input_len = 16

    train_sampler1 = torch.utils.data.distributed.DistributedSampler(
    	dataset1,
    	num_replicas=len(opt.gpus),
    	rank=rank
    )
    train_loader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, sampler=train_sampler1
    )

    dataset_class2 = get_dataset(opt.dataset2)
    dataset2 = dataset_class2(opt, 'train')
    if opt.ignore_wh_insym:
      dataset2.wh_weight = 0
    train_sampler2 = torch.utils.data.distributed.DistributedSampler(
    	dataset2,
    	num_replicas=len(opt.gpus),
    	rank=rank
    )
    train_loader2 = torch.utils.data.DataLoader(
        dataset2, batch_size=opt.batch_size, shuffle=False,
        num_workers=2, pin_memory=True, sampler=train_sampler2
    )

    train_sampler = [train_sampler1, train_sampler2]
    train_loaders = [train_loader1, train_loader2]

    train_loader = JointLoader(train_loaders[0], train_loaders[1])

  trainer = Trainer(opt, model, optimizer)
  trainer.set_device(opt.gpus, opt.chunk_sizes, rank)

  print('Starting training...')
  for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    if train_sampler is not None:
      if type(train_sampler) != list:
        train_sampler.set_epoch(epoch)
      else:
        for sampler in train_sampler:
          sampler.set_epoch(epoch)

    mark = epoch if opt.save_all else 'last'
    log_dict_train, _ = trainer.train(epoch, train_loader)
    if rank == 0:
      logger.write('epoch: {} |'.format(epoch))
      stats = {}
      for k, v in log_dict_train.items():
        if k == "vids":
          continue
        logger.scalar_summary('train_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
      if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
        log_outputs = {}
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), 
                    epoch, model, optimizer)

        torch.cuda.empty_cache()
        launch_test(opt)
      else:
        save_model(os.path.join(opt.save_dir, 'model_last.pth'), 
                  epoch, model, optimizer)
      logger.write('\n')
      if epoch in opt.save_point:
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), 
                  epoch, model, optimizer)
    
    if ((epoch + 1) % opt.lr_step[0]) == 0:
      lr = opt.lr * 0.1
      print('Drop LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr  

    if (epoch % opt.lr_step[0]) == 0:
      lr = opt.lr
      print('Raise LR to', lr)
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr  
  
  save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)

  logger.close()
  cleanup()


def main(opt):
  if opt.dataset != 'joint':
    Dataset = get_dataset(opt.dataset)
  else:
    Dataset = get_dataset(opt.dataset1)
  opt = opts().update_dataset_info_and_set_heads(opt, Dataset)
  print(opt)
  if not opt.not_set_cuda_env:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  
  torch.manual_seed(opt.seed)
  torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  
  mp.spawn(train, nprocs=len(opt.gpus), args=(opt, Dataset,))   

if __name__ == '__main__':
  opt = opts().parse()
  main(opt)
