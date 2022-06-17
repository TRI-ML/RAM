# Getting Started

This document provides tutorials to train and evaluate RAM. Before getting started, make sure you have finished [installation](INSTALL.md) and [dataset setup](DATA.md).

## Benchmark evaluation

### PD

To test our pretrained model on the validation set of PD, download the [model](https://tri-ml-public.s3.amazonaws.com/github/permatrack/ram_pd.pth), copy it to `$RAM_ROOT/models/`, and run

~~~
cd $RAM_ROOT/src
python test.py tracking --exp_id ram_pd --dataset pd_tracking --dataset_version val --track_thresh 0.4 --load_model ../models/ram_pd.pth --is_recurrent --input_len 16 --random_walk --rw_head_depth 2 --pool_kernel 3 --max_age 16 --local_rw_r 0.2 --stream_test --new_thresh 0.5 --sup_reg
~~~

This will give a Track mAP of `71.96` if set up correctly. You can append `--debug 4` to the above command to visualize the predictions.

### KITTI Tracking

To test the tracking performance on the validation set of KITTI with our pretrained model, download the [model](https://tri-ml-public.s3.amazonaws.com/github/permatrack/ram_kittihalf.pth), copy it to `$RAM_ROOT/models/`, and run

~~~
python test.py tracking --exp_id ram_kittihalf --dataset kitti_tracking --dataset_version val_half --track_thresh 0.4 --load_model ../models/ram_kittihalf.pth --is_recurrent --input_len 16 --debug 4 --random_walk --rw_head_depth 2 --pool_kernel 3 --max_age 16 --local_rw_r 0.2  --stream_test --new_thresh 0.5 --sup_reg --max_out_age 4
~~~

### LA-CATER

To test the tracking performance on the test set of LA-CATER, download the [model](https://tri-ml-public.s3.amazonaws.com/github/permatrack/ram_lacater_stage2.pth), copy it to `$RAM_ROOT/models/`, and run

~~~
python test.py tracking --exp_id ram_lacater_stage2 --dataset la_cater --dataset_version train --track_thresh 0.4 --load_model ../models/ram_lacater_stage2.pth --is_recurrent --debug 4 --input_len 70 --num_gru_layers 1 --debug 4 --random_walk --rw_head_depth 2 --pool_kernel 1 --max_age 300 --rw_score_thresh 0.005 --local_rw_r 0.1 --new_thresh 0.5 --stream_test --sup_reg  --trainval
~~~

### LA-CATER-Moving

To test the tracking performance on the test set of LA-CATER-Moving, download the [model](https://tri-ml-public.s3.amazonaws.com/github/permatrack/ram_lacater_moving_stage2.pth), copy it to `$RAM_ROOT/models/`, and run

~~~
python test.py tracking --exp_id ram_lacater_moving_stage2 --dataset la_cater_moving --dataset_version train --track_thresh 0.4 --load_model ../models/ram_lacater_moving_stage2.pth --is_recurrent --debug 4 --input_len 70 --num_gru_layers 1 --debug 4 --random_walk --rw_head_depth 2 --pool_kernel 1 --max_age 300 --rw_score_thresh 0.005 --local_rw_r 0.1 --new_thresh 0.5 --stream_test --sup_reg  --trainval
~~~


## Training
We have packed all the training scripts in the [experiments](../experiments) folder.
Each model is trained on 8 Tesla V100 GPUs with 32GB of memory.
If the training is terminated before finishing, you can use the same command with `--resume` to resume training. It will found the latest model with the same `exp_id`.
All experiments rely on existing pretrained models, we provide the links to the corresponding models directly in the training scripts.