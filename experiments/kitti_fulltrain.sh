# Initial model pre-trained on PD: 
# Resulting model trained on KITTI full train: 

cd src
# train
python main.py tracking --exp_id ram_kittifull --occlusion_thresh 0.15 --visibility_thresh 0.05 --dataset joint --dataset1 kitti_tracking --dataset2 pd_tracking --dataset_version train --same_aug_pre --hm_disturb 0.0 --lost_disturb 0.0 --fp_disturb 0.0 --gpus 0,1,2,3,4,5,6,7 --batch_size 2 --load_model ../models/ram_pd.pth --val_intervals 10 --is_recurrent --input_len 16 --pre_thresh 0.4 --hm_weight 0.5 --num_iter 5000 --num_epochs 6 --lr_step 7  --random_walk --rw_head_depth 2 --pool_kernel 3 --sup_centeroverlap --sup_reg --rw_temp 0.1
# test
CUDA_VISIBLE_DEVICES=1 python test.py tracking --exp_id ram_kittifull --dataset kitti_tracking --dataset_version test --track_thresh 0.4 --resume --is_recurrent --input_len 16 --random_walk --rw_head_depth 2 --pool_kernel 3 --max_age 16 --local_rw_r 0.2  --stream_test --new_thresh 0.5 --max_out_age 4 --sup_reg --flip_test --trainval