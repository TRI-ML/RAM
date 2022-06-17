# Initial model pre-trained on PD: 
# Resulting model trained on LA-CATER: 

cd src
# first stage of training
python main.py tracking --exp_id ram_lacater_stage1 --occlusion_thresh 0.15 --visibility_thresh 0.05 --dataset la_cater --dataset_version train --same_aug_pre --hm_disturb 0.0 --lost_disturb 0.0 --fp_disturb 0.0 --gpus 0,1,2,3,4,5,6,7 --batch_size 2 --load_model ../models/ram_pd.pth --val_intervals 1 --is_recurrent --input_len 70 --pre_thresh 0.4 --hm_weight 0.5 --num_iter 1000 --num_epochs 8 --lr_step 4 --random_walk --rw_head_depth 2 --pool_kernel 1 --sup_reg --local_rw_r 0.1 --rw_temp 0.1
# fine-tuning on longer sequences with a frozen backbone
python main.py tracking --exp_id ram_lacater_stage2 --occlusion_thresh 0.15 --visibility_thresh 0.05 --dataset la_cater --dataset_version train --same_aug_pre --hm_disturb 0.0 --lost_disturb 0.0 --fp_disturb 0.0 --gpus 0,1,2,3,4,5,6,7 --batch_size 2 --load_model ../exp/tracking/ram_lacater_stage1/model_last.pth --val_intervals 2 --is_recurrent --input_len 120 --pre_thresh 0.4 --hm_weight 0.5 --num_iter 1000 --num_epochs 2 --lr_step 2 --random_walk --rw_head_depth 2 --pool_kernel 1 --sup_reg --local_rw_r 0.1 --freeze_backbone --rw_temp 0.1
# test
python test.py tracking --exp_id ram_lacater_stage2 --dataset la_cater --dataset_version test --track_thresh 0.4 --load_model ../models/lacater_120fr_20ep.pth --is_recurrent --debug 4 --input_len 70 --num_gru_layers 1 --debug 4 --random_walk --rw_head_depth 2 --pool_kernel 1 --max_age 300 --rw_score_thresh 0.005 --local_rw_r 0.100000 --new_thresh 0.5 --stream_test --sup_reg  --trainval