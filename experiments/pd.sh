# Initial model pre-trained on NuScenes3D: https://drive.google.com/open?id=1ZSG9swryMEfBJ104WH8CP7kcypCobFlU
# Resulting model trained on PD: 

cd src
# train
python main.py tracking --exp_id ram_pd --occlusion_thresh 0.15 --visibility_thresh 0.05 --dataset pd_tracking --dataset_version val --same_aug_pre --hm_disturb 0.0 --lost_disturb 0.0 --fp_disturb 0.0 --gpus 0,1,2,3,4,5,6,7 --batch_size 2 --load_model ../models/nuScenes_3Ddetection_e140.pth --val_intervals 2 --is_recurrent --input_len 16 --pre_thresh 0.4 --hm_weight 0.5 --num_epochs 28 --lr_step 7 --num_iter 5000 --save_point 7,14,21 --random_walk --rw_head_depth 2 --pool_kernel 3 --sup_centeroverlap --sup_reg --rw_temp 0.1
# test
python test.py tracking --exp_id ram_pd --dataset pd_tracking --dataset_version val --track_thresh 0.4 --resume --is_recurrent --input_len 16 --debug 4 --random_walk --rw_head_depth 2 --pool_kernel 3 --max_age 16 --local_rw_r 0.2 --stream_test --new_thresh 0.5 --sup_reg
cd ..