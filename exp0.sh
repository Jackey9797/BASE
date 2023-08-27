# python main.py --conf ECL-PatchTST --pred_len 96 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 0 --refiner 0 --enhance 0 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 10.0 --ref_block_num 2 --rec_length_ratio 0.8 --data_name weather  > cuda3/tmp-ori.log 
# python main.py --conf ECL-PatchTST --pred_len 96 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 0 --enhance 0 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 10.0 --ref_block_num 2 --rec_length_ratio 0.8 --data_name weather  > cuda3/tmp-a.log 
# python main.py --conf ECL-PatchTST --pred_len 96 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 10.0 --ref_block_num 2 --rec_length_ratio 0.8 --data_name weather --batch_size 32 > cuda3/tmp-are.log 

# python main.py --conf ECL-PatchTST --pred_len 720 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 0 --refiner 0 --enhance 0 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 10.0 --ref_block_num 2 --rec_length_ratio 0.8 --data_name weather --batch_size 64 > cuda3/tmp720-ori.log 
# python main.py --conf ECL-PatchTST --pred_len 720 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 0 --enhance 0 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 10.0 --ref_block_num 2 --rec_length_ratio 0.8 --data_name weather --batch_size 64 > cuda3/tmp720-a.log 
# python main.py --conf ECL-PatchTST --pred_len 720 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 10.0 --ref_block_num 2 --rec_length_ratio 0.8 --data_name weather --batch_size 32 > cuda3/tmp720-are.log 


# python main.py --conf ECL-PatchTST --pred_len 336 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 0 --refiner 0 --enhance 0 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 10.0 --ref_block_num 2 --rec_length_ratio 0.8 --data_name weather --batch_size 64 > cuda3/tmp336-ori.log 
# python main.py --conf ECL-PatchTST --pred_len 336 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 0 --enhance 0 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 10.0 --ref_block_num 2 --rec_length_ratio 0.8 --data_name weather --batch_size 64 > cuda3/tmp336-a.log 
# python main.py --conf ECL-PatchTST --pred_len 336 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 10.0 --ref_block_num 2 --rec_length_ratio 0.8 --data_name weather --batch_size 64 > cuda3/tmp336-are.log 

python main.py --conf ECL-PatchTST --pred_len 48 --noise_rate 0.5 --idx -1 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.5 --mask_border 1 --sup_weight 10.0 --ref_block_num 2 --rec_length_ratio 0.8 --data_name national_illness
python main.py --conf ECL-PatchTST --pred_len 48 --noise_rate 0.5 --idx -1 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.5 --mask_border 1 python main.py --conf ECL-PatchTST --pred_len 720 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 0 --test_en 0 --theta 10 --mask_border 1 --sup_weight 10.0 --ref_block_num 2 --rec_length_ratio 0.8 --data_name weather --batch_size 32 --test_en 0 --test_model_path "/Disk/fhyega/code/BASE/exp/ECL-PatchTST2023-08-27-00:05:18.802839/0/best_model.pkl"
