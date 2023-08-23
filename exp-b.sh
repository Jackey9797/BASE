python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 35 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 > cuda0/seed-5.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 33 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 > cuda0/seed-3.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 31 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 > cuda0/seed-1.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 32 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 > cuda0/seed-2.log

python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 --add_FFN 1 > cuda0/a-FFN.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 --add_residual 1 > cuda0/a-res.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 --add_FFN 1 --add_residual 1 > cuda0/a-FFN-rsd.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 --add_FFN 1 --add_residual 1 --ref_dropout 0.1 > cuda0/a-FFN-rsd_dpo.log

python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 --ref_dropout 0.1 > cuda0/dpo-01.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 --ref_dropout 0.2 > cuda0/dpo-02.log

python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 --rec_all 1 > cuda0/rec_all.log

python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 --omega 2.0 > cuda0/omage-2.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 34 --same_init   --aligner 1 --loss huber --refiner 1 --enhance 1 --enhance_type 5 --add_noise 1 --mid_dim 128 --feature_jittering 1 --rec_intra_feature 0 --rec_ori 1 --jitter_sigma 2 --train 1 --test_en 0 --theta 1.1 --mask_border 1 --sup_weight 20.0 --omega 3.0 > cuda0/omage-3.log

