# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 2033 --same_init   > exp_a/ETT_t33_always_a_si_A10_b.log
# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 2033 --same_init  --refiner 1 > exp_r/ETT_33_r-AE.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 2033 --same_init  --refiner 1 > exp_r/ETT_33_r-re_dp.log
