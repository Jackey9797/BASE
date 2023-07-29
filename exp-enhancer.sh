python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 22033 --same_init  --refiner 1 --enhance 1 --enhance_type 1 > exp_r_new/ETT_203_r-re_en_t1.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 32033 --same_init  --refiner 1 --enhance 1 --enhance_type 1 > exp_r_new/ETT_303_r-re_en_t1.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 42033 --same_init  --refiner 1 --enhance 1 --enhance_type 1 > exp_r_new/ETT_403_r-re_en_t1.log

python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 22033 --same_init  --refiner 1 --enhance 1 --enhance_type 2 > exp_r_new/ETT_203_r-re_en_t2.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 32033 --same_init  --refiner 1 --enhance 1 --enhance_type 2 > exp_r_new/ETT_303_r-re_en_t2.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 42033 --same_init  --refiner 1 --enhance 1 --enhance_type 2 > exp_r_new/ETT_403_r-re_en_t2.log

python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 22033 --same_init  --refiner 1 --enhance 1 --enhance_type 3 > exp_r_new/ETT_203_r-re_en_t3.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 32033 --same_init  --refiner 1 --enhance 1 --enhance_type 3 > exp_r_new/ETT_303_r-re_en_t3.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 42033 --same_init  --refiner 1 --enhance 1 --enhance_type 3 > exp_r_new/ETT_403_r-re_en_t3.log



python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 12033 --same_init  --refiner 1 --enhance 1 > exp_r_new/ETT_103_r-re_en.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 22033 --same_init  --refiner 1 --enhance 1 > exp_r_new/ETT_203_r-re_en.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 32033 --same_init  --refiner 1 --enhance 1 > exp_r_new/ETT_303_r-re_en.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 42033 --same_init  --refiner 1 --enhance 1 > exp_r_new/ETT_403_r-re_en.log

# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 2033 --same_init  --refiner 1 --enhance 1 --enhance_type 1 > exp_r_new/ETT_33_r-re_en_.log

