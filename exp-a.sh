# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --seed 2021 --data_name ETTh2 > expnew/1_0_noise5.log
# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --seed 2023 --data_name ETTh2  > expnew/3_0_noise5.log

python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --seed 2021 --data_name ETTh2 --aligner 1 --same_init > expnew/1_4_si_noise5.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --seed 2022 --data_name ETTh2 --same_init > expnew/2_0_si_noise5.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --seed 2022 --data_name ETTh2 --aligner 1  > expnew/2_4_noise5.log