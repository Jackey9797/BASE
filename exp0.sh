# python main.py --conf ECL-PatchTST --noise_rate 0.1 --idx 89 --device "cuda:0" --seed 2021 >expnew/o1.log
# python main.py --conf ECL-PatchTST --noise_rate 0.1 --idx 89 --device "cuda:0" --seed 2022 >expnew/o2.log
python main.py --conf ECL-PatchTST --noise_rate 0.1 --idx 89 --device "cuda:0" --seed 2023 >expnew/o3.log

python main.py --conf ECL-PatchTST --noise_rate 0.1 --idx 89 --device "cuda:0" --seed 2021 --refiner 1 >expnew/o_r1.log
python main.py --conf ECL-PatchTST --noise_rate 0.1 --idx 89 --device "cuda:0" --seed 2022 --refiner 1 >expnew/o_r2.log
python main.py --conf ECL-PatchTST --noise_rate 0.1 --idx 89 --device "cuda:0" --seed 2023 --refiner 1 >expnew/o_r3.log

