python main.py --conf ECL-PatchTST --noise_rate 0.01 --idx 89 --device "cuda:1" --seed 2021 > expnew1/_1_0_noise01.log
python main.py --conf ECL-PatchTST --noise_rate 0.05 --idx 89 --device "cuda:1" --seed 2021 > expnew1/_1_0_noise05.log
python main.py --conf ECL-PatchTST --noise_rate 0.1 --idx 89 --device "cuda:1" --seed 2021 > expnew1/_1_0_noise1.log
python main.py --conf ECL-PatchTST --noise_rate 0.2 --idx 89 --device "cuda:1" --seed 2021 > expnew1/_1_0_noise2.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx 89 --device "cuda:1" --seed 2021 > expnew1/_1_0_noise5.log


python main.py --conf ECL-PatchTST --noise_rate 0.01 --idx 213 --device "cuda:1" --seed 2021 > expnew1/_1_0_noise01_213.log
python main.py --conf ECL-PatchTST --noise_rate 0.1 --idx 213 --device "cuda:1" --seed 2021 > expnew1/_1_0_noise1_213.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx 213 --device "cuda:1" --seed 2021 > expnew1/_1_0_noise5_213.log


python main.py --conf ECL-PatchTST --noise_rate 0.01 --idx 213 --device "cuda:1" --seed 2022 > expnew1/_2_0_noise01_213.log
python main.py --conf ECL-PatchTST --noise_rate 0.1 --idx 213 --device "cuda:1" --seed 2022 > expnew1/_2_0_noise1_213.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx 213 --device "cuda:1" --seed 2022 > expnew1/_2_0_noise5_213.log
