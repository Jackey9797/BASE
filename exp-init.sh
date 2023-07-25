python main.py --conf ECL-PatchTST --noise_rate 0.01 --idx -1 --device "cuda:0" --seed 2022   --same_init --batch_size 16 > expnew/wholeECL_noise01.log
python main.py --conf ECL-PatchTST --noise_rate 0.1 --idx -1 --device "cuda:0" --seed 2022   --same_init --batch_size 16 > expnew/wholeECL_noise1.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --seed 2022   --same_init --batch_size 16 > expnew/wholeECL_noise5.log

python main.py --conf ECL-PatchTST --noise_rate 0.01 --idx -1 --device "cuda:0" --seed 2022 --data_name traffic --same_init --batch_size 16 > expnew/wholeTFF_noise01.log
python main.py --conf ECL-PatchTST --noise_rate 0.1 --idx -1 --device "cuda:0" --seed 2022  --data_name traffic --same_init --batch_size 16 > expnew/wholeTFF_noise1.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --seed 2022  --data_name traffic --same_init --batch_size 16 > expnew/wholeTFF_noise5.log

python main.py --conf ECL-PatchTST --noise_rate 0.01 --idx -1 --device "cuda:0" --seed 2022 --data_name traffic --same_init --batch_size 8 > expnew/wholeTFF8_noise01.log
python main.py --conf ECL-PatchTST --noise_rate 0.1 --idx -1 --device "cuda:0" --seed 2022  --data_name traffic --same_init --batch_size 8 > expnew/wholeTFF8_noise1.log
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --seed 2022  --data_name traffic --same_init --batch_size 8 > expnew/wholeTFF8_noise5.log

下traffic https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy
建文件夹
记得改 process = True
然后再跑