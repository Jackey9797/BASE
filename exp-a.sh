python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 33 --same_init   --aligner 1 --loss huber 
# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --seed 33 --same_init   --aligner 1 --loss huber  --always_align 0
python main.py --conf ECL-PatchTST --pred_len 336 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:1" --seed 35 --same_init   --aligner 1 --loss huber
