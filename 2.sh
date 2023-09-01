python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --ref_dropout 0.2
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --ref_dropout 0.2

python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --ref_block_num 1
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --ref_block_num 1

python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --add_FFN 1
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --add_FFN 1

