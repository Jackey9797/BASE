# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 0 --refiner 0 --enhance 0 --pred_len 96 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128
# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 0 --refiner 0 --enhance 0 --pred_len 192 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128
# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 0 --refiner 0 --enhance 0 --pred_len 336 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128
# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 0 --refiner 0 --enhance 0 --pred_len 720 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128

python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss mse --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 0 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 0 --enhance 0 --pred_len 96 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 
# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 192 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss mse --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 0 --loss huber --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 0 --enhance 0 --pred_len 336 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 
# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 720 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 


python main.py --conf ECL-PatchTST --noise_rate 0.8 --idx -1 --device "cuda:0" --aligner 1 --loss mse --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name weather --batch_size 32 --early_break 15
python main.py --conf ECL-PatchTST --noise_rate 0.8 --idx -1 --device "cuda:0" --aligner 0 --loss huber --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name weather --batch_size 32 --early_break 15
python main.py --conf ECL-PatchTST --noise_rate 0.8 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 0 --enhance 0 --pred_len 336 --train 1 --theta 1.5 --data_name weather --batch_size 32 --early_break 15

python main.py --conf ECL-PatchTST --noise_rate 0.8 --idx -1 --device "cuda:1" --aligner 1 --loss mse --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name weather --batch_size 32
python main.py --conf ECL-PatchTST --noise_rate 0.8 --idx -1 --device "cuda:1" --aligner 0 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name weather --batch_size 32
python main.py --conf ECL-PatchTST --noise_rate 0.8 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 0 --enhance 0 --pred_len 96 --train 1 --theta 1.5 --data_name weather --batch_size 32

python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss mse --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 128 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 0 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 128 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 0 --enhance 0 --pred_len 96 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 128 

python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss mse --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 128 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 0 --loss huber --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 128 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 0 --enhance 0 --pred_len 336 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 128 