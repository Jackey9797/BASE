
python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss mse --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --refiner_residual 1
python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 0 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --refiner_residual 1
python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --refiner_residual 1
# python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 192 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --refiner_residual 1
python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss mse --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --refiner_residual 1
python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 0 --loss huber --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --refiner_residual 1
python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTm1  --batch_size 128 --abl 1 --refiner_residual 1


# python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss mse --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 128 --abl 1 --refiner_residual 1
# python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 0 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 128 --abl 1 --refiner_residual 1
# python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 128 --abl 1 --refiner_residual 1

# python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss mse --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 128 --abl 1 --refiner_residual 1
# python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 0 --loss huber --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 128 --abl 1 --refiner_residual 1
# python mainv2.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 128 --abl 1--refiner_residual 1 