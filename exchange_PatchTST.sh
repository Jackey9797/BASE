python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name exchange_rate  --batch_size 64 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 192 --train 1 --theta 1.5 --data_name exchange_rate  --batch_size 64 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 336 --train 1 --theta 1.5 --data_name exchange_rate  --batch_size 64 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 720 --train 1 --theta 1.5 --data_name exchange_rate  --batch_size 64 

# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 0 --refiner 0 --enhance 0 --pred_len 96 --train 1 --theta 1.5 --data_name exchange_rate  --batch_size 64 
# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 0 --refiner 0 --enhance 0 --pred_len 192 --train 1 --theta 1.5 --data_name exchange_rate  --batch_size 64 
# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 0 --refiner 0 --enhance 0 --pred_len 336 --train 1 --theta 1.5 --data_name exchange_rate  --batch_size 64 
# python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 0 --refiner 0 --enhance 0 --pred_len 720 --train 1 --theta 1.5 --data_name exchange_rate  --batch_size 64 

