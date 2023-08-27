# python main.py --conf ECL-PatchTST --pred_len 96 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --aligner 0 --refiner 0 --enhance 0 --test_en 0 --thet --pred_len 720  --theta 1.5 --data_name weather  > cuda3/tmp-ori.log 
# python main.py --conf ECL-PatchTST --pred_len 96 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --aligner 1 --loss huber --refiner 0 --enhance 0 --test_en 0 --thet --pred_len 720  --theta 1.5 --data_name weather  > cuda3/tmp-a.log 
# python main.py --conf ECL-PatchTST --pred_len 96 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --test_en 0 --thet --pred_len 720  --theta 1.5 --data_name weather --batch_size 32 > cuda3/tmp-are.log 

# python main.py --conf ECL-PatchTST --pred_len 336 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --aligner 0 --refiner 0 --enhance 0 --test_en 0 --thet --pred_len 720  --theta 1.5 --data_name weather --batch_size 64 > cuda3/tmp336-ori.log 
# python main.py --conf ECL-PatchTST --pred_len 336 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --aligner 1 --loss huber --refiner 0 --enhance 0 --test_en 0 --thet --pred_len 720  --theta 1.5 --data_name weather --batch_size 64 > cuda3/tmp336-a.log 
# python main.py --conf ECL-PatchTST --pred_len 336 --noise_rate 0.5 --idx -1 --data_name ETTh2 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --test_en 0 --thet --pred_len 720  --theta 1.5 --data_name weather --batch_size 64 > cuda3/tmp336-are.log 

# python main.py --conf ECL-PatchTST --pred_len 48 --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 720  --theta 1.5 --data_name national_illness
# python main.py --conf ECL-PatchTST --pred_len 48 --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 720 --train 1 --theta 1.5 --data_name weather --batch_size 64 --test_en 0
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 720 --train 1 --theta 1.5 --data_name ETTh2 --batch_size 128 --test_en 0
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 720 --train 1 --theta 1.5 --data_name electricity --batch_size 32 --test_en 0
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 720 --train 1 --theta 1.5 --data_name national_illness --batch_size 128 --test_en 0
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 720 --train 1 --theta 1.5 --data_name exchange_rate --batch_size 128 --test_en 0
python main.py --conf ECL-PatchTST --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 720 --train 1 --theta 1.5 --data_name traffic --batch_size 128 --test_en 0
