# 定义要循环的 model 和 data_name
#!/bin/bash


models=("ECL-Informer" "ECL-FEDformer" "ECL-Autoformer")
data_names=("ETTh2" "weather" "exchange_rate")

for model in "${models[@]}"
do
  for data_name in "${data_names[@]}"
  do
    for pred_len in 96 192 336 720
    do 
      python -u main.py \
          --conf "$model" \
          --noise_rate 0.5 \
          --n_heads 8 \
          --d_model 128 \
          --d_ff 512 \
          --dropout 0.05 \
          --fc_dropout 0.05 \
          --e_layers 2 \
          --seq_len 96 \
          --idx -1 \
          --device "cuda:0" \
          --aligner 1 \
          --loss huber \
          --refiner 1 \
          --enhance 1 \
          --pred_len "$pred_len" \
          --train 1 \
          --data_name "$data_name" \
          --batch_size 64 \
          --mainrs 1 \
          > mainresult/"$model"'_OURS_'"$pred_len"'_'"$data_name"'.log'

      python -u main.py \
          --conf "$model" \
          --noise_rate 0.5 \
          --n_heads 8 \
          --d_model 128 \
          --d_ff 512 \
          --dropout 0.05 \
          --fc_dropout 0.05 \
          --e_layers 2 \
          --seq_len 96 \
          --idx -1 \
          --device "cuda:0" \
          --aligner 0 \
          --loss mse \
          --refiner 0 \
          --enhance 0 \
          --pred_len "$pred_len" \
          --train 1 \
          --data_name "$data_name" \
          --batch_size 64 \
          --mainrs 1 \
          > mainresult/"$model"'_ori_'"$pred_len"'_'"$data_name"'.log'
    done
  done
done

data_names=("electricity" "traffic")

for model in "${models[@]}"
do
  for data_name in "${data_names[@]}"
  do
    for pred_len in 96 192 336 720
    do 
      python -u main.py \
          --conf "$model" \
          --noise_rate 0.5 \
          --n_heads 8 \
          --d_model 512 \
          --d_ff 2048 \
          --dropout 0.05 \
          --fc_dropout 0.05 \
          --e_layers 2 \
          --seq_len 96 \
          --idx -1 \
          --device "cuda:0" \
          --aligner 1 \
          --loss huber \
          --refiner 1 \
          --enhance 1 \
          --pred_len "$pred_len" \
          --train 1 \
          --data_name "$data_name" \
          --batch_size 16 \
          --mainrs 1 \
          > mainresult/"$model"'_OURS_'"$pred_len"'_'"$data_name"'.log'

      python -u main.py \
          --conf "$model" \
          --noise_rate 0.5 \
          --n_heads 8 \
          --d_model 512 \
          --d_ff 2048 \
          --dropout 0.05 \
          --fc_dropout 0.05 \
          --e_layers 2 \
          --seq_len 96 \
          --idx -1 \
          --device "cuda:0" \
          --aligner 0 \
          --loss mse \
          --refiner 0 \
          --enhance 0 \
          --pred_len "$pred_len" \
          --train 1 \
          --data_name "$data_name" \
          --batch_size 16 \
          --mainrs 1 \
          > mainresult/"$model"'_ori_'"$pred_len"'_'"$data_name"'.log'
    done
  done
done