

data_name=ETTm1

for seq_len in 96 192 336 720
do 
  python -u main.py \
      --conf ECL-Informer \
      --noise_rate 0.5 \
      --n_heads 8 \
      --d_model 128 \
      --d_ff 512 \
      --dropout 0.05 \
      --fc_dropout 0.05 \
      --e_layers 2 \
      --seq_len $seq_len \
      --idx -1 \
      --device "cuda:1" \
      --aligner 1 \
      --loss huber \
      --refiner 1 \
      --enhance 1 \
      --pred_len 96 \
      --train 1 \
      --data_name $data_name \
      --batch_size 64 \
      --abl 1 \
      -- RA/'ECL-Informer_woREFandALL_'$seq_len.log 
#   python main.py --conf ECL-Informer --seq_len 336 --e_layers 2 --n_heads 8 --d_model 128 --d_ff 512 --dropout 0.05 --fc_dropout 0.05 --lradj "TST" --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 0 --theta 1.5 --data_name ETTh2  --batch_size 64
  
#   python -u main.py \
#       --conf ECL-Informer \
#       --noise_rate 0.5 \
#       --n_heads 8 \
#       --d_model 128 \
#       --d_ff 512 \
#       --dropout 0.05 \
#       --fc_dropout 0.05 \
#       --e_layers 2 \
#       --seq_len 336 \
#       --idx -1 \
#       --device "cuda:1" \
#       --aligner 0 \
#       --loss huber \
#       --refiner 1 \
#       --enhance 1 \
#       --pred_len $pred_len \
#       --train 0 \
#       --data_name $data_name \
#       --batch_size 64 \
#       --abl 1 \
#       --lo ablation336/'ECL-Informer_woALI_'$pred_len.log \

#   python -u main.py \
#       --conf ECL-Informer \
#       --noise_rate 0.5 \
#       --n_heads 8 \
#       --d_model 128 \
#       --d_ff 512 \
#       --dropout 0.05 \
#       --fc_dropout 0.05 \
#       --e_layers 2 \
#       --seq_len 336 \
#       --idx -1 \
#       --device "cuda:1" \
#       --aligner 1 \
#       --loss huber \
#       --refiner 1 \
#       --enhance 1 \
#       --pred_len $pred_len \
#       --train 0 \
#       --data_name $data_name \
#       --batch_size 64 \
#       --sup_weight 0 \
#       --abl 1 \
#       --lo ablation336/'ECL-Informer_woSUP_'$pred_len.log \

#   python -u main.py \
#       --conf ECL-Informer \
#       --noise_rate 0.5 \
#       --n_heads 8 \
#       --d_model 128 \
#       --d_ff 512 \
#       --dropout 0.05 \
#       --fc_dropout 0.05 \
#       --e_layers 2 \
#       --seq_len 336 \
#       --idx -1 \
#       --device "cuda:1" \
#       --aligner 1 \
#       --loss huber \
#       --refiner 1 \
#       --enhance 1 \
#       --pred_len $pred_len \
#       --train 0 \
#       --data_name $data_name \
#       --batch_size 64 \
#       --no_tmp 1 \
#       --abl 1 \
#       --lo ablation336/'ECL-Informer_woTMP_'$pred_len.log \

  python -u main.py \
      --conf ECL-Informer \
      --noise_rate 0.5 \
      --n_heads 8 \
      --d_model 128 \
      --d_ff 512 \
      --dropout 0.05 \
      --fc_dropout 0.05 \
      --e_layers 2 \
      --seq_len $seq_len \
      --idx -1 \
      --device "cuda:1" \
      --aligner 0 \
      --loss mse \
      --refiner 0 \
      --enhance 0 \
      --pred_len 96 \
      --train 0 \
      --data_name $data_name \
      --batch_size 64 \
      --abl 1 \
      > RA/'ECL-Informer_ori_'$seq_len.log
done

  
#   python main.py --conf ECL-Informer --seq_len 336 --e_layers 2 --n_heads 8 --d_model 128 --d_ff 512 --dropout 0.05 --fc_dropout 0.05 --lradj "TST" --noise_rate 0.5 --idx -1 --device "cuda:1" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 0 --theta 1.5 --data_name ETTh2  --batch_size 64