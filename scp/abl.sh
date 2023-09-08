
data_name=ETTm1


python -u main.py \
    --conf ECL-PatchTST \
    --noise_rate 0.5 \
    --idx -1 \
    --device "cuda:0" \
    --aligner 1 \
    --loss huber \
    --refiner 1 \
    --enhance 1 \
    --pred_len 96 \
    --train 1 \
    --data_name $data_name \
    --batch_size 64 \
    --abl 1 \
    > ablation/ECL-PatchTST_woREFandALL_96.log

python -u main.py \
    --conf ECL-PatchTST \
    --noise_rate 0.5 \
    --idx -1 \
    --device "cuda:0" \
    --aligner 1 \
    --loss huber \
    --refiner 1 \
    --enhance 1 \
    --pred_len 336 \
    --train 1 \
    --data_name $data_name \
    --batch_size 64 \
    --abl 1 \
    > ablation/ECL-PatchTST_woREFandALL_336.log
#   python main.py --conf ECL-Informer --seq_len 96 --e_layers 2 --n_heads 8 --d_model 128 --d_ff 512 --dropout 0.05 --fc_dropout 0.05 --lradj "TST" --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 64

python -u main.py \
    --conf ECL-PatchTST \
    --noise_rate 0.5 \
    --idx -1 \
    --device "cuda:0" \
    --aligner 0 \
    --loss huber \
    --refiner 1 \
    --enhance 1 \
    --pred_len 192 \
    --train 1 \
    --data_name $data_name \
    --batch_size 64 \
    --abl 1 \
    > ablation/ECL-PatchTST_woALI_192.log

python -u main.py \
    --conf ECL-PatchTST \
    --noise_rate 0.5 \
    --idx -1 \
    --device "cuda:0" \
    --aligner 1 \
    --loss huber \
    --refiner 1 \
    --enhance 1 \
    --pred_len 336 \
    --train 1 \
    --data_name $data_name \
    --batch_size 64 \
    --sup_weight 0 \
    --abl 1 \
    > ablation/ECL-PatchTST_woSUP_336.log



python -u main.py \
    --conf ECL-PatchTST \
    --noise_rate 0.5 \
    --idx -1 \
    --device "cuda:0" \
    --aligner 0 \
    --loss mse \
    --refiner 0 \
    --enhance 0 \
    --pred_len 720 \
    --train 1 \
    --data_name $data_name \
    --batch_size 64 \
    --abl 1 \
    > ablation/ECL-PatchTST_ori_720.log

  python -u main.py \
    --conf ECL-PatchTST \
    --noise_rate 0.5 \
    --idx -1 \
    --device "cuda:0" \
    --aligner 1 \
    --loss huber \
    --refiner 1 \
    --enhance 1 \
    --pred_len 336 \
    --train 1 \
    --data_name $data_name \
    --batch_size 64 \
    --abl 1 \
    > ablation/ECL-PatchTST_woREFandALL_336.log
#   python main.py --conf ECL-Informer --seq_len 96 --e_layers 2 --n_heads 8 --d_model 128 --d_ff 512 --dropout 0.05 --fc_dropout 0.05 --lradj "TST" --noise_rate 0.5 --idx -1 --device "cuda:0" --aligner 1 --loss huber --refiner 1 --enhance 1 --pred_len 96 --train 1 --theta 1.5 --data_name ETTh2  --batch_size 64