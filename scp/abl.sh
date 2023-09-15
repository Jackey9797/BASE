
data_name=ETTm1

for pred_len in 96 192 336 720
do 
  python -u main.py \
      --conf ECL-PatchTST \
      --noise_rate 0.5 \
      --idx -1 \
      --device "cuda:0" \
      --aligner 1 \
      --loss huber \
      --refiner 1 \
      --enhance 1 \
      --pred_len $pred_len \
      --train 1 \
      --data_name $data_name \
      --batch_size 64 \
      --abl 1 \
      > ablation/new/'omega_ECL-PatchTST_woREFandALL_'$pred_len.log 
  
  # python -u main.py \
  #     --conf ECL-PatchTST \
  #     --noise_rate 0.5 \
  #     --idx -1 \
  #     --device "cuda:1" \
  #     --aligner 1 \
  #     --loss huber \
  #     --refiner 0 \
  #     --enhance 0 \
  #     --pred_len $pred_len \
  #     --train 1 \
  #     --data_name $data_name \
  #     --batch_size 64 \
  #     --abl 1 \
  #     > ablation/new/'ECL-PatchTST_woREF_'$pred_len.log 

#   python -u main.py \
#       --conf ECL-PatchTST \
#       --noise_rate 0.5 \
#       --idx -1 \
#       --device "cuda:0" \
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
#       --lo ablation/'ECL-PatchTST_woSUP_'$pred_len.log \

#   python -u main.py \
#       --conf ECL-PatchTST \
#       --noise_rate 0.5 \
#       --idx -1 \
#       --device "cuda:0" \
#       --aligner 1 \
#       --loss huber \
#       --refiner 1 \
#       --enhance 1 \
#       --pred_len $pred_len \
#       --train 0 \
#       --data_name $data_name \
#       --batch_size 64 \
#       --no_tmp 1 \
#       --lo ablation/'ECL-PatchTST_woTMP_'$pred_len.log \
      

#   python -u main.py \
#       --conf ECL-PatchTST \
#       --noise_rate 0.5 \
#       --idx -1 \
#       --device "cuda:0" \
#       --aligner 0 \
#       --loss mse \
#       --refiner 0 \
#       --enhance 0 \
#       --pred_len $pred_len \
#       --train 0 \
#       --data_name $data_name \
#       --batch_size 64 \
#       --abl 1 \
#       --lo ablation/'ECL-PatchTST_ori_'$pred_len.log 
done

  