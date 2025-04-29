lr_policys=(0.0005 0.0001 0.00005 0.00001)
for i in {0..3}
do
  lr_policy=${lr_policys[$i]}
  CUDA_VISIBLE_DEVICES=$i python src/train.py \
    --date $(date +%Y-%m-%d_%H:%M:%S) \
    --project aldp_lr \
    --data_dir data/aldp/md \
    --energy aldp \
    --both_ways \
    --clipping \
    --seed 0 \
    --lr_policy $lr_policy &
  sleep 1
done
wait