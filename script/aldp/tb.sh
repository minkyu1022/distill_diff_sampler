CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project aldp \
  --data_dir data/aldp/md_600_05 \
  --energy aldp \
  --both_ways \
  --clipping \
  --seed 1
