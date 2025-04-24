CUDA_VISIBLE_DEVICES=7 python src/train.py \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project aldp_300K \
  --data_dir data/md_600_05 \
  --energy aldp \
  --both_ways \
  --clipping \
  --seed 0
