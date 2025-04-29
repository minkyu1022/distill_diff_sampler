CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project aldp \
  --data_dir data/aldp/md \
  --energy aldp \
  --both_ways \
  --epochs 10000 \
  --clipping \
