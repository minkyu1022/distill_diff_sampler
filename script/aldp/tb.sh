CUDA_VISIBLE_DEVICES=7 python src/train.py \
  --method tb \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project aldp \
  --energy aldp \
  --epochs 30000 \
  --clipping \
