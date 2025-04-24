CUDA_VISIBLE_DEVICES=2 python src/train.py \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project aldp_300K \
  --energy aldp \
  --langevin \
  --clipping \
  --seed 0
