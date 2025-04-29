CUDA_VISIBLE_DEVICES=5 python src/train.py \
  --method tb_lp \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project aldp \
  --energy aldp \
  --langevin \
  --batch_size 16 \
  --clipping \
