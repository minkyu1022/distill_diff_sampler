CUDA_VISIBLE_DEVICES=1 python src/train.py \
  --method pis_lp \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project aldp \
  --mode_fwd pis \
  --energy aldp \
  --langevin \
  --batch_size 16 \
  --clipping \
