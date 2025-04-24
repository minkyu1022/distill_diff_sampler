CUDA_VISIBLE_DEVICES=3 python src/train.py \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project aldp_300K \
  --data_dir data/md_600_05 \
  --energy aldp \
  --bwd \
  --mode_bwd mle \
  --clipping \
  --seed 0
