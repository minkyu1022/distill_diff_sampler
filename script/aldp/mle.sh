CUDA_VISIBLE_DEVICES=7 python src/train.py \
  --method mle \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project aldp \
  --data_dir data/aldp/md \
  --energy aldp \
  --bwd \
  --mode_bwd mle \
  --clipping \
