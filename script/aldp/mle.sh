CUDA_VISIBLE_DEVICES=2 python src/train.py \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project aldp \
  --data_dir data/aldp/md_600_05 \
  --energy aldp \
  --bwd \
  --mode_bwd mle \
  --clipping \
  --seed 1
