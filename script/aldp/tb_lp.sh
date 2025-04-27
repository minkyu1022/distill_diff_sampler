CUDA_VISIBLE_DEVICES=1 python src/train.py \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project aldp \
  --data_dir data/aldp/md_600_05 \
  --energy aldp \
  --langevin \
  --batch_size 16 \
  --clipping \
  --seed 1

