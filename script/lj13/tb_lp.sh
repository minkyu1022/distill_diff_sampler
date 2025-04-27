CUDA_VISIBLE_DEVICES=7 python src/train.py \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project lj13 \
  --data_dir data/lj13/mala \
  --energy lj13 \
  --langevin \
  --batch_size 16 \
  --max_grad_norm 1.0 \
  --teacher_batch_size 300 \
  --clipping \
  --seed 1

