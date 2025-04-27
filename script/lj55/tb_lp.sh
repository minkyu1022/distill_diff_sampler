CUDA_VISIBLE_DEVICES=4 python src/train.py \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project lj55 \
  --data_dir data/lj55/mala \
  --energy lj55 \
  --teacher mala \
  --max_grad_norm 1.0 \
  --ld_schedule \
  --teacher_batch_size 300 \
  --langevin \
  --batch_size 2 \
  --clipping \
  --seed 1

