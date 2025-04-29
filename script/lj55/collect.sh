CUDA_VISIBLE_DEVICES=2 python src/collect.py \
  --project lj55_mala \
  --energy lj55 \
  --teacher mala \
  --ld_schedule \
  --teacher_batch_size 200 \