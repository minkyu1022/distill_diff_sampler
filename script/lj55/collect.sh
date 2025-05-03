# CUDA_VISIBLE_DEVICES=0 python src/collect.py \
#   --project lj55_mala \
#   --energy lj55 \
#   --teacher mala \
#   --ld_schedule \
#   --teacher_batch_size 200 \

# CUDA_VISIBLE_DEVICES=0 python src/collect.py \
#   --project lj55_mala \
#   --energy lj55 \
#   --save_dir lj55_bad \
#   --burn_in 10000 \
#   --max_iter_ls 15000 \
#   --teacher mala \
#   --ld_schedule \
#   --teacher_batch_size 200 \

CUDA_VISIBLE_DEVICES=0 python src/collect.py \
  --project lj55_mala \
  --energy lj55 \
  --save_dir lj55_too_bad \
  --burn_in 5000 \
  --max_iter_ls 10000 \
  --teacher mala \
  --ld_schedule \
  --teacher_batch_size 200 \