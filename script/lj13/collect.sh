# CUDA_VISIBLE_DEVICES=0 python src/collect.py \
#   --project lj13_mala \
#   --energy lj13 \
#   --teacher mala \
#   --burn_in 4000 \
#   --max_iter_ls 6000 \
#   --teacher_batch_size 500 \
#   --ld_schedule \

# CUDA_VISIBLE_DEVICES=0 python src/collect.py \
#   --project lj13_mala \
#   --energy lj13 \
#   --save_dir lj13_bad \
#   --teacher mala \
#   --burn_in 2000 \
#   --max_iter_ls 4000 \
#   --teacher_batch_size 500 \
#   --ld_schedule \

CUDA_VISIBLE_DEVICES=0 python src/collect.py \
  --project lj13_mala \
  --energy lj13 \
  --save_dir lj13_too_bad \
  --teacher mala \
  --burn_in 1000 \
  --max_iter_ls 3000 \
  --teacher_batch_size 500 \
  --ld_schedule \