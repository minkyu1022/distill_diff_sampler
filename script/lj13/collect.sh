#!/bin/bash

# 터미널에서 GPU 디바이스와 seed 값을 필수 인자로 받음
GPU_DEVICE=$1  # 첫 번째 인자
MAX_ITER_LS=$2 # 두 번째 인자
BURN_IN=$3 # 세 번째 인자
BATCH_SIZE=$4 # 네 번째 인자
SAVE_DIR=$5

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python src/collect.py \
  --project lj13_mala \
  --energy lj13 \
  --teacher mala \
  --max_iter_ls $MAX_ITER_LS \
  --burn_in $BURN_IN \
  --teacher_batch_size $BATCH_SIZE \
  --save_dir $SAVE_DIR \
  --ld_schedule \

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

# CUDA_VISIBLE_DEVICES=0 python src/collect.py \
#   --project lj13_mala \
#   --energy lj13 \
#   --save_dir lj13_too_bad \
#   --teacher mala \
#   --burn_in 1000 \
#   --max_iter_ls 3000 \
#   --teacher_batch_size 500 \
#   --ld_schedule \