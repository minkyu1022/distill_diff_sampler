#!/bin/bash

# 터미널에서 GPU 디바이스와 seed 값을 필수 인자로 받음
GPU_DEVICE=$1  # 첫 번째 인자
MAX_ITER_LS=$2 # 두 번째 인자
BURN_IN=$3
BATCH_SIZE=$4 # 네 번째 인자
SAVE_DIR=$5

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python src/collect.py \
  --project aldp_md \
  --energy aldp \
  --teacher md \
  --ld_step 5e-4 \
  --burn_in $BURN_IN \
  --max_iter_ls $MAX_ITER_LS \
  --teacher_batch_size $BATCH_SIZE \
  --save_dir $SAVE_DIR \