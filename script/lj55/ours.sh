#!/bin/bash

# 터미널에서 GPU 디바이스와 seed 값을 필수 인자로 받음
GPU_DEVICE=$1  # 첫 번째 인자
SEED=$2        # 두 번째 인자
RND_WEIGHT=$3  # 세 번째 인자 (선택적, 기본값은 1000000)

if [ -z "$GPU_DEVICE" ] || [ -z "$SEED" ]; then
  echo "Usage: $0 <GPU_DEVICE> <SEED>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python src/train.py \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project real_final_lj55 \
  --data_dir data/lj55/mala \
  --energy lj55 \
  --teacher mala \
  --rnd_weight ${RND_WEIGHT:-1000000} \
  --max_grad_norm 1.0 \
  --ld_schedule \
  --burn_in 10000 \
  --max_iter_ls 15000 \
  --teacher_batch_size 100 \
  --batch_size 4 \
  --epochs 10000 30000 \
  --both_ways \
  --clipping \
  --seed $SEED \