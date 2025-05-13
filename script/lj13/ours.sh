#!/bin/bash

# 터미널에서 GPU 디바이스와 seed 값을 필수 인자로 받음
GPU_DEVICE=$1  # 첫 번째 인자
SEED=$2        # 두 번째 인자

if [ -z "$GPU_DEVICE" ] || [ -z "$SEED" ]; then
  echo "Usage: $0 <GPU_DEVICE> <SEED>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python src/train.py \
  --method ours \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project Neurips_lj13 \
  --data_dir data/lj13_16K/mala \
  --energy lj13 \
  --teacher mala \
  --time_scheduler random \
  --epochs 5000 15000 \
  --max_grad_norm 1.0 \
  --burn_in 2000 \
  --max_iter_ls 4000 \
  --teacher_batch_size 16 \
  --rnd_weight 10 \
  --ld_schedule \
  --both_ways \
  --clipping \
  --seed $SEED \
