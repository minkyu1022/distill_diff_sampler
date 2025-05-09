#!/bin/bash

# 터미널에서 GPU 디바이스와 seed 값을 필수 인자로 받음
GPU_DEVICE=$1  # 첫 번째 인자
SEED=$2        # 두 번째 인자
LR=$3

if [ -z "$GPU_DEVICE" ] || [ -z "$SEED" ]; then
  echo "Usage: $0 <GPU_DEVICE> <SEED>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python src/train.py \
  --method ours \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project Neurips_aldp_rnd \
  --data_dir data/aldp_400K/md \
  --energy aldp \
  --scheduler_type random \
  --epochs 10000 30000 \
  --mle_epoch 5000 \
  --rnd_weight 10000 \
  --burn_in 10000 \
  --lr_policy $LR \
  --lr_flow $LR \
  --max_iter_ls 110000 \
  --teacher_batch_size 4 \
  --both_ways \
  --clipping \
  --seed $SEED \