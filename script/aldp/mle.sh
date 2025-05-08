#!/bin/bash

# 터미널에서 GPU 디바이스와 seed 값을 필수 인자로 받음
GPU_DEVICE=$1  # 첫 번째 인자
SEED=$2        # 두 번째 인자
DATA_DIR=$3

if [ -z "$GPU_DEVICE" ] || [ -z "$SEED" ]; then
  echo "Usage: $0 <GPU_DEVICE> <SEED>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python src/train.py \
  --method mle \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project teacher_aldp \
  --data_dir $DATA_DIR \
  --energy aldp \
  --bwd \
  --mode_bwd mle \
  --hidden_dim 128 \
  --batch_size 8 \
  --T 200 \
  --epochs 30000 \
  --clipping \
  --seed $SEED \
