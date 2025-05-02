#!/bin/bash

# 터미널에서 GPU 디바이스와 seed 값을 필수 인자로 받음
GPU_DEVICE=$1  # 첫 번째 인자
SEED=$2        # 두 번째 인자
RND_WEIGHT=$3  # 세 번째 인자 (선택적, 기본값은 1e9)

if [ -z "$GPU_DEVICE" ] || [ -z "$SEED" ]; then
  echo "Usage: $0 <GPU_DEVICE> <SEED>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python src/train.py \
  --method ours \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project aldp \
  --data_dir data/aldp/md \
  --rnd_weight ${RND_WEIGHT:-1000000000} \
  --energy aldp \
  --epochs 20000 60000 \
  --both_ways \
  --clipping \
  --seed $SEED