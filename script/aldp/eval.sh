#!/bin/bash

# 터미널에서 GPU 디바이스와 seed 값을 필수 인자로 받음
GPU_DEVICE=$1  # 첫 번째 인자
METHOD=$2
CHECKPOINT=$3
CHECKPOINT_EPOCH=$4

if [ -z "$GPU_DEVICE" ] || [ -z "$METHOD" ]; then
  echo "Usage: $0 <GPU_DEVICE> <METHOD>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python src/aldp_eval.py \
  --method $METHOD \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project Neurips_aldp_eval \
  --data_dir data/aldp_400K/md \
  --checkpoint $CHECKPOINT \
  --checkpoint_epoch 0 \
  --clipping \