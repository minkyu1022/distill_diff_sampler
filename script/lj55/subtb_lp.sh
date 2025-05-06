#!/bin/bash

# 터미널에서 GPU 디바이스와 seed 값을 필수 인자로 받음
GPU_DEVICE=$1  # 첫 번째 인자
SEED=$2        # 두 번째 인자

if [ -z "$GPU_DEVICE" ] || [ -z "$SEED" ]; then
  echo "Usage: $0 <GPU_DEVICE> <SEED>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python src/train.py \
  --method subtb_lp \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project Neurips_lj55 \
  --energy lj55 \
  --mode_fwd subtb \
  --epochs 5000 \
  --langevin \
  --partial_energy \
  --conditional_flow_model \
  --batch_size 4 \
  --max_grad_norm 1.0 \
  --clipping \
  --seed $SEED \
