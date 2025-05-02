#!/bin/bash

# 터미널에서 GPU 디바이스와 seed 값을 필수 인자로 받음
GPU_DEVICE=$1  # 첫 번째 인자
SEED=$2        # 두 번째 인자
RND_WEIGHT=$3  # 세 번째 인자 (선택적, 기본값은 1000000)
LR_POLICY=$4  # 네 번째 인자 (선택적, 기본값은 5e-4)
# LR_FLOW=$5  # 다섯 번째 인자 (선택적, 기본값은 1e-3)

if [ -z "$GPU_DEVICE" ] || [ -z "$SEED" ]; then
  echo "Usage: $0 <GPU_DEVICE> <SEED>"
  exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python src/train.py \
  --method ours \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project real_final_lj13 \
  --data_dir data/lj13/mala \
  --energy lj13 \
  --teacher mala \
  --epochs 5000 10000 \
  --max_grad_norm 1.0 \
  --burn_in 4000 \
  --max_iter_ls 6000 \
  --teacher_batch_size 250 \
  --rnd_weight ${RND_WEIGHT:-1000000} \
  --lr_policy ${LR_POLICY:-5e-4} \
  --lr_flow ${LR_POLICY:-1e-3} \
  --gfn_clip ${RND_WEIGHT:-1000000} \
  --ld_schedule \
  --both_ways \
  --clipping \
  --seed $SEED \
