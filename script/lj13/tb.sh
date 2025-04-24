CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project lj13 \
  --data_dir data/lj13_300 \
  --energy lj13 \
  --both_ways \
  --clipping \
  --batch_size 2 \
  --joint_layers 1 \
  --seed 1
