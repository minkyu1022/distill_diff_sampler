CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project lj55 \
  --data_dir data/lj55_300 \
  --energy lj55 \
  --both_ways \
  --clipping \
  --batch_size 2 \
  --joint_layers 1 \
  --seed 0
