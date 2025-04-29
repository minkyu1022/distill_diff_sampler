CUDA_VISIBLE_DEVICES=7 python src/train.py \
  --method tb \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project lj55 \
  --energy lj55 \
  --epochs 30000 \
  --max_grad_norm 1.0 \
  --batch_size 4 \
  --clipping \