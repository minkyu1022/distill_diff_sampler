CUDA_VISIBLE_DEVICES=7 python src/train.py \
  --method tb \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project final_lj13 \
  --energy lj13 \
  --epochs 20000 \
  --max_grad_norm 1.0 \
  --clipping \
