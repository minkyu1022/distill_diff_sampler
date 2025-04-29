CUDA_VISIBLE_DEVICES=1 python src/train.py \
  --method pis_lp \
  --date $(date +%Y-%m-%d_%H:%M:%S) \
  --project lj55 \
  --energy lj55 \
  --mode_fwd pis \
  --langevin \
  --batch_size 2 \
  --max_grad_norm 1.0 \
  --clipping \