CUDA_VISIBLE_DEVICES=1 python src/collect.py \
  --project lj13_mala \
  --energy lj13 \
  --teacher mala \
  --max_iter_ls 20000 \
  --burn_in 15000 \
  --ld_schedule \
  --ld_step 0.00001 \
  --seed 0
