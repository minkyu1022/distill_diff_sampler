CUDA_VISIBLE_DEVICES=3 python src/train.py \
  --project lj13_300K_md \
  --energy lj13 \
  --n_steps 10000000 \
  --temperature 300 \
  --timestep 0.001 \
  --seed 0