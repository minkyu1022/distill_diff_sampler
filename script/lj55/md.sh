CUDA_VISIBLE_DEVICES=0 python src/train.py \
  --project lj55_300K_md \
  --energy lj55 \
  --n_steps 10000000 \
  --temperature 300 \
  --timestep 0.001 \
  --seed 0