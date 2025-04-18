echo "0.6-0.3 on GPU 2..."

CUDA_VISIBLE_DEVICES=2 python energy_sampling/train.py \
  --method sample_eff/current_ours \
  --round 2 \
  --teacher ais \
  --t_scale 1.0 \
  --energy many_well_128 \
  --pis_architectures \
  --zero_init \
  --clipping \
  --mode_fwd tb \
  --mode_bwd tb \
  --both_ways \
  --lr_policy 1e-3 \
  --lr_flow 1e-3 \
  --hidden_dim 256 \
  --s_emb_dim 256 \
  --t_emb_dim 256 \
  --buffer_size 6000 \
  &

