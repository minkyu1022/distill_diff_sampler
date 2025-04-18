#PIS+LP
CUDA_VISIBLE_DEVICES=0 python energy_sampling/train_baseline.py \
  --method PIS+LP \
  --t_scale 1.0 \
  --energy many_well_128 \
  --pis_architectures \
  --zero_init \
  --clipping \
  --mode_fwd pis \
  --lr_policy 1e-3 \
  --langevin \
  --hidden_dim 256 \
  --s_emb_dim 256 \
  --t_emb_dim 256 \
  --epochs 10000 \
  &

wait

#TB+LP
CUDA_VISIBLE_DEVICES=1 python energy_sampling/train_baseline.py \
  --method TB+LP \
  --t_scale 1.0 \
  --energy many_well_128 \
  --pis_architectures \
  --zero_init \
  --clipping \
  --mode_fwd tb \
  --lr_policy 1e-3 \
  --lr_flow 1e-1 \
  --langevin \
  --hidden_dim 256 \
  --s_emb_dim 256 \
  --t_emb_dim 256 \
  --epochs 10000 \
  &

wait

#SubTB+LP
CUDA_VISIBLE_DEVICES=2 python energy_sampling/train_baseline.py \
  --method SubTB+LP \
  --t_scale 1.0 \
  --energy many_well_128 \
  --pis_architectures \
  --zero_init \
  --clipping \
  --mode_fwd subtb \
  --lr_policy 1e-3 \
  --lr_flow 1e-2 \
  --langevin \
  --partial_energy \
  --conditional_flow_model \
  --hidden_dim 256 \
  --s_emb_dim 256 \
  --t_emb_dim 256 \
  --epochs 10000 \
  &

wait

#TB+LS+LP
CUDA_VISIBLE_DEVICES=3 python energy_sampling/train_baseline.py \
  --method TB+LS+LP \
  --t_scale 1.0 \
  --energy many_well_128 \
  --pis_architectures \
  --zero_init \
  --clipping \
  --mode_fwd tb \
  --mode_bwd tb \
  --both_ways \
  --lr_policy 1e-3 \
  --lr_back 1e-3 \
  --lr_flow 1e-1 \
  --langevin \
  --local_search \
  --ld_step 0.1 \
  --ld_schedule \
  --hidden_dim 256 \
  --s_emb_dim 256 \
  --t_emb_dim 256 \
  --epochs 10000 \
  &

wait