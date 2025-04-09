SEEDS=(0 1 2 3)

for i in "${!SEEDS[@]}"; do
  CUDA_VISIBLE_DEVICES=$i python energy_sampling/train.py \
    --round 2 \
    --teacher ais \
    --t_scale 1 \
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
    --seed ${SEEDS[$i]} \
    &
done

wait