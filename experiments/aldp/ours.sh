SEEDS=(4 5 6 7)
# --teacher ais \
for i in {0..3}; do
  
  echo "Running seed=${SEEDS[$i]} on GPU ${SEEDS[$i]}..."
  
  CUDA_VISIBLE_DEVICES=${SEEDS[$i]} python energy_sampling/train.py \
    --round 2 \
    --project aldp_ais_md \
    --teacher ais_md \
    --t_scale 1.0 \
    --energy aldp \
    --teacher_traj_len 5000 \
    --joint_layers 5 \
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

echo "All experiments completed!"