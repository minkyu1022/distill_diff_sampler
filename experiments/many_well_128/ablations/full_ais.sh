SEEDS=(12345 23456 34567 45678)

for i in {0..3}; do
  
  echo "Running seed=${SEEDS[$i]} on GPU $i..."
  
  CUDA_VISIBLE_DEVICES=$i python energy_sampling/train.py \
    --method full_AIS \
    --round 1 \
    --teacher ais \
    --teacher_traj_len 460 \
    --t_scale 1.0 \
    --energy many_well_128 \
    --pis_architectures \
    --zero_init \
    --clipping \
    --bwd \
    --mode_bwd mle \
    --lr_policy 1e-3 \
    --hidden_dim 256 \
    --s_emb_dim 256 \
    --t_emb_dim 256 \
    --seed ${SEEDS[$i]} \
    &

done

wait

echo "All experiments completed!"