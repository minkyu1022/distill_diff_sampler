SEEDS=(12345 23456 34567 45678)

for i in {0..3}; do
  
  echo "Running seed=${SEEDS[$i]} on GPU $i..."
  
  CUDA_VISIBLE_DEVICES=$i python energy_sampling/train.py \
    --method TB \
    --t_scale 1.0 \
    --energy many_well_64 \
    --pis_architectures \
    --zero_init \
    --clipping \
    --mode_fwd tb \
    --lr_policy 1e-3 \
    --lr_flow 1e-1 \
    --hidden_dim 256 \
    --s_emb_dim 256 \
    --t_emb_dim 256 \
    --seed ${SEEDS[$i]} \
    &

done

wait

echo "All experiments completed!"