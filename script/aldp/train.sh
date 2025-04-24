# exploration_factors=(100 1000 10000 100000 1000000 10000000 100000000)
# for i in {1..7}; do
#   CUDA_VISIBLE_DEVICES=$i python src/train.py \
#     --date $(date +%Y-%m-%d_%H:%M:%S) \
#     --project 250421_aldp_md_300 \
#     --data_dir data/md_300_05 \
#     --energy aldp_mlff \
#     --both_ways \
#     --exploration_factor ${exploration_factors[$((i-1))]} \
#     --clipping \
#     --seed 0 \
#     &
#   sleep 1
# done

# wait

# --langevin \
for i in {0..0}; do
  CUDA_VISIBLE_DEVICES=$i python src/train.py \
    --date $(date +%Y-%m-%d_%H:%M:%S) \
    --project 250421_aldp_md_300 \
    --data_dir data/md_600_05 \
    --energy aldp \
    --both_ways \
    --n_steps 100 \
    --batch_size 2 \
    --epochs 2 \
    --clipping \
    --seed 0 \
    &
  sleep 1
done

wait