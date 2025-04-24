# CUDA_VISIBLE_DEVICES=0 python src/rnd.py \
#   --date $(date +%Y-%m-%d_%H:%M:%S) \
#   --project aldp_300K_rnd \
#   --data_dir data/md_600_05 \
#   --energy aldp \
#   --both_ways \
#   --clipping \
#   --seed 0 \

# exploration_factors=(1000000000 10000000000 100000000000 1000000000000 10000000000000 100000000000000 1000000000000000)
# for i in {0..6}; do
#   CUDA_VISIBLE_DEVICES=$i python src/rnd.py \
#     --date $(date +%Y-%m-%d_%H:%M:%S) \
#     --project aldp_300K_rnd_expl \
#     --data_dir data/md_600_05 \
#     --exploration_factor ${exploration_factors[$i]} \
#     --energy aldp \
#     --both_ways \
#     --clipping \
#     --seed 0 \
#     &
#   sleep 1
# done

wait
