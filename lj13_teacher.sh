bash script/lj13/ours.sh 0 0 "data/lj13_5K/mala" 4000 2000 4 &
sleep 1
bash script/lj13/ours.sh 1 0 "data/lj13_10K/mala" 4000 2000 8 &
sleep 1
bash script/lj13/ours.sh 2 0 "data/lj13_20K/mala" 4000 2000 16 &
sleep 1
bash script/lj13/ours.sh 3 0 "data/lj13_40K/mala" 4000 2000 32 &
sleep 1
bash script/lj13/ours.sh 4 0 "data/lj13_80K/mala" 4000 2000 64 &
sleep 1
bash script/lj13/ours.sh 5 0 "data/lj13_160K/mala" 4000 2000 128 &
sleep 1
bash script/lj13/ours.sh 6 0 "data/lj13_320K/mala" 4000 2000 256 &
sleep 1
bash script/lj13/ours.sh 7 0 "data/lj13_640K/mala" 4000 2000 512 &
wait