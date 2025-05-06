bash script/lj13/ours.sh 0 0 "data/lj13_5K/mala" 12000 2000 1 &
sleep 1
bash script/lj13/ours.sh 1 0 "data/lj13_10K/mala" 12000 2000 2 &
sleep 1
bash script/lj13/ours.sh 2 0 "data/lj13_20K/mala" 12000 2000 4 &
sleep 1
bash script/lj13/ours.sh 3 0 "data/lj13_40K/mala" 12000 2000 8 &
sleep 1
bash script/lj13/ours.sh 4 0 "data/lj13_80K/mala" 12000 2000 16 &
sleep 1
bash script/lj13/ours.sh 5 0 "data/lj13_160K/mala" 12000 2000 32 &
sleep 1
bash script/lj13/ours.sh 6 0 "data/lj13_320K/mala" 12000 2000 64 &
sleep 1
bash script/lj13/ours.sh 7 0 "data/lj13_640K/mala" 12000 2000 128 &
wait