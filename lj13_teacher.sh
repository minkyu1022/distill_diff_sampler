bash script/lj13/ours.sh 0 0 "data/lj13_4K/mala" 4000 2000 4 &
sleep 1
bash script/lj13/ours.sh 1 0 "data/lj13_8K/mala" 4000 2000 8 &
sleep 1
bash script/lj13/ours.sh 2 0 "data/lj13_16K/mala" 4000 2000 16 &
sleep 1
bash script/lj13/ours.sh 3 0 "data/lj13_32K/mala" 4000 2000 32 &
sleep 1
bash script/lj13/ours.sh 4 0 "data/lj13_64K/mala" 4000 2000 64 &
sleep 1
bash script/lj13/ours.sh 5 0 "data/lj13_128K/mala" 4000 2000 128 &
sleep 1
bash script/lj13/ours.sh 6 0 "data/lj13_256K/mala" 4000 2000 256 &
sleep 1
bash script/lj13/ours.sh 7 0 "data/lj13_512K/mala" 4000 2000 512 &
wait