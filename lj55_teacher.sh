bash script/lj55/ours.sh 0 0 "data/lj55_5K/mala" 20000 10000 1 &
sleep 1
bash script/lj55/ours.sh 1 0 "data/lj55_10K/mala" 20000 10000 2 &
sleep 1
bash script/lj55/ours.sh 2 0 "data/lj55_20K/mala" 20000 10000 4 &
sleep 1
bash script/lj55/ours.sh 3 0 "data/lj55_40K/mala" 20000 10000 8 &
sleep 1
bash script/lj55/ours.sh 4 0 "data/lj55_80K/mala" 20000 10000 16 &
sleep 1
bash script/lj55/ours.sh 5 0 "data/lj55_160K/mala" 20000 10000 32 &
sleep 1
bash script/lj55/ours.sh 6 0 "data/lj55_320K/mala" 20000 10000 64 &
sleep 1
bash script/lj55/ours.sh 7 0 "data/lj55_640K/mala" 20000 10000 128 &
wait