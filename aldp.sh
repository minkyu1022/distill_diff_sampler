bash script/aldp/ours.sh 0 0 &
sleep 1
bash script/aldp/mle.sh 1 0 &
sleep 1
bash script/aldp/ours.sh 2 1 &
sleep 1
bash script/aldp/mle.sh 3 1 &
sleep 1
bash script/aldp/ours.sh 4 2 &
sleep 1
bash script/aldp/mle.sh 5 2 &
sleep 1
bash script/aldp/ours.sh 6 3 &
sleep 1
bash script/aldp/mle.sh 7 3 &

wait