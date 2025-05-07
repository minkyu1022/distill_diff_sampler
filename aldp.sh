bash script/aldp/ours.sh 0 0 &
sleep 1
bash script/aldp/ours.sh 1 1 &
sleep 1
bash script/aldp/ours.sh 2 2 &
sleep 1
bash script/aldp/ours.sh 3 3 &
sleep 1
bash script/aldp/mle.sh 4 0 &
sleep 1
bash script/aldp/mle.sh 5 1 &
sleep 1
bash script/aldp/mle.sh 6 2 &
sleep 1
bash script/aldp/mle.sh 7 3 &
wait