bash script/aldp/ours.sh 0 0 &
sleep 1
bash script/aldp/ours.sh 1 1 &
sleep 1
bash script/aldp/mle.sh 2 0 &
sleep 1
bash script/aldp/mle.sh 3 1 &
sleep 1
bash script/aldp/pis.sh 4 0 &
sleep 1
bash script/aldp/pis.sh 5 1 &
sleep 1
bash script/aldp/tb_expl_ls.sh 6 0 &
sleep 1
bash script/aldp/tb_expl_ls.sh 7 1 &
wait