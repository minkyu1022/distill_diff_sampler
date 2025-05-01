bash script/lj55/ours.sh 0 0 &
sleep 1
bash script/lj55/ours.sh 1 1 &
sleep 1
bash script/lj55/pis_lp.sh 2 0 &
sleep 1
bash script/lj55/pis_lp.sh 3 1 &
sleep 1
bash script/lj55/subtb_lp.sh 4 0 &
sleep 1
bash script/lj55/subtb_lp.sh 5 1 &
sleep 1
bash script/lj55/tb_ls.sh 6 0 &
sleep 1
bash script/lj55/tb_ls.sh 7 1 &
sleep 1
wait