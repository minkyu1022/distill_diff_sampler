bash script/lj55/ours.sh 0 0 &
sleep 1
bash script/lj55/ours.sh 0 1 &
sleep 1
bash script/lj55/tb.sh 1 0 &
sleep 1
bash script/lj55/tb.sh 1 1 &
sleep 1
bash script/lj55/pis.sh 2 0 &
sleep 1
bash script/lj55/pis.sh 2 1 &
sleep 1
bash script/lj55/tb_ls.sh 3 0 &
sleep 1
bash script/lj55/tb_ls.sh 3 1 &
sleep 1
bash script/lj55/tb_expl_ls.sh 4 0 &
sleep 1
bash script/lj55/tb_expl_ls.sh 4 1 &
wait
