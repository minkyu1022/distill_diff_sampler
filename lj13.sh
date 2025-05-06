bash script/lj13/ours.sh 0 0 &
sleep 1
bash script/lj13/ours.sh 0 1 &
sleep 1
bash script/lj13/tb.sh 1 0 &
sleep 1
bash script/lj13/tb.sh 1 1 &
sleep 1
bash script/lj13/pis.sh 2 0 &
sleep 1
bash script/lj13/pis.sh 2 1 &
sleep 1
bash script/lj13/tb_ls.sh 3 0 &
sleep 1
bash script/lj13/tb_ls.sh 3 1 &
sleep 1
bash script/lj13/tb_expl_ls.sh 4 0 &
sleep 1
bash script/lj13/tb_expl_ls.sh 4 1 &
sleep 1
bash script/lj13/pis_lp.sh 5 0 &
sleep 1
bash script/lj13/subtb_lp.sh 5 0 &
sleep 1
bash script/lj13/tb_expl_lp.sh 6 0 &
sleep 1
bash script/lj13/tb_expl_ls_lp.sh 6 0 &
sleep 1
bash script/lj13/tb_ls_lp.sh 7 0 &
sleep 1
bash script/lj13/tb_lp.sh 7 0 &
wait
