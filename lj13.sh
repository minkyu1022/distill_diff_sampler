bash script/lj13/ours.sh 0 0 &
sleep 1
bash script/lj13/ours.sh 1 1 &
sleep 1
bash script/lj13/pis_lp.sh 2 0 &
sleep 1
bash script/lj13/pis_lp.sh 3 1 &
sleep 1
bash script/lj13/subtb_lp.sh 4 0 &
sleep 1
bash script/lj13/subtb_lp.sh 5 1 &
sleep 1
bash script/lj13/tb_ls.sh 6 0 &
sleep 1
bash script/lj13/tb_ls.sh 7 1 &
sleep 1
wait







# bash script/lj13/tb_expl_ls_lp.sh &
# sleep 1
# bash script/lj13/tb_expl_ls.sh &
# sleep 1
# bash script/lj13/tb_lp.sh &
# sleep 1
# bash script/lj13/tb.sh &