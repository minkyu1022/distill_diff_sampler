bash script/lj55/collect.sh 0 10000 5000 2 lj55_5K      & # 20K
bash script/lj55/collect.sh 1 10000 5000 4 lj55_10K     & # 40K
bash script/lj55/collect.sh 2 10000 5000 8 lj55_20K     & # 80K
bash script/lj55/collect.sh 3 10000 5000 16 lj55_40K    & # 160K
bash script/lj55/collect.sh 4 10000 5000 32 lj55_80K    & # 320K
bash script/lj55/collect.sh 5 10000 5000 64 lj55_160K   & # 640K
bash script/lj55/collect.sh 6 10000 5000 128 lj55_320K  & # 1280K
bash script/lj55/collect.sh 7 10000 5000 256 lj55_640K  & # 2560K
wait
