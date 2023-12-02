

for i in $(ls -t /p/home/jusers/porian1/juwels/porian1/open_lm_fork/exps/lm_grid_wd); do
  sbatch scripts/eval_concurrent.sh "$i"
done

# echo "DONE DONE DONE!"