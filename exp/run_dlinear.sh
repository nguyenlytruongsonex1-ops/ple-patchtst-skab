for f in 0 1 2 3 4; do for s in 0 1 2; do
 [ -f results/f${f}_DLinear_s$s.csv ] || python3 run_detect.py --fold $f --model DLinear --seeds $s >> logs/f${f}_DLinear.log 2>&1
done; done; echo ALLDONE >> logs/dlinear_done
