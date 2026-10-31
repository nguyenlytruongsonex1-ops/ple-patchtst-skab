cd "$(dirname "$0")"
for s in 0 1 2 3 4; do for f in 0 1 2 3 4; do for m in TabM PatchTST PatchTST-wide PatchTST-PLE-input DLinear; do echo "$f $m $s"; done; done; done | xargs -P 8 -n 3 sh -c 'python3 eval5.py --fold $0 --model $1 --seed $2 >> logs/s5_f$0.log 2>&1'
echo ALLDONE > logs/s5_done
