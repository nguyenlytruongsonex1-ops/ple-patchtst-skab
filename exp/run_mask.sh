cd "$(dirname "$0")"
for f in 0 1 2 3 4; do for s in 0 1 2; do for m in PatchTST PatchTST-PLE-input PatchTST-wide; do echo "$f $m $s"; done; done; done | xargs -P 8 -n 3 sh -c 'python3 mask_eval.py --fold $0 --model $1 --seed $2 >> logs/mask_f$0.log 2>&1'
echo ALLDONE > logs/mask_done
