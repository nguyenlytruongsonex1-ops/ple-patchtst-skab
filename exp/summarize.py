import pandas as pd,glob
d=pd.concat(pd.read_csv(f) for f in glob.glob('results/f*_s*.csv'));M=['AP','AUC','F1','Precision','Recall']
pf=d.groupby(['fold','model'])[M+['alpha']].mean().reset_index();pf.to_csv('results/per_fold.csv',index=False)
L=['# Kết quả: phát hiện bất thường tại t, SKAB, patch 8/4, 5 fold × 3 seed\n','## Theo fold (mean 3 seed)\n',pf.round(4).to_markdown(index=False),'\n## Mean ± std qua 5 fold\n']
s=pf.groupby('model')[M].agg(['mean','std']);rows=[[m]+[f"{s.loc[m,(k,'mean')]:.4f} ± {s.loc[m,(k,'std')]:.4f}" for k in M] for m in s.index]
L.append(pd.DataFrame(rows,columns=['model']+M).to_markdown(index=False))
w=pf.pivot(index='fold',columns='model',values='AP');wins=int((w['PatchTST-RBF']>w['PatchTST']).sum());gain=(w['PatchTST-RBF']-w['PatchTST']).mean()
sd=s.loc['PatchTST',('AP','std')];ok=gain>=sd and wins>=4
L+=[f"\n- RBF thắng AP: {wins}/5 fold; AP gain trung bình {gain:+.4f}; std AP baseline {sd:.4f}",f"- Đạt tiêu chí (gain ≥ 1 std VÀ ≥4/5 fold): {'CÓ' if ok else 'KHÔNG'}",
 "- alpha học được (RBF): "+', '.join(f"f{r.fold}s{r.seed}={r.alpha:.4f}" for r in d[d.model=='PatchTST-RBF'].sort_values(['fold','seed']).itertuples())]
open('results/summary.md','w').write('\n'.join(L));print('\n'.join(L))
