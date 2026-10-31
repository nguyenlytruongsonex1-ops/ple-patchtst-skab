import pandas as pd,numpy as np,glob,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
d=pd.concat(pd.read_csv(f) for f in glob.glob('results/masking/*.csv'))
M=['PatchTST','PatchTST-wide','PatchTST-PLE-input'];L=sorted(d.level.unique())
# average over mask repeats & seeds -> per fold; std over folds; also seed-level std
fm=d.groupby(['model','fold','level'])[['AP','AUC','F1','FPR']].mean().reset_index()
o=["# D10 – Che ngẫu nhiên đầu vào test (SKAB)","",f"Số lượt: {d.groupby(['model','fold','seed']).ngroups} (cần 45). Mean ± std qua 5 fold (mỗi fold = trung bình 3 seed × 3 lần che). Ngưỡng giữ nguyên từ val không che.",""]
for k in ['AP','AUC','F1','FPR']:
 o+=[f"## {k}","","| Mức che | "+" | ".join(M)+" |","|---|"+"---|"*len(M)]
 for lv in L:o.append(f"| {int(lv*100)}% | "+" | ".join(f"{fm[(fm.model==m)&(fm.level==lv)][k].mean():.3f} ± {fm[(fm.model==m)&(fm.level==lv)][k].std():.3f}" for m in M)+" |")
 o.append("")
o+=["## Suy giảm AP (0% → 50%) và thắng theo fold","","| Mô hình | ΔAP mean | std qua fold | std qua seed (TB fold) |","|---|---|---|---|"]
sm=d.groupby(['model','fold','seed','level']).AP.mean().unstack()
for m in M:
 a=fm[fm.model==m].pivot(index='fold',columns='level',values='AP');dl=a[0.5]-a[0.0]
 ds=(sm.loc[m][0.5]-sm.loc[m][0.0]).groupby('fold').std().mean();o.append(f"| {m} | {dl.mean():+.3f} | {dl.std():.3f} | {ds:.3f} |")
o.append("")
P=fm.pivot_table(index=['fold','level'],columns='model',values='AP')
for b in ['PatchTST','PatchTST-wide']:
 o.append(f"- PLE-input thắng {b} theo AP: "+", ".join(f"{int(lv*100)}%: {(P.xs(lv,level='level')['PatchTST-PLE-input']>P.xs(lv,level='level')[b]).sum()}/5" for lv in L))
 dd=(P.xs(0.5,level='level')['PatchTST-PLE-input']-P.xs(0.0,level='level')['PatchTST-PLE-input'])-(P.xs(0.5,level='level')[b]-P.xs(0.0,level='level')[b])
 o.append(f"  - Chênh độ dốc (ΔAP PLE-input − ΔAP {b}): {dd.mean():+.3f} ± {dd.std():.3f} qua fold; fold PLE-input suy giảm ít hơn: {(dd>0).sum()}/5")
open('results/summary_masking.md','w').write("\n".join(o)+"\n");print("\n".join(o))
fig,ax=plt.subplots(1,3,figsize=(13,3.8))
for i,k in enumerate(['AP','F1','FPR']):
 for m in M:
  g=fm[fm.model==m].groupby('level')[k];ax[i].errorbar(np.array(L)*100,g.mean(),g.std(),marker='o',capsize=3,label=m)
 ax[i].set_xlabel('% giá trị bị che');ax[i].set_title(k+' (mean ± std qua 5 fold)')
ax[0].legend();plt.tight_layout();plt.savefig('results/masking.png',dpi=130)
