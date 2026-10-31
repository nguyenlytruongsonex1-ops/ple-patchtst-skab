import pandas as pd,glob,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
d=pd.concat(pd.read_csv(f) for f in glob.glob('results/s5/*.csv'));M=['DLinear','PatchTST','PatchTST-wide','PatchTST-PLE-input','TabM']
K=['AP','AUC','F1','Precision','Recall','FPR'];fm=d.groupby(['model','fold'])[K].mean().reset_index()
o=["# SKAB – 5 mô hình × 5 fold × 5 seed (D11)","",f"Số lượt: {len(d)} (cần 125). Mean ± std qua 5 fold (mỗi fold = TB 5 seed). FPR đếm trực tiếp từ dự đoán test (results/preds5).","","| Mô hình | Tham số | "+" | ".join(K)+" |","|---|---|"+"---|"*len(K)]
for m in M:
 g=fm[fm.model==m];o.append(f"| {m} | {int(d[d.model==m].params.iloc[0])} | "+" | ".join(f"{g[k].mean():.3f} ± {g[k].std():.3f}" for k in K)+" |")
o+=["","Std AP qua seed (TB trên fold): "+", ".join(f"{m} {d[d.model==m].groupby('fold').AP.std().mean():.3f}" for m in M),"","## AP theo fold","","| Fold | "+" | ".join(M)+" |","|---|"+"---|"*len(M)]
P=fm.pivot(index='fold',columns='model',values='AP')[M]
for f,r in P.iterrows():o.append(f"| {f} | "+" | ".join(f"{v:.3f}" for v in r)+" |")
o+=["","## PLE-input so với từng mô hình (AP)",""]
for m in M:
 if m=='PatchTST-PLE-input':continue
 df=P['PatchTST-PLE-input']-P[m];o.append(f"- vs {m}: thắng {(df>0).sum()}/5 fold, chênh {df.mean():+.3f} (std của {m} qua fold {P[m].std():.3f}); seed-level thắng {(d[d.model=='PatchTST-PLE-input'].set_index(['fold','seed']).AP.sort_index()>d[d.model==m].set_index(['fold','seed']).AP.sort_index()).sum()}/25")
src=d.src.value_counts().to_dict();o+=["",f"Nguồn mô hình: {src}"]
# repro check vs old csvs (seeds 0-2)
old=[];import os
for _,r in d[d.seed<3].iterrows():
 fn=f"results/f{r.fold}_{r.model}_s{r.seed}.csv"
 if os.path.exists(fn):old.append(abs(pd.read_csv(fn).AP.iloc[0]-r.AP))
o.append(f"Kiểm tra tái lập seed 0–2 so với CSV cũ: {len(old)} lượt, |ΔAP| lớn nhất = {max(old):.2e}" if old else "")
# masking
mk=pd.concat(pd.read_csv(f) for f in glob.glob('results/masking5/*.csv'));MM=['PatchTST','PatchTST-wide','PatchTST-PLE-input']
mf=mk.groupby(['model','fold','level'])[['AP','AUC','F1','FPR']].mean().reset_index();L=sorted(mk.level.unique())
o+=["","## Che ngẫu nhiên test (5 seed × 3 lần che)","","| Mức | "+" | ".join(f"{m} AP" for m in MM)+" | PLE-in thắng base | PLE-in thắng wide |","|---|---|---|---|---|---|"]
Q=mf.pivot_table(index=['fold','level'],columns='model',values='AP')
for lv in L:
 q=Q.xs(lv,level='level');o.append(f"| {int(lv*100)}% | "+" | ".join(f"{q[m].mean():.3f} ± {q[m].std():.3f}" for m in MM)+f" | {(q['PatchTST-PLE-input']>q['PatchTST']).sum()}/5 | {(q['PatchTST-PLE-input']>q['PatchTST-wide']).sum()}/5 |")
o.append("")
for k in ['F1','FPR']:
 o.append(f"{k}: "+"; ".join(f"{int(lv*100)}%: "+"/".join(f"{mf[(mf.model==m)&(mf.level==lv)][k].mean():.3f}" for m in MM) for lv in L)+" (base/wide/PLE-in)")
o.append("")
for b in ['PatchTST','PatchTST-wide']:
 dd=(Q.xs(.5,level='level')['PatchTST-PLE-input']-Q.xs(0,level='level')['PatchTST-PLE-input'])-(Q.xs(.5,level='level')[b]-Q.xs(0,level='level')[b])
 o.append(f"- Chênh độ dốc AP 0→50% (PLE-in − {b}): {dd.mean():+.3f} ± {dd.std():.3f}; PLE-in giảm ít hơn ở {(dd>0).sum()}/5 fold")
o.append("- ΔAP 0→50%: "+", ".join(f"{m} {(Q.xs(.5,level='level')[m]-Q.xs(0,level='level')[m]).mean():+.3f}" for m in MM))
open('results/summary_5models_5seeds.md','w').write("\n".join(o)+"\n");print("\n".join(o))
fig,ax=plt.subplots(1,3,figsize=(13,3.8))
for i,k in enumerate(['AP','F1','FPR']):
 for m in MM:
  g=mf[mf.model==m].groupby('level')[k];ax[i].errorbar(np.array(L)*100,g.mean(),g.std(),marker='o',capsize=3,label=m)
 ax[i].set_xlabel('% giá trị bị che');ax[i].set_title(k+' (5 seed, mean ± std qua fold)')
ax[0].legend();plt.tight_layout();plt.savefig('results/masking_5seeds.png',dpi=130)
