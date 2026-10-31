import pandas as pd,glob,numpy as np
d=pd.concat(pd.read_csv(f) for f in glob.glob('results/f*_s*.csv'))
npos=d.n_test*d.pos_test;tp=d.Recall*npos;fp=np.where(d.Precision>0,tp*(1-d.Precision)/d.Precision,0);d['FPR']=fp/(d.n_test-npos)
d.to_csv('results/all_runs.csv',index=False)
M=['AP','AUC','F1','Precision','Recall','FPR'];O=['DLinear','PatchTST','PatchTST-wide','PatchTST-PLE','TabM','PatchTST-RBF']
P={'DLinear':7801,'PatchTST':13154,'PatchTST-wide':21409,'PatchTST-PLE':21411,'TabM':48656,'PatchTST-RBF':21411}
pf=d.groupby(['fold','model'])[M].mean().reset_index();s=pf.groupby('model')[M].agg(['mean','std']);t=d.groupby('model').seconds.mean()
rows=[[m+(' (tham khảo)' if m.endswith('RBF') else ''),P[m],f"{t[m]:.0f}"]+[f"{s.loc[m,(k,'mean')]:.4f} ± {s.loc[m,(k,'std')]:.4f}" for k in M] for m in O]
w=pf.pivot(index='fold',columns='model',values='AP')[O]
L=['# 5 mô hình (DLinear, PatchTST, wide, PLE, TabM; RBF tham khảo) — SKAB phát hiện tại t, patch 8/4, 5 fold × 3 seed\n',pd.DataFrame(rows,columns=['model','params','s/run']+M).to_markdown(index=False),'\n## AP theo fold\n',w.round(4).to_markdown(),'']
import numpy as np
for a in ['DLinear','PatchTST-PLE','PatchTST-wide','TabM']:
  for b in ['DLinear','PatchTST','PatchTST-wide']:
    if a!=b:L.append(f"- {a} thắng {b}: {int((w[a]>w[b]).sum())}/5 fold, chênh AP TB {(w[a]-w[b]).mean():+.4f}")
L.append('\n## FPR thật DLinear (đếm từ dự đoán test đã lưu, ngưỡng F1 chọn trên val)')
import glob
fp=[]
for f in sorted(glob.glob('results/preds/f*_DLinear_s*.npz')):
  z=np.load(f);pr=z['p']>=z['thr'];y=z['y']>0.5;fp.append(((pr&~y).sum()/(~y).sum()))
L.append(f"- FPR thật DLinear: TB {np.mean(fp):.4f} (n={len(fp)} lượt); FPR suy ra trong bảng: {d[d.model=='DLinear'].FPR.mean():.4f}")
L.append('- PLE alpha: '+', '.join(f"f{r.fold}s{r.seed}={r.alpha:.4f}" for r in d[d.model=='PatchTST-PLE'].sort_values(['fold','seed']).itertuples()))
open('results/summary_5models.md','w').write('\n'.join(L));print('\n'.join(L))
