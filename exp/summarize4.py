import pandas as pd,glob,numpy as np
d=pd.concat(pd.read_csv(f) for f in glob.glob('results/f*_s*.csv'))
npos=d.n_test*d.pos_test;tp=d.Recall*npos;fp=np.where(d.Precision>0,tp*(1-d.Precision)/d.Precision,0);d['FPR']=fp/(d.n_test-npos)
d.to_csv('results/all_runs.csv',index=False)
M=['AP','AUC','F1','Precision','Recall','FPR'];O=['PatchTST','PatchTST-wide','PatchTST-PLE','TabM','PatchTST-RBF']
P={'PatchTST':13154,'PatchTST-wide':21409,'PatchTST-PLE':21411,'TabM':48656,'PatchTST-RBF':21411}
pf=d.groupby(['fold','model'])[M].mean().reset_index();s=pf.groupby('model')[M].agg(['mean','std']);t=d.groupby('model').seconds.mean()
rows=[[m+(' (tham khảo)' if m.endswith('RBF') else ''),P[m],f"{t[m]:.0f}"]+[f"{s.loc[m,(k,'mean')]:.4f} ± {s.loc[m,(k,'std')]:.4f}" for k in M] for m in O]
w=pf.pivot(index='fold',columns='model',values='AP')[O]
L=['# PLE vs baseline/wide/TabM — SKAB phát hiện tại t, patch 8/4, 5 fold × 3 seed\n',pd.DataFrame(rows,columns=['model','params','s/run']+M).to_markdown(index=False),'\n## AP theo fold\n',w.round(4).to_markdown(),'']
for a in ['PatchTST-PLE','PatchTST-wide','TabM']:
  for b in ['PatchTST','PatchTST-wide']:
    if a!=b:L.append(f"- {a} thắng {b}: {int((w[a]>w[b]).sum())}/5, gain {(w[a]-w[b]).mean():+.4f}")
m=lambda x:s.loc[x,('AP','mean')];sd=lambda x:s.loc[x,('AP','std')]
c1=all(m('PatchTST-PLE')-m(b)>=sd(b) for b in ['PatchTST','PatchTST-wide']);c2=all((w['PatchTST-PLE']>w[b]).sum()>=4 for b in ['PatchTST','PatchTST-wide'])
c3=sd('PatchTST-PLE')<=max(sd('PatchTST'),sd('PatchTST-wide'))
L.append(f"- Tiêu chí: gain≥1std cả hai={c1}; thắng≥4/5 cả hai={c2}; std không lớn hơn={c3} → {'ĐẠT' if (c1 or c2) and c3 else 'KHÔNG ĐẠT'}")
a=d[d.model=='PatchTST-PLE'].sort_values(['fold','seed']);L.append('- alpha PLE: '+', '.join(f"f{r.fold}s{r.seed}={r.alpha:.4f}" for r in a.itertuples()))
open('results/summary_ple.md','w').write('\n'.join(L));print('\n'.join(L))
