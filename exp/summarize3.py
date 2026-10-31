import pandas as pd,glob
d=pd.concat(pd.read_csv(f) for f in glob.glob('results/f*_s*.csv'));M=['AP','AUC','F1','Precision','Recall']
P={'PatchTST':13154,'PatchTST-RBF':21411,'TabM':48656}
pf=d.groupby(['fold','model'])[M].mean().reset_index();pf.to_csv('results/per_fold_3models.csv',index=False)
s=pf.groupby('model')[M].agg(['mean','std']);t=d.groupby('model').seconds.agg(['mean','sum'])
rows=[[m,P[m],f"{t.loc[m,'mean']:.0f}"]+[f"{s.loc[m,(k,'mean')]:.4f} ± {s.loc[m,(k,'std')]:.4f}" for k in M] for m in s.index]
w=pf.pivot(index='fold',columns='model',values='AP')
L=['# 3 mô hình: phát hiện bất thường tại t, SKAB, 5 fold × 3 seed\n','## Mean ± std qua 5 fold\n',pd.DataFrame(rows,columns=['model','params','s/run']+M).to_markdown(index=False),'\n## AP theo fold\n',w.round(4).to_markdown()]
for m in ['PatchTST-RBF','TabM']:L.append(f"- {m} thắng PatchTST (AP): {int((w[m]>w['PatchTST']).sum())}/5 fold, gain {(w[m]-w['PatchTST']).mean():+.4f}")
L.append(f"- TabM vs RBF (AP): TabM thắng {int((w['TabM']>w['PatchTST-RBF']).sum())}/5")
open('results/summary_3models.md','w').write('\n'.join(L));print('\n'.join(L))
