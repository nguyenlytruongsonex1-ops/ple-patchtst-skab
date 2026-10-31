import pandas as pd,glob,numpy as np
d=pd.concat(pd.read_csv(f) for f in glob.glob('results/f*_s*.csv'))
npos=d.n_test*d.pos_test;tp=d.Recall*npos;fp=np.where(d.Precision>0,tp*(1-d.Precision)/d.Precision,0);d['FPR']=fp/(d.n_test-npos)
d.to_csv('results/all_runs.csv',index=False)
N='PatchTST-PLE-input';M=['AP','AUC','F1','Precision','Recall','FPR'];O=['DLinear','PatchTST','PatchTST-wide','PatchTST-PLE',N,'TabM']
P=d.groupby('model').params.first().fillna({'PatchTST':13154,'PatchTST-RBF':21411})
pf=d.groupby(['fold','model'])[M].mean().reset_index();s=pf.groupby('model')[M].agg(['mean','std']);t=d.groupby('model').seconds.mean()
rows=[[m+(' (PLE-FFN)' if m=='PatchTST-PLE' else ''),P[m],f"{t[m]:.0f}"]+[f"{s.loc[m,(k,'mean')]:.4f} ± {s.loc[m,(k,'std')]:.4f}" for k in M] for m in O]
w=pf.pivot(index='fold',columns='model',values='AP')[O]
L=['# PatchTST-PLE-input — SKAB phát hiện tại t, cửa sổ 60 s, patch 8/4, 5 fold nhóm × 3 seed\n',
'x\' = x + W_c·PLE(x) theo từng kênh (8 bin đều [-2,2], W khởi tạo 0, không bias, không chuẩn hoá) → PatchTST gốc (1 lớp, d_model 32, 4 head, d_ff 64, RevIN tắt). Chỉ thêm 8×8 = 64 tham số (13,218 vs 13,154).\n',
pd.DataFrame(rows,columns=['model','params','s/run']+M).to_markdown(index=False),'\n(± = độ lệch chuẩn giữa 5 fold, trên trung bình 3 seed/fold. FPR suy ra từ P, R, tỉ lệ dương.)\n','## AP theo fold\n',w.round(4).to_markdown(),'\n## Số fold thắng (AP)']
for b in ['PatchTST','PatchTST-wide','PatchTST-PLE']:L.append(f"- PLE-input thắng {b}: {int((w[N]>w[b]).sum())}/5, chênh AP TB {(w[N]-w[b]).mean():+.4f} (std {b}: {s.loc[b,('AP','std')]:.4f})")
fpr=[];W=[]
for f in sorted(glob.glob(f'results/preds/f*_{N}_s*.npz')):
  z=np.load(f);pr=z['p']>=z['thr'];y=z['y']>0.5;fpr.append((pr&~y).sum()/(~y).sum());W.append(z['W'])
W=np.stack(W);L+=['\n## FPR thật (đếm từ dự đoán test)',f"- PLE-input: {np.mean(fpr):.4f} ± {np.std(fpr):.4f} (n={len(fpr)}); suy ra: {d[d.model==N].FPR.mean():.4f}",
'\n## Độ lớn W đã học (khởi tạo 0)',f"- |W| TB {np.abs(W).mean():.4f}, max {np.abs(W).max():.4f} (trên 15 lượt)",
'- |W| TB theo kênh: '+', '.join(f"{c}={v:.3f}" for c,v in zip(['Acc1','Acc2','Current','Pressure','Temp','Thermo','Voltage','Flow'],np.abs(W).mean((0,2))))]
open('results/summary_ple_input.md','w').write('\n'.join(L));print('\n'.join(L))
