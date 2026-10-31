import json,numpy as np,pandas as pd,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from pathlib import Path;from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score as AP,roc_auc_score as AUC
from sklearn.linear_model import Ridge
R=Path('..');F=['Accelerometer1RMS','Accelerometer2RMS','Current','Pressure','Temperature','Thermocouple','Voltage','Volume Flow RateRMS']
S=json.loads((R/'splits.json').read_text());names=sorted({n for f in S['folds'] for k in('train','val','test') for n in f[k]})
def load(n):
 f=pd.read_csv(R/'data'/n,sep=';');f.index=pd.to_datetime(f.datetime)
 g=f.reindex(pd.date_range(f.index.min(),f.index.max(),freq='s'));g[F]=g[F].ffill(limit=1);return g
D={n:load(n) for n in names};out={}
# (1) effect size per file
rows=[]
for n,g in D.items():
 a=g[g.anomaly==1];b=g[g.anomaly==0]
 if len(a)<10 or len(b)<10:continue
 for c in F:
  sd=np.sqrt((a[c].var()+b[c].var())/2)+1e-12
  rows.append(dict(file=n,sensor=c,d=(a[c].mean()-b[c].mean())/sd,ks=ks_2samp(a[c].dropna(),b[c].dropna()).statistic))
E=pd.DataFrame(rows);E.to_csv('effect_per_file.csv',index=False)
agg=E.groupby('sensor').agg(median_abs_d=('d',lambda x:np.median(abs(x))),median_d=('d','median'),frac_pos=('d',lambda x:(x>0).mean()),median_ks=('ks','median'),files=('d','size')).sort_values('median_ks',ascending=False)
agg.to_csv('effect_agg.csv');print(agg.round(3))
# lead time: z vs baseline [-120,-60) before onset, rolling 5s mean
prof={c:[] for c in F};lead=[]
for n,g in D.items():
 y=g.anomaly.fillna(0).values;on=np.where((y[1:]==1)&(y[:-1]==0))[0]+1
 X=g[F].values
 for o in on:
  if o<120 or o+30>len(g) or np.nansum(y[o-120:o])>0:continue
  base=X[o-120:o-60];mu=np.nanmean(base,0);sd=np.nanstd(base,0)+1e-9
  z=(pd.DataFrame((X[o-120:o+30]-mu)/sd).rolling(5).mean().values)
  for i,c in enumerate(F):prof[c].append(z[:,i])
  pre=np.abs(z[60:120]);hit=np.where((pre>3).any(1))[0]
  lead.append(dict(file=n,onset=o,first_shift_s_before=(60-hit[0]) if len(hit) else 0,sensor=F[(pre[hit[0]]>3).argmax()] if len(hit) else None))
ctl=[]
rng=np.random.default_rng(0)
for n,g in D.items():
 y=g.anomaly.fillna(1).values;X=g[F].values
 cand=[o for o in range(120,len(g)-30) if y[o-120:o+30].sum()==0]
 for o in (rng.choice(cand,min(20,len(cand)),replace=False) if cand else []):
  base=X[o-120:o-60];mu=np.nanmean(base,0);sd=np.nanstd(base,0)+1e-9
  z=pd.DataFrame((X[o-120:o+30]-mu)/sd).rolling(5).mean().values
  ctl.append(bool((np.abs(z[60:120])>3).any()))
out['ctl']=(len(ctl),float(np.mean(ctl)));print('control normal points shift rate',out['ctl'])
L=pd.DataFrame(lead);L.to_csv('lead_time.csv',index=False)
print('onsets',len(L),'with shift>3sd in 60s before:',(L.first_shift_s_before>0).sum(),'median s before',L.first_shift_s_before[L.first_shift_s_before>0].median())
out['lead']=dict(n=len(L),shift=int((L.first_shift_s_before>0).sum()),med=float(L.first_shift_s_before[L.first_shift_s_before>0].median()) if (L.first_shift_s_before>0).any() else None)
t=np.arange(-120,30);fig,ax=plt.subplots(2,4,figsize=(16,6))
for a,c in zip(ax.flat,F):
 P=np.array(prof[c]);a.plot(t,np.nanmedian(P,0));a.fill_between(t,np.nanpercentile(P,25,0),np.nanpercentile(P,75,0),alpha=.3);a.axvline(0,c='r');a.set_title(c);a.set_ylim(-10,10)
fig.suptitle('z-score quanh thời điểm bắt đầu bất thường (median, IQR)');fig.tight_layout();fig.savefig('onset_profiles.png',dpi=90)
fig,a=plt.subplots(figsize=(8,4));agg[['median_abs_d','median_ks']].plot.bar(ax=a);a.set_title('Effect size per sensor (median over files)');fig.tight_layout();fig.savefig('effect_size.png',dpi=90)
# ACF on normal data
fig,a=plt.subplots(figsize=(8,4));acf={}
for c in F:
 v=[];
 for g in D.values():
  s=g[c][g.anomaly==0].dropna().values
  if len(s)>200:s=s-s.mean();v.append([np.corrcoef(s[:-k],s[k:])[0,1] for k in range(1,31)])
 acf[c]=np.nanmean(v,0);a.plot(range(1,31),acf[c],label=c)
a.legend(fontsize=7);a.set_xlabel('lag (s)');a.set_title('ACF (normal)');fig.tight_layout();fig.savefig('acf.png',dpi=90)
pd.DataFrame(acf,index=range(1,31)).loc[[1,5,10,30]].round(3).to_csv('acf.csv');print(pd.read_csv('acf.csv'))
# (2)(3) forecasting
P=10;H=[1,5,10]
def mk(g,mu,sd,h):
 X=((g[F].values-mu)/sd);y=g.anomaly.values;N=len(X);idx=np.arange(P-1,N-h)
 Z=np.stack([X[i-P+1:i+1].ravel() for i in idx]);T=X[idx+h];last=X[idx]
 lab=y[idx+h];ok=np.isfinite(Z).all(1)&np.isfinite(T).all(1)&np.isfinite(lab);lw=np.array([np.nanmax(y[max(0,i-P+1):i+h+1]) for i in idx])
 return Z[ok],T[ok],last[ok],lab[ok],lw[ok]
res=[];det=[]
for fi,f in enumerate(S['folds']):
 tr=pd.concat([D[n] for n in f['train']]);mu=tr[F].mean().values;sd=tr[F].std().values+1e-9
 for h in H:
  A=[mk(D[n],mu,sd,h) for n in f['train']];Z=np.concatenate([a[0][a[4]==0] for a in A]);T=np.concatenate([a[1][a[4]==0] for a in A])
  m=Ridge(1.0).fit(Z,T-Z[:,-8:])
  sc=[];yy=[]
  B=[mk(D[n],mu,sd,h) for n in f['test']]
  for (z,t,l,lab,lw) in B:
   pr=l+m.predict(z);e=t-pr;e0=t-l
   nm=lw==0
   for k,c in enumerate(F):
    for nmn,ee in (('naive',e0),('AR10',e)):
     res.append(dict(fold=fi,h=h,sensor=c,model=nmn,rmse=np.sqrt(np.mean(ee[nm,k]**2)),mae=np.mean(abs(ee[nm,k])),sst=np.var(t[nm,k]),sse=np.mean(ee[nm,k]**2),n=nm.sum()))
   sc.append(pd.Series((e**2).sum(1)).rolling(5,min_periods=1).mean().values);yy.append(lab)
  sc=np.concatenate(sc);yy=np.concatenate(yy)
  det.append(dict(fold=fi,h=h,AP=AP(yy,sc),ROC=AUC(yy,sc),pos=yy.mean()))
Rz=pd.DataFrame(res);g=Rz.groupby(['h','model','sensor']).apply(lambda d:pd.Series(dict(RMSE=np.sqrt((d.sse*d.n).sum()/d.n.sum()),MAE=(d.mae*d.n).sum()/d.n.sum(),R2=1-(d.sse*d.n).sum()/(d.sst*d.n).sum())))
g.round(3).to_csv('forecast_metrics.csv');print(g.round(3).to_string())
Dt=pd.DataFrame(det);Dt.to_csv('residual_detect.csv',index=False);print(Dt.round(3));print(Dt.groupby('h')[['AP','ROC','pos']].agg(['mean','std']).round(3))
