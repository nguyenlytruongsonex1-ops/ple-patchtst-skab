import numpy as np,pandas as pd
F=['Accelerometer1RMS','Accelerometer2RMS','Current','Pressure','Temperature','Thermocouple','Voltage','Volume Flow RateRMS']
g=pd.read_csv('../data/anomaly-free/anomaly-free.csv',sep=';');X=g[F].values;rng=np.random.default_rng(0);r=[];s=[]
for o in rng.choice(np.arange(120,len(X)-30),300,replace=False):
  base=X[o-120:o-60];z=pd.DataFrame((X[o-120:o+30]-base.mean(0))/(base.std(0)+1e-9)).rolling(5).mean().values
  pre=np.abs(z[60:120])>3;r.append(pre.any());
  if pre.any():s.append(F[pre[pre.any(1).argmax()].argmax()])
print('anomaly-free control: false-shift rate',np.mean(r),len(X),'rows');print(pd.Series(s).value_counts())
