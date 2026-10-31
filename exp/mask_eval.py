"""D10: random-masking robustness. Retrain (identical fixed setup, run_detect.train) since no checkpoints were saved; save ckpt; mask test inputs only."""
import sys,json,time,argparse;from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.metrics import average_precision_score,precision_recall_curve,f1_score,roc_auc_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import run_detect as R
ROOT=R.ROOT;LEVELS=[0.0,0.1,0.3,0.5];REPS=3
def main(fold,name,seed):
 torch.set_num_threads(1);out=ROOT/'results'/'masking';f=out/f'f{fold}_{name}_s{seed}.csv'
 if f.exists():return
 sp=json.loads((ROOT/'splits.json').read_text())['folds'][fold];frames,raw=R.prepare(sp['train']+sp['val']+sp['test'])
 sc=StandardScaler().fit(np.concatenate([frames[n][R.FEATURES].values for n in sp['train']]));D={}
 for s in ['train','val','test']:
  it=[z for n in sp[s] for z in raw[n]];xs=((np.stack([z[0] for z in it])-sc.mean_)/sc.scale_).astype('float32')
  D[s]=(torch.tensor(xs),torch.tensor([z[1] for z in it],dtype=torch.float32))
 (xtr,ytr),(xv,yv),(xt,yt)=D['train'],D['val'],D['test']
 ck=ROOT/'ckpt'/f'f{fold}_{name}_s{seed}.pt'
 if ck.exists():m=R.Model(name,R.DROP);m.load_state_dict(torch.load(ck))
 else:m,_,_=R.train(name,seed,xtr,ytr,xv,yv);torch.save(m.state_dict(),ck)
 pv=R.predict(m,xv);p,r,th=precision_recall_curve(yv,pv);fs=2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-12);thr=float(th[fs.argmax()])
 rows=[];y=yt.numpy()
 for li,lv in enumerate(LEVELS):
  for rep in range(REPS if lv>0 else 1):
   g=torch.Generator().manual_seed(10000*fold+100*li+rep)  # same mask for all models/seeds within fold
   x=xt.clone();x[torch.rand(x.shape,generator=g)<lv]=0.0  # 0 = standardized mean; applied before PLE/patching
   pt=R.predict(m,x);pr=pt>=thr;tn,fp,fn,tp=confusion_matrix(y,pr,labels=[0,1]).ravel()
   rows.append(dict(fold=fold,model=name,seed=seed,level=lv,rep=rep,thr=thr,AP=average_precision_score(y,pt),AUC=roc_auc_score(y,pt),F1=f1_score(y,pr),FPR=fp/(fp+tn)))
 pd.DataFrame(rows).to_csv(f,index=False);print(name,fold,seed,rows[0]['AP'],rows[-1]['AP'],flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--fold',type=int);a.add_argument('--model');a.add_argument('--seed',type=int);a=a.parse_args();main(a.fold,a.model,a.seed)
