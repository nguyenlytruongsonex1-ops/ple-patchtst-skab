"""D11: 5 models x 5 folds x 5 seeds, identical fixed setup (run_detect.train). Load ckpt if present (same code, D10) else train+save. Direct FPR. Masking for 3 models."""
import json,time,argparse;import numpy as np,pandas as pd,torch
from sklearn.metrics import average_precision_score,precision_recall_curve,f1_score,precision_score,recall_score,roc_auc_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import run_detect as R
ROOT=R.ROOT;LEVELS=[0.0,0.1,0.3,0.5];MASK={'PatchTST','PatchTST-wide','PatchTST-PLE-input'}
def main(fold,name,seed):
 torch.set_num_threads(1);f=ROOT/'results'/'s5'/f'f{fold}_{name}_s{seed}.csv'
 if f.exists():return
 sp=json.loads((ROOT/'splits.json').read_text())['folds'][fold];frames,raw=R.prepare(sp['train']+sp['val']+sp['test'])
 sc=StandardScaler().fit(np.concatenate([frames[n][R.FEATURES].values for n in sp['train']]));D={}
 for s in ['train','val','test']:
  it=[z for n in sp[s] for z in raw[n]];xs=((np.stack([z[0] for z in it])-sc.mean_)/sc.scale_).astype('float32')
  D[s]=(torch.tensor(xs),torch.tensor([z[1] for z in it],dtype=torch.float32))
 (xtr,ytr),(xv,yv),(xt,yt)=D['train'],D['val'],D['test'];ck=ROOT/'ckpt'/f'f{fold}_{name}_s{seed}.pt';t0=time.time()
 if ck.exists():m=R.Model(name,R.DROP);m.load_state_dict(torch.load(ck));src='ckpt'
 else:m,_,_=R.train(name,seed,xtr,ytr,xv,yv);torch.save(m.state_dict(),ck);src='trained'
 pv=R.predict(m,xv);p,r,th=precision_recall_curve(yv,pv);fs=2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-12);thr=float(th[fs.argmax()]);y=yt.numpy()
 def met(pt):
  pr=pt>=thr;tn,fp,fn,tp=confusion_matrix(y,pr,labels=[0,1]).ravel()
  return dict(AP=average_precision_score(y,pt),AUC=roc_auc_score(y,pt),F1=f1_score(y,pr),Precision=precision_score(y,pr,zero_division=0),Recall=recall_score(y,pr),FPR=fp/(fp+tn))
 pt=R.predict(m,xt);np.savez_compressed(ROOT/'results'/'preds5'/f'f{fold}_{name}_s{seed}.npz',y=y,p=pt,thr=thr)
 row=dict(fold=fold,model=name,seed=seed,params=sum(q.numel() for q in m.parameters()),src=src,val_AP=average_precision_score(yv,pv),thr=thr,**met(pt),seconds=time.time()-t0)
 if name in MASK:
  rows=[]
  for li,lv in enumerate(LEVELS):
   for rep in range(3 if lv>0 else 1):
    g=torch.Generator().manual_seed(10000*fold+100*li+rep);x=xt.clone();x[torch.rand(x.shape,generator=g)<lv]=0.0
    rows.append(dict(fold=fold,model=name,seed=seed,level=lv,rep=rep,**met(R.predict(m,x))))
  pd.DataFrame(rows).to_csv(ROOT/'results'/'masking5'/f'f{fold}_{name}_s{seed}.csv',index=False)
 pd.DataFrame([row]).to_csv(f,index=False);print(json.dumps(row),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--fold',type=int);a.add_argument('--model');a.add_argument('--seed',type=int);a=a.parse_args();main(a.fold,a.model,a.seed)
