"""Detection at t: DLinear, PatchTST, PatchTST-wide, PatchTST-PLE-input, TabM; patch 8 stride 4, grouped 5-fold, 3 seeds. Adapted from source/run_forecast.py."""
import sys,json,time,copy,random,argparse
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import average_precision_score,precision_recall_curve,f1_score,precision_score,recall_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'vendor'))
from layers.PatchTST_backbone import PatchTST_backbone
from fast_attention import enable
from tabm import TabM
FEATURES=['Accelerometer1RMS','Accelerometer2RMS','Current','Pressure','Temperature','Thermocouple','Voltage','Volume Flow RateRMS']
LR,DROP,EPOCHS,BS=1e-3,0.1,30,256
WIDE_DFF=191
class PLEInput(nn.Module):  # x' = x + W_c . PLE(x) per channel, 8 equal bins on [-2,2], W zero-init, no bias, no norm
    def __init__(self,C=8,B=8):
        super().__init__();e=torch.linspace(-2,2,B+1);self.register_buffer('lo',e[:-1]);self.register_buffer('w',e[1:]-e[:-1]);self.W=nn.Parameter(torch.zeros(C,B))
    def forward(self,x):  # B,L,C
        z=((x.unsqueeze(-1)-self.lo)/self.w).clamp(0,1);return x+(z*self.W).sum(-1)
class DLinear(nn.Module):  # moving-avg(25) decomposition, per-channel Linear(60->60) on trend & seasonal, summed -> head
    def __init__(self,drop,L=60,C=8,k=25):
        super().__init__();self.k=k;self.trend=nn.Linear(L,L);self.seas=nn.Linear(L,L);self.head=nn.Sequential(nn.Flatten(1),nn.Dropout(drop),nn.Linear(L*C,1))
    def forward(self,x):  # x: B,L,C
        z=x.transpose(1,2);p=(self.k-1)//2
        tr=nn.functional.avg_pool1d(torch.cat([z[...,:1].repeat(1,1,p),z,z[...,-1:].repeat(1,1,p)],-1),self.k,1)
        return self.head(self.trend(tr)+self.seas(z-tr))
class Model(nn.Module):
 def __init__(self,name,drop):
  super().__init__();self.patch=name not in('TabM','DLinear')
  self.dl=name=='DLinear'
  if self.dl:self.net=DLinear(drop);self.n_patches=None;return
  if not self.patch:  # same as source/run_forecast.py
   self.net=TabM.make(n_num_features=480,d_out=1,n_blocks=2,d_block=64,dropout=drop,k=16);self.n_patches=None;return
  self.net=PatchTST_backbone(c_in=8,context_window=60,target_window=1,patch_len=8,stride=4,padding_patch='end',n_layers=1,d_model=32,n_heads=4,d_ff=WIDE_DFF if name=='PatchTST-wide' else 64,norm='LayerNorm',dropout=drop,revin=False)
  self.n_patches=self.net.head_nf//32
  self.pin=PLEInput() if name=='PatchTST-PLE-input' else None
  self.net.head=nn.Sequential(nn.Flatten(1),nn.Dropout(drop),nn.Linear(8*self.net.head_nf,1))
  enable(self)
 def forward(self,x):return self.net(x).reshape(len(x),-1) if self.dl else self.net((self.pin(x) if self.pin is not None else x).transpose(1,2) if self.patch else x.flatten(1)).reshape(len(x),-1)
@torch.no_grad()
def predict(m,x):m.eval();return torch.cat([m(b).sigmoid().mean(1) for b in x.split(512)]).numpy()
def prepare(names):
 frames={};raw={}
 for n in names:
  f=pd.read_csv(ROOT/'data'/n,sep=';');frames[n]=f
  f.index=pd.to_datetime(f.datetime);g=f.reindex(pd.date_range(f.index.min(),f.index.max(),freq='s'))
  x=g[FEATURES].ffill(limit=1).to_numpy(dtype='float32');y=g.anomaly.to_numpy()
  raw[n]=[(x[t-59:t+1],int(y[t])) for t in range(59,len(g),4) if np.isfinite(x[t-59:t+1]).all() and np.isfinite(y[t])]
 return frames,raw
def train(name,seed,xtr,ytr,xv,yv):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);m=Model(name,DROP)
 opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=1e-4);gen=torch.Generator().manual_seed(seed+100);best=-1;hist=[]
 for ep in range(1,EPOCHS+1):
  m.train()
  for idx in torch.randperm(len(xtr),generator=gen).split(BS):
   lo=m(xtr[idx]);loss=nn.functional.binary_cross_entropy_with_logits(lo,ytr[idx,None].expand_as(lo));opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  s=average_precision_score(yv,predict(m,xv));hist.append(s)
  if s>best:best,state,be=s,copy.deepcopy(m.state_dict()),ep
 m.load_state_dict(state);return m,best,be
def run(fold,name,seeds):
 torch.set_num_threads(1);out=ROOT/'results';out.mkdir(exist_ok=True)
 sp=json.loads((ROOT/'splits.json').read_text())['folds'][fold]
 frames,raw=prepare(sp['train']+sp['val']+sp['test'])
 sc=StandardScaler().fit(np.concatenate([frames[n][FEATURES].values for n in sp['train']]))
 D={}
 for s in ['train','val','test']:
  it=[z for n in sp[s] for z in raw[n]];xs=((np.stack([z[0] for z in it])-sc.mean_)/sc.scale_).astype('float32')
  D[s]=(torch.tensor(xs),torch.tensor([z[1] for z in it],dtype=torch.float32))
 (xtr,ytr),(xv,yv),(xt,yt)=D['train'],D['val'],D['test']
 for seed in seeds:
  t0=time.time();m,vap,be=train(name,seed,xtr,ytr,xv,yv);pv=predict(m,xv);pt=predict(m,xt)
  p,r,th=precision_recall_curve(yv,pv);fs=2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-12);thr=float(th[fs.argmax()]);pr=pt>=thr
  row=dict(fold=fold,model=name,seed=seed,params=sum(p.numel() for p in m.parameters()),n_patches=m.n_patches,n_train=len(ytr),n_val=len(yv),n_test=len(yt),pos_test=float(yt.mean()),val_AP=vap,best_epoch=be,threshold=thr,
   AP=average_precision_score(yt,pt),AUC=roc_auc_score(yt,pt),F1=f1_score(yt,pr),Precision=precision_score(yt,pr,zero_division=0),Recall=recall_score(yt,pr),W_abs_mean=float(m.pin.W.abs().mean()) if getattr(m,'pin',None) is not None else None,W_abs_max=float(m.pin.W.abs().max()) if getattr(m,'pin',None) is not None else None,seconds=time.time()-t0)
  (out/'preds').mkdir(exist_ok=True);np.savez_compressed(out/'preds'/f'f{fold}_{name}_s{seed}.npz',y=yt.numpy(),p=pt,thr=thr,**({'W':m.pin.W.detach().numpy()} if getattr(m,'pin',None) is not None else {}))
  pd.DataFrame([row]).to_csv(out/f'f{fold}_{name}_s{seed}.csv',index=False);print(json.dumps(row),flush=True)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--fold',type=int);a.add_argument('--model');a.add_argument('--seeds',default='0,1,2');a=a.parse_args()
 run(a.fold,a.model,[int(s) for s in a.seeds.split(',')])
