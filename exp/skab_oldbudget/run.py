"""Explicit 30-epoch SKAB-only protocol; does not change lightweight defaults.
Run: python -m exp.skab_oldbudget.run --output /workspace/results/unique-name
"""
import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, f1_score, precision_score, recall_score)
from exp.quantile_pilot.data_skab import load_skab_fold, groups_are_intact
from exp.quantile_pilot.ple import fit_revin_quantile_edges
from exp.quantile_pilot.models import build_model, predict
from exp.quantile_pilot.train import train_one

ROOT = Path(__file__).resolve().parents[2]
CONFIG = dict(model='PatchTST-PLE-quantile', folds=list(range(5)), seeds=list(range(5)),
              max_epochs=30, patience=None, batch_size=256, lr=1e-3,
              weight_decay=1e-4, grad_clip=1.0, threads=1,
              task='classification', lookback=60, window_stride=4,
              patch_len=8, patch_stride=4, n_layers=1, d_model=32,
              n_heads=4, d_ff=64, dropout=.1, actual_encoder_norm='BatchNorm',
              n_bins=8, revin_eps=1e-5, revin_affine=True,
              subtract_last=False, variance_unbiased=False, device='cpu',
              checkpoint='best validation AP', label='final anomaly',
              pos_weight=None, sampler=None)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    out = a.output
    out.mkdir(parents=True, exist_ok=False)  # never overwrite another experiment
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert not groups_are_intact()
    pins = json.loads((ROOT/'exp/data_hashes.json').read_text())
    actual = {p:sha(ROOT/'exp'/p) for p in pins}
    assert actual == pins, 'raw data hash mismatch'
    diff = subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT)
    env = dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               diff_sha256=hashlib.sha256(diff).hexdigest(), config=CONFIG,
               config_sha256=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest(),
               raw_hashes=actual, splits_sha256=sha(ROOT/'exp/splits.json'),
               platform=platform.platform(), processor=platform.processor(),
               cpu_count=os.cpu_count(), cpu_info=Path('/proc/cpuinfo').read_text(),
               torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__,
               num_threads=torch.get_num_threads(), num_interop_threads=torch.get_num_interop_threads(),
               start_time_saigon=time.strftime('%Y-%m-%d %H:%M:%S %Z'))
    save(out/'manifest.json',env)
    (out/'source.diff').write_bytes(diff)
    (out/'pip-freeze.txt').write_bytes(subprocess.check_output([os.sys.executable,'-m','pip','freeze']))
    rows=[]
    started=time.monotonic()
    for fold in CONFIG['folds']:
        data=load_skab_fold(fold)
        parts=data['parts']
        train,val,test=[parts[s] for s in ('train','val','test')]
        # Verify exact scaler/window coordinates against old run_detect.prepare,
        # which load_skab_fold directly calls. Record full arrays and scaler hashes.
        hashes={s:{k:array_sha(parts[s][k]) for k in ('x','y')} for s in parts}
        hashes.update(scaler_mean=array_sha(data['scaler_mean']),scaler_scale=array_sha(data['scaler_scale']))
        edges=fit_revin_quantile_edges(train['x'])
        np.savez_compressed(out/f'fold{fold}_preprocessing.npz', edges=edges.numpy(),
                            scaler_mean=data['scaler_mean'],scaler_scale=data['scaler_scale'])
        save(out/f'fold{fold}_hashes.json',hashes)
        # No singleton merging differs from old batching for these data.
        assert len(train['y']) % CONFIG['batch_size'] != 1
        for seed in CONFIG['seeds']:
            t0=time.monotonic()
            row=dict(fold=fold, seed=seed, model=CONFIG['model'], status='running')
            print(json.dumps(row),flush=True)
            try:
                fit=train_one(lambda:build_model(CONFIG['model'],task='classification',
                    n_features=data['n_features'],lookback=60,edges=edges),
                    task='classification',x_train=train['x'],y_train=train['y'],
                    x_val=val['x'],y_val=val['y'],
                    score_from_pred=lambda p:average_precision_score(val['y'],p),maximize=True,
                    seed=seed,max_epochs=30,patience=None,batch_size=256,lr=1e-3,
                    weight_decay=1e-4,grad_clip=1.)
                model=fit.pop('model')
                assert fit['epochs_ran']==30
                pv=predict(model,torch.as_tensor(val['x']),task='classification').numpy()
                pt=predict(model,torch.as_tensor(test['x']),task='classification').numpy()
                p,r,th=precision_recall_curve(val['y'],pv)
                f=2*p[:-1]*r[:-1]/(p[:-1]+r[:-1]+1e-12)
                threshold=float(th[f.argmax()]); predicted=pt>=threshold
                row.update(fit)
                row.update(status='complete', AP=float(average_precision_score(test['y'],pt)),
                    AUC=float(roc_auc_score(test['y'],pt)), F1=float(f1_score(test['y'],predicted)),
                    Precision=float(precision_score(test['y'],predicted,zero_division=0)),
                    Recall=float(recall_score(test['y'],predicted)),threshold=threshold,
                    params=sum(p.numel() for p in model.parameters()),
                    n_train=len(train['y']),n_val=len(val['y']),n_test=len(test['y']))
                torch.save(model.state_dict(),out/f'f{fold}_s{seed}.pt')
                np.savez_compressed(out/f'f{fold}_s{seed}_predictions.npz',y=test['y'],p=pt,val_y=val['y'],val_p=pv)
            except Exception:
                row.update(status='failed',error=traceback.format_exc())
            row.update(seconds=time.monotonic()-t0, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            save(out/f'f{fold}_s{seed}.json',row)
            rows.append(row)
            pd.DataFrame([{k:v for k,v in r.items() if k!='history'} for r in rows]).to_csv(out/'runs.csv',index=False)
            print(json.dumps(row),flush=True)
    df=pd.DataFrame(rows)
    assert len(df)==25 and (df.status=='complete').all(), 'incomplete jobs; see runs.csv'
    fold_means=df.groupby('fold')[['AP','AUC','F1','Precision','Recall']].mean()
    fold_means.to_csv(out/'fold_means.csv')
    summary=dict(mean=fold_means.mean().to_dict(),std_ddof1=fold_means.std(ddof=1).to_dict(),
                 std_ddof0=fold_means.std(ddof=0).to_dict(),wall_seconds=time.monotonic()-started,
                 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 note='Seeds averaged within fold, then mean/std across five folds. Historical comparison conflates RevIN and quantile bins; not causal or hardware-matched timing.')
    save(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    main()
