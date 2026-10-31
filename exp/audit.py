import json, hashlib
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
FEATURES=['Accelerometer1RMS','Accelerometer2RMS','Current','Pressure','Temperature','Thermocouple','Voltage','Volume Flow RateRMS']

def load():
    frames={p.parent.name+'/'+p.name:pd.read_csv(p,sep=';') for p in sorted((ROOT/'data').glob('*/*.csv')) if p.parent.name!='anomaly-free'}
    assert len(frames)==34
    for n,f in frames.items():
        assert np.isfinite(f[FEATURES].to_numpy()).all()
        assert set(f.anomaly.unique())=={0,1}
        d=pd.to_datetime(f.datetime)
        assert d.is_monotonic_increasing and not d.duplicated().any()
    return frames

def keys(f,with_time=True):
    return set(f[['datetime']+FEATURES if with_time else FEATURES].itertuples(index=False,name=None))

def plan(frames):
    names=sorted(frames);parent={n:n for n in names}
    def find(n):
        while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
        return n
    k={n:keys(f) for n,f in frames.items()};v={n:keys(f,False) for n,f in frames.items()}
    intervals={n:(pd.to_datetime(f.datetime).min(),pd.to_datetime(f.datetime).max()) for n,f in frames.items()}
    pairs=[]
    for a,b in combinations(names,2):
        overlap=intervals[a][0]<=intervals[b][1] and intervals[b][0]<=intervals[a][1]
        shared=k[a]&k[b];sharedv=v[a]&v[b]
        if overlap or shared or sharedv:
            parent[find(b)]=find(a)
            pairs.append(dict(a=a,b=b,time_overlap=bool(overlap),shared_time_sensor=len(shared),shared_sensor_only=len(sharedv)))
    groups={}
    for n in names:groups.setdefault(find(n),[]).append(n)
    rng=np.random.default_rng(20261028);ids=list(groups);rng.shuffle(ids);ids.sort(key=lambda g:-len(groups[g]))
    buckets=[[] for _ in range(5)];counts=[{} for _ in range(5)]
    for g in ids:
        cats={n.split('/')[0] for n in groups[g]};assert len(cats)==1
        cat=next(iter(cats));j=min(range(5),key=lambda j:(counts[j].get(cat,0),len(buckets[j]),j))
        buckets[j].extend(groups[g]);counts[j][cat]=counts[j].get(cat,0)+len(groups[g])
    folds=[dict(fold=i,train=sorted(n for j,b in enumerate(buckets) if j not in (i,(i+1)%5) for n in b),val=sorted(buckets[(i+1)%5]),test=sorted(buckets[i])) for i in range(5)]
    checks=[]
    for fold in folds:
        combined={s:set().union(*(k[n] for n in fold[s])) for s in ['train','val','test']}
        values={s:set().union(*(v[n] for n in fold[s])) for s in ['train','val','test']}
        for a,b in combinations(['train','val','test'],2):
            timecross=sum(intervals[x][0]<=intervals[y][1] and intervals[y][0]<=intervals[x][1] for x in fold[a] for y in fold[b])
            row=dict(fold=fold['fold'],a=a,b=b,shared_time_sensor=len(combined[a]&combined[b]),shared_sensor_only=len(values[a]&values[b]),overlapping_file_pairs=int(timecross))
            assert not any(row[z] for z in ['shared_time_sensor','shared_sensor_only','overlapping_file_pairs']),row
            checks.append(row)
    result=dict(split_seed=20261028,group_rule='Connected components: overlapping absolute intervals OR identical timestamp+8 sensors OR identical 8 sensors; labels unchanged',groups=list(groups.values()),pairs=pairs,buckets=buckets,folds=folds,checks=checks)
    (ROOT/'splits.json').write_text(json.dumps(result,indent=2))
    provenance={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'data').rglob('*.csv'))}
    (ROOT/'data_hashes.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(dict(files=len(names),groups=len(groups),pairs=pairs,fold_sizes=[{s:len(f[s]) for s in ['train','val','test']} for f in folds],cross_split_checks=len(checks)),indent=2))
    return result
if __name__=='__main__':plan(load())
