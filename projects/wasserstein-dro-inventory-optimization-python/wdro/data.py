
from __future__ import annotations
import numpy as np

def sample_demand(n,*,seed=42,shift=0.0):
    rng=np.random.default_rng(seed)
    base=rng.gamma(shape=9.0,scale=5.0,size=n)
    seasonal=rng.normal(0,5,size=n)
    surge=(rng.random(n)<(.06+.04*max(shift,0)))*rng.uniform(15,35,size=n)
    d=base+seasonal+surge+shift*10
    return np.clip(d,0,120)

def empirical_w1_equal(a,b):
    a=np.sort(np.asarray(a,float)); b=np.sort(np.asarray(b,float))
    if len(a)!=len(b): raise ValueError("equal sample sizes required")
    return float(np.mean(np.abs(a-b)))

def calibrate_radius(samples,*,seed=42,repetitions=120,quantile=.90):
    x=np.asarray(samples,float)
    if len(x)<20: raise ValueError("need >=20 observations")
    rng=np.random.default_rng(seed)
    k=len(x)//2
    vals=[]
    for _ in range(repetitions):
        idx=rng.choice(len(x),size=2*k,replace=False)
        vals.append(empirical_w1_equal(x[idx[:k]],x[idx[k:]]))
    return float(np.quantile(vals,quantile,method="higher"))
