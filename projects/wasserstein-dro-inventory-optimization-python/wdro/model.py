
from __future__ import annotations
from dataclasses import dataclass
import itertools, numpy as np

@dataclass(frozen=True)
class NewsvendorParams:
    unit_cost: float=3.0
    holding_cost: float=1.0
    shortage_cost: float=6.0
    shortage_quadratic: float=0.12
    max_order: int=120
    support_upper: float=120.0

def loss(q,d,p:NewsvendorParams):
    q=float(q); d=np.asarray(d,float)
    shortage=np.maximum(d-q,0)
    return (
        p.unit_cost*q
        + p.holding_cost*np.maximum(q-d,0)
        + p.shortage_cost*shortage
        + p.shortage_quadratic*shortage**2
    )

def saa_order(samples,p=NewsvendorParams()):
    x=np.asarray(samples,float)
    vals=[float(np.mean(loss(q,x,p))) for q in range(p.max_order+1)]
    q=int(np.argmin(vals))
    return q,float(vals[q])

def box_robust_order(samples,p=NewsvendorParams()):
    lo=max(float(np.min(samples)),0.0); hi=min(float(np.max(samples)),p.support_upper)
    vals=[]
    for q in range(p.max_order+1):
        vals.append(max(float(loss(q,lo,p)),float(loss(q,hi,p))))
    q=int(np.argmin(vals))
    return q,float(vals[q])

def _sup_candidates(q,di,p):
    return np.unique(np.asarray([0.0,p.support_upper,float(q),float(di)]))

def worst_case_w1_cost_for_q(q,samples,epsilon,p=NewsvendorParams()):
    """
    Exact bounded-support 1D W1 dual for fixed integer q.

    inf_{lambda>=0} lambda*epsilon +
      mean_i sup_{z in [0,U]} [loss(q,z)-lambda|z-d_i|].

    For fixed lambda the declared loss minus the transport penalty is convex
    on each region split by {0, q, d_i, U}; therefore its maximum is attained
    at a region endpoint. The outer objective is convex piecewise linear in
    lambda, so all pairwise affine breakpoints plus boundary values suffice.
    """
    x=np.asarray(samples,float)
    # Global demand-direction Lipschitz bound of the bounded-support loss.
    L=max(
        p.holding_cost,
        p.shortage_cost + 2*p.shortage_quadratic*p.support_upper,
    )
    lambdas={0.0,float(L)}
    per=[]
    for di in x:
        zs=_sup_candidates(q,di,p)
        intercept=np.asarray([float(loss(q,z,p)) for z in zs])
        dist=np.abs(zs-di)
        per.append((intercept,dist))
        for a,b in itertools.combinations(range(len(zs)),2):
            den=dist[a]-dist[b]
            if abs(den)<1e-12: continue
            lam=(intercept[a]-intercept[b])/den
            if -1e-12<=lam<=L+1e-12:
                lambdas.add(float(np.clip(lam,0,L)))
    best=float("inf"); bestlam=None
    for lam in lambdas:
        val=lam*epsilon + np.mean([
            np.max(inter-lam*dist) for inter,dist in per
        ])
        if val<best:
            best=float(val); bestlam=float(lam)
    return best,bestlam

def wasserstein_dro_order(samples,epsilon,p=NewsvendorParams()):
    vals=[]; lams=[]
    for q in range(p.max_order+1):
        v,lam=worst_case_w1_cost_for_q(q,samples,epsilon,p)
        vals.append(v); lams.append(lam)
    q=int(np.argmin(vals))
    return q,float(vals[q]),float(lams[q])
