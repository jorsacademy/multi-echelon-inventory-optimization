
from __future__ import annotations
import argparse
from wdro import *

def self_test():
    p=NewsvendorParams(max_order=30,support_upper=30)
    x=[8,10,12,14,16]
    q,v,lam=wasserstein_dro_order(x,2.0,p)
    assert 0<=q<=30 and v>0 and 0<=lam<=max(p.holding_cost,p.shortage_cost+2*p.shortage_quadratic*p.support_upper)
    print("Wasserstein DRO inventory self-test: OK")

def main(a):
    p=NewsvendorParams(max_order=a.max_order,support_upper=a.support_upper)
    train=sample_demand(a.train_samples,seed=a.seed)
    nominal=sample_demand(a.test_samples,seed=a.seed+1_000_000)
    shifted=sample_demand(a.test_samples,seed=a.seed+2_000_000,shift=a.shift)
    eps=calibrate_radius(train,seed=a.seed+3_000_000,repetitions=a.radius_reps,quantile=a.radius_quantile)
    print(f"calibrated W1 radius={eps:.4f}")
    qdro,vdro,lam=wasserstein_dro_order(train,eps,p)
    print(f"DRO order={qdro} worst-case sample objective={vdro:.3f} dual lambda={lam:.3f}")
    rows=evaluate(train,nominal,shifted,eps,p)
    print(f"{'method':<18}{'order':>8}{'nominal cost':>16}{'shifted cost':>15}{'shift stockout':>16}")
    for r in rows:
        print(f"{r.method:<18}{r.order:8d}{r.nominal_cost:16.2f}{r.shifted_cost:15.2f}{r.shifted_stockout:16.4f}")

def parse():
    p=argparse.ArgumentParser(); p.add_argument("--self-test",action="store_true")
    p.add_argument("--seed",type=int,default=42); p.add_argument("--train-samples",type=int,default=120)
    p.add_argument("--test-samples",type=int,default=3000); p.add_argument("--shift",type=float,default=.8)
    p.add_argument("--max-order",type=int,default=120); p.add_argument("--support-upper",type=float,default=120)
    p.add_argument("--radius-reps",type=int,default=100); p.add_argument("--radius-quantile",type=float,default=.9)
    return p.parse_args()
if __name__=="__main__":
    a=parse(); self_test() if a.self_test else main(a)
