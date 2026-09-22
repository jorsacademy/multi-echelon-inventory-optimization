
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .model import NewsvendorParams,loss,saa_order,box_robust_order,wasserstein_dro_order

@dataclass(frozen=True)
class EvalRow:
    method:str; order:int; nominal_cost:float; shifted_cost:float; shifted_stockout:float

def evaluate(train,nominal_test,shifted_test,epsilon,p=NewsvendorParams()):
    methods={
        "SAA":saa_order(train,p)[0],
        "Box robust":box_robust_order(train,p)[0],
        "Wasserstein DRO":wasserstein_dro_order(train,epsilon,p)[0],
    }
    out=[]
    for name,q in methods.items():
        nc=float(np.mean(loss(q,nominal_test,p)))
        sc=float(np.mean(loss(q,shifted_test,p)))
        stock=float(np.mean(np.asarray(shifted_test)>q))
        out.append(EvalRow(name,q,nc,sc,stock))
    return tuple(out)
