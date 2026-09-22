
import unittest, numpy as np
from wdro import *

class Tests(unittest.TestCase):
    def test_empirical_w1_hand(self):
        self.assertAlmostEqual(empirical_w1_equal([0,2],[1,5]),2.0)

    def test_radius_positive(self):
        x=sample_demand(60,seed=1)
        self.assertGreater(calibrate_radius(x,seed=2,repetitions=20),0)

    def test_w1_zero_radius_matches_saa_cost_at_fixed_q(self):
        p=NewsvendorParams(max_order=30,support_upper=30)
        x=np.array([5.,10.,15.,20.])
        for q in [0,7,15,30]:
            wc,_=worst_case_w1_cost_for_q(q,x,0.0,p)
            self.assertAlmostEqual(wc,float(np.mean(loss(q,x,p))),places=8)

    def test_inner_dual_matches_dense_lambda_grid(self):
        p=NewsvendorParams(max_order=25,support_upper=25)
        x=np.array([4.,9.,16.,20.]); q=11; eps=2.5
        exact,lam=worst_case_w1_cost_for_q(q,x,eps,p)
        grid=np.linspace(0,max(p.holding_cost,p.shortage_cost+2*p.shortage_quadratic*p.support_upper),20001)
        vals=[]
        for la in grid:
            terms=[]
            for di in x:
                z=np.linspace(0,p.support_upper,5001)
                terms.append(np.max(loss(q,z,p)-la*np.abs(z-di)))
            vals.append(la*eps+np.mean(terms))
        self.assertAlmostEqual(exact,min(vals),delta=2e-3)

    def test_dro_outer_order_is_global_integer_minimum(self):
        p=NewsvendorParams(max_order=35,support_upper=35)
        x=np.array([8.,12.,14.,20.,22.]); eps=3.0
        q,v,_=wasserstein_dro_order(x,eps,p)
        vals=[worst_case_w1_cost_for_q(k,x,eps,p)[0] for k in range(36)]
        self.assertEqual(q,int(np.argmin(vals))); self.assertAlmostEqual(v,min(vals),places=9)

    def test_orders_valid(self):
        p=NewsvendorParams()
        x=sample_demand(50,seed=5)
        eps=calibrate_radius(x,seed=6,repetitions=20)
        for q in [saa_order(x,p)[0],box_robust_order(x,p)[0],wasserstein_dro_order(x,eps,p)[0]]:
            self.assertTrue(0<=q<=p.max_order)

if __name__=="__main__":unittest.main()
