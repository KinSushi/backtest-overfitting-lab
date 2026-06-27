import unittest, math, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from egp_meta_algos import *

class TestMetaAlgorithms(unittest.TestCase):
    def test_lshade_improves_sphere(self):
        state=lshade_minimize(sphere,[(-5,5)]*5,max_evals=800,seed=11)
        self.assertLess(state.best_score, 1.0)
        self.assertGreater(len(state.history), 0)
    def test_cmaes_improves_sphere(self):
        state=cmaes_minimize(sphere,[(-3,3)]*3,max_evals=360,seed=12)
        self.assertLess(state.best_score, 1.0)
    def test_nsga2_dominance_sort(self):
        objs=[(1,5),(2,4),(5,1),(3,3),(4,4)]
        fronts=fast_non_dominated_sort(objs)
        self.assertIn(0, fronts[0]); self.assertIn(2, fronts[0]); self.assertNotIn(4, fronts[0])
        order=nsga2_sort(objs)
        self.assertEqual(set(order), set(range(len(objs))))
    def test_tpe_moves_towards_good_region(self):
        sp={'x':{'type':'float','min':-5,'max':5},'flag':{'type':'cat','choices':['a','b']}}
        t=SimpleTPE(sp,seed=4)
        for i in range(40):
            x=t.random_candidate(); y=(x['x']-1.0)**2+(0 if x['flag']=='b' else 3); t.observe(x,y)
        s=t.suggest(128)
        self.assertIn(s['flag'], ['a','b'])
        self.assertTrue(-5<=s['x']<=5)
    def test_tabu_vns_onemax_min_cost(self):
        start=[0,0,0,0,0]
        best,score=tabu_vns_optimize(start, lambda x: sum(1-v for v in x), max_iter=50)
        self.assertEqual(score, 0)
        self.assertEqual(best, [1,1,1,1,1])
    def test_successive_halving_keeps_best(self):
        configs=list(range(9))
        survivors,hist=successive_halving(configs, lambda c,b: abs(c-2)/(b), min_budget=1, max_budget=9, eta=3)
        self.assertIn(2, survivors)
        self.assertTrue(hist)

if __name__=='__main__': unittest.main()
