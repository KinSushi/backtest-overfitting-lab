"""Offline tests for egp_bohb (deterministic, synthetic multi-fidelity objective)."""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_bohb as B


def _sphere(cfg):
    return sum(v * v for v in cfg.values())


def _objective(cfg, budget):
    # Multi-fidelity: cost = sphere(x) + bias that vanishes at full budget.
    # The argmin stays x=0 regardless of budget; a larger budget reduces the cost.
    return _sphere(cfg) + 1.0 / budget


SPACE = {f"x{i}": {"type": "float", "min": -5.0, "max": 5.0} for i in range(3)}


def test_bohb_bracket_structure_matches_hyperband():
    # s_max = floor(log_eta(R)) ; nb de brackets = s_max + 1 (reutilise hyperband).
    R, eta = 27, 3
    _, _, brackets = B.bohb(_objective, SPACE, max_resource=R, eta=eta, seed=0)
    s_max = int(math.floor(math.log(R, eta)))
    assert len(brackets) == s_max + 1
    assert all("rounds" in bk for bk in brackets)


def test_bohb_converges_near_optimum():
    # On a sphere, BOHB must find a low-value config (close to the optimum 0).
    best_c, best_s, _ = B.bohb(_objective, SPACE, max_resource=27, eta=3, seed=1)
    assert _sphere(best_c) < 5.0, _sphere(best_c)
    # best_s includes the fidelity term 1/R ; it stays small
    assert best_s < 6.0


def test_bohb_uses_the_model():
    # The TPE model must have been used (otherwise BOHB reduces to random Hyperband).
    _, _, brackets = B.bohb(_objective, SPACE, max_resource=27, eta=3,
                            rand_frac=1.0 / 3.0, min_points=8, seed=2)
    assert brackets[-1]["n_model"] >= 1
    assert brackets[-1]["n_random"] >= 1     # exploration conservee


def test_space_from_dims_roundtrip():
    # Conversion OptDim -> space : types corrects.
    class D:
        def __init__(self, name, kind, lo=0.0, hi=1.0, choices=()):
            self.name = name; self.kind = kind; self.lo = lo; self.hi = hi
            self.choices = choices
    dims = [D("a", "cont", -1.0, 2.0), D("b", "int", 0, 10),
            D("c", "bool"), D("d", "enum", choices=("x", "y", "z"))]
    sp = B.space_from_dims(dims)
    assert sp["a"]["type"] == "float" and sp["a"]["min"] == -1.0
    assert sp["b"]["type"] == "int" and sp["b"]["max"] == 10
    assert sp["c"]["type"] == "cat" and sp["c"]["choices"] == [0, 1]
    assert sp["d"]["type"] == "cat" and sp["d"]["choices"] == [0, 1, 2]
