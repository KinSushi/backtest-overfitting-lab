"""Tests for the optimization->validation assembler (PBO/RC/SPA from N configs)."""
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_opt_validation as OV


def test_align_to_grid_fills_zero():
    trades = {
        "cfgA": [{"net_pnl": 10.0, "exit_time": "2024.01.02 10:00:00"},
                 {"net_pnl": 5.0,  "exit_time": "2024.01.03 10:00:00"}],
        "cfgB": [{"net_pnl": -3.0, "exit_time": "2024.01.03 11:00:00"}],
    }
    grid, aligned = OV.align_to_grid(trades)
    assert grid == ["2024.01.02", "2024.01.03"]
    assert aligned["cfgA"] == [10.0, 5.0]
    assert aligned["cfgB"] == [0.0, -3.0]       # 02 absent de B -> 0


def test_build_matrices_shapes_and_lossdiff():
    cr = {"a": [1.0, -1.0, 2.0], "b": [0.5, 0.5, -0.5]}
    perf, loss, ids = OV.build_matrices(cr)
    assert len(perf) == 3 and len(perf[0]) == 2          # T x C
    assert loss == perf                                   # benchmark=0 -> loss == perf
    perf2, loss2, _ = OV.build_matrices(cr, benchmark=[0.1, 0.1, 0.1])
    assert abs(loss2[0][0] - 0.9) < 1e-9                  # 1.0 - 0.1


def test_unequal_length_raises():
    try:
        OV.build_matrices({"a": [1, 2], "b": [1, 2, 3]})
        assert False
    except ValueError:
        pass


def test_validate_optimization_runs_full_battery():
    # 40 pure-noise configs -> PBO must be HIGH (selection = overfitting)
    random.seed(0)
    T = 120
    cr = {f"c{i}": [random.gauss(0, 1) for _ in range(T)] for i in range(40)}
    r = OV.validate_optimization(cr)
    g = r["gate"]
    assert r["n_configs"] == 40
    assert g["pbo"] is not None and g["rc_pvalue"] is not None and g["spa_pvalue"] is not None
    assert 0.0 <= g["pbo"] <= 1.0
    # pure noise: the battery must REJECT (no real edge)
    assert g["decision"] == "REJECT"


def test_pbo_discriminates_edge_vs_noise():
    # PBO must be LOWER when a real edge is present than for pure noise
    random.seed(2)
    T = 120
    noise = {f"n{i}": [random.gauss(0, 1) for _ in range(T)] for i in range(20)}
    edge = {f"n{i}": [random.gauss(0, 1) for _ in range(T)] for i in range(19)}
    edge["winner"] = [random.gauss(0.5, 1) for _ in range(T)]      # real and persistent edge
    pbo_noise = OV.validate_optimization(noise)["gate"]["pbo"]
    pbo_edge = OV.validate_optimization(edge)["gate"]["pbo"]
    assert pbo_edge < pbo_noise          # a real edge -> less selection overfitting
    assert pbo_edge < 0.5                # the real winner holds out-of-sample
    # and the config selected in the "edge" set is indeed the winner
    assert OV.validate_optimization(edge)["selected_config"] == "winner"
