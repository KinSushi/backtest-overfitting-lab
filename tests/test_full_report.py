"""Tests for the single full_report command (deals + optimization + render)."""
import os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import egp_full_report as FR


def _deals_csv(n=18, seed=5):
    """Petit CSV au format MT5 : depot (Type=2) + 2 lignes/trade (in/out)."""
    rng = random.Random(seed)
    cols = ["Time", "Deal", "Symbol", "Type", "Direction", "Volume", "Price",
            "Commission", "Swap", "Profit", "Balance", "Position", "Fee"]
    lines = [",".join(cols)]
    did = 1
    lines.append(f"2024.01.01 00:00:00,{did},,2,in,0,0,0,0,10000,10000,0,0")   # depot
    did += 1
    for i in range(n):
        pos = 500 + i
        t = f"2024.{(i // 8) % 12 + 1:02d}.{(i % 8) + 1:02d} 10:00:00"
        pnl = round(rng.gauss(15, 80), 2)
        lines.append(f"{t},{did},XAUUSD,0,in,0.1,2000,-0.5,0,0,10000,{pos},0"); did += 1
        lines.append(f"{t},{did},XAUUSD,0,out,0.1,2005,-0.5,0,{pnl},10000,{pos},0"); did += 1
    return "\n".join(lines) + "\n"


def test_deals_section_present_and_deposit_filtered():
    r = FR.full_report(_deals_csv(n=18), is_path=False)
    assert "deals_gate" in r["sections"]
    s = r["summary"]
    assert s["n_trades"] == 18                 # the deposit (Type=2) is NOT counted
    assert s["deals_decision"] in ("ACCEPT", "REJECT")
    assert s["risk_of_ruin"] is not None


def test_optimization_section_runs_pbo():
    random.seed(2)
    T = 100
    opt_returns = {f"c{i}": [random.gauss(0, 1) for _ in range(T)] for i in range(25)}
    r = FR.full_report(_deals_csv(), is_path=False, opt_config_returns=opt_returns)
    assert "optimization" in r["sections"]
    s = r["summary"]
    assert s["opt_n_configs"] == 25
    assert s["opt_pbo"] is not None
    assert s["opt_rc_pvalue"] is not None and s["opt_spa_pvalue"] is not None
    assert s["opt_decision"] in ("ACCEPT", "REJECT")


def test_render_markdown_contains_sections():
    random.seed(3)
    opt = {f"c{i}": [random.gauss(0, 1) for _ in range(80)] for i in range(12)}
    r = FR.full_report(_deals_csv(), is_path=False, opt_config_returns=opt)
    md = FR.render_markdown(r)
    assert "Real deals" in md
    assert "Optimization" in md
    assert "Summary" in md
    assert isinstance(md, str) and len(md) > 200


def test_summary_keys_stable():
    r = FR.full_report(_deals_csv(), is_path=False)
    for k in ("n_trades", "net_total", "dsr", "risk_of_ruin", "deals_decision"):
        assert k in r["summary"]
