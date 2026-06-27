#!/usr/bin/env python3
"""
demo/run_demo.py - END-TO-END demonstration of the EGP pipeline on SYNTHETIC data.

Goal: prove, reproducibly and without MetaTrader, that EVERY path of the pipeline
runs and produces a coherent verdict:

  (A) MODEL path from PRICES               : run_pipeline (triple-barrier -> features ->
      forest -> purged CV -> calibration -> gate).
  (B) REAL DEALS path -> risk              : full_report section "deals_gate"
      (DSR + Monte-Carlo risk_of_ruin + costs), with the DEPOSIT line filtered.
  (C) OPTIMIZATION path -> overfitting     : full_report section "optimization"
      (PBO + White's Reality Check + Hansen SPA, n_trials = number of configs).
  (D) FEATURES path -> MODEL               : full_report section "model"
      (does the ML filter add value vs raw?).

The data is generated with fixed seeds -> reproducible result.
The markdown report is written to demo/REPORT.md.

Run:  python3 demo/run_demo.py
"""
from __future__ import annotations

import os
import random
import sys

# --- resolve tools/ relative to THIS file (works from anywhere) ---
HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "tools")
sys.path.insert(0, TOOLS)

import egp_runner as R                  # noqa: E402
import egp_full_report as FR            # noqa: E402

DEPOSIT = 10000.0


# --------------------------------------------------------------------------- #
# 1) Ground truth: a list of trades with WEAKLY predictive features           #
# --------------------------------------------------------------------------- #
def gen_ground_truth(n=260, seed=7):
    """Each trade: position, side, exit instant, 4 features, win/loss outcome, P&L.
    f0 is weakly tied to the outcome (real signal + noise); f1..f3 are noise."""
    rng = random.Random(seed)
    trades = []
    for i in range(n):
        pos = 1000 + i
        side = rng.choice([0, 1])                      # 0=buy, 1=sell
        # spread the exits over ~2024 (days 1..27 to stay valid all months)
        month = (i // 22) % 12 + 1
        day = (i % 22) + 1
        t = f"2024.{month:02d}.{day:02d} {9 + (i % 8):02d}:30:00"
        f0 = rng.gauss(0.0, 1.0)                        # weak signal
        f1, f2, f3 = rng.gauss(0, 1), rng.gauss(0, 1), rng.gauss(0, 1)   # noise
        win = (0.8 * f0 + rng.gauss(0.0, 1.3)) > 0     # weak but real relation
        pnl = (abs(rng.gauss(85, 35)) if win else -abs(rng.gauss(72, 33)))
        trades.append({"pos": pos, "side": side, "time": t,
                       "feats": [round(f0, 5), round(f1, 5), round(f2, 5), round(f3, 5)],
                       "pnl": round(pnl, 2)})
    return trades


def write_deals_csv(trades, path, commission_per_deal=-0.5):
    """Writes a CSV in MT5 export format: DEPOSIT line (Type=2) + 2 lines/trade (in/out)."""
    cols = ["Time", "Deal", "Symbol", "Type", "Direction", "Volume", "Price",
            "Commission", "Swap", "Profit", "Balance", "Position", "Fee"]
    rows = [cols]
    deal_id = 1
    bal = DEPOSIT
    # initial deposit : Type=2 (DEAL_TYPE_BALANCE), Position=0 -> must be FILTERED by the pipeline
    rows.append(["2024.01.01 00:00:00", deal_id, "", 2, "in", 0, 0,
                 0, 0, DEPOSIT, bal, 0, 0])
    deal_id += 1
    for tr in trades:
        typ = 0 if tr["side"] == 0 else 1
        # entry (in) : profit 0, commission charged
        rows.append([tr["time"], deal_id, "XAUUSD", typ, "in", 0.10, 2000.0,
                     commission_per_deal, 0, 0.0, bal, tr["pos"], 0])
        deal_id += 1
        # exit (out) : profit = trade P&L
        bal += tr["pnl"] + 2 * commission_per_deal
        rows.append([tr["time"], deal_id, "XAUUSD", typ, "out", 0.10, 2005.0,
                     commission_per_deal, 0, tr["pnl"], round(bal, 2), tr["pos"], 0])
        deal_id += 1
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    return path


def write_features_csv(trades, path):
    """Writes a CSV in EGP_MHO_FeaturesExport format : Position,Time,Side,f0..f3 (1 line/position)."""
    k = len(trades[0]["feats"])
    header = "Position,Time,Side," + ",".join(f"f{j}" for j in range(k))
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for tr in trades:
            f.write(f"{tr['pos']},{tr['time']},{tr['side']}," +
                    ",".join(str(v) for v in tr["feats"]) + "\n")
    return path


# --------------------------------------------------------------------------- #
# 2) Synthetic prices + signals for the MODEL-from-prices path                #
# --------------------------------------------------------------------------- #
def gen_prices_signals(n_prices=1500, every=11, seed=3):
    rng = random.Random(seed)
    p = 2000.0
    prices = [p]
    for _ in range(n_prices - 1):
        p *= (1.0 + rng.gauss(0.0002, 0.010))          # random walk with slight drift
        prices.append(round(p, 3))
    idx, sides = [], []
    for i in range(120, n_prices - 20, every):         # periodic signals, margin at the edges
        idx.append(i)
        sides.append(1 if prices[i] > prices[i - 20] else -1)   # simple momentum
    return prices, idx, sides


# --------------------------------------------------------------------------- #
# 3) Synthetic optimization configs (for PBO / RC / SPA)                      #
# --------------------------------------------------------------------------- #
def gen_opt_configs(n_cfg=30, n_trades=90, seed=21, edge_cfgs=2):
    """{config: [trades net_pnl+exit_time]}. Most = noise; a few have a slight edge."""
    rng = random.Random(seed)
    cfgs = {}
    for c in range(n_cfg):
        mu = 6.0 if c < edge_cfgs else 0.0             # 2 configs with a real (small) edge
        trades = []
        for i in range(n_trades):
            month = (i // 8) % 12 + 1
            day = (i % 8) + 1
            t = f"2024.{month:02d}.{day:02d} 12:00:00"
            trades.append({"net_pnl": round(rng.gauss(mu, 70), 2), "exit_time": t})
        cfgs[f"cfg_{c:02d}"] = trades
    return cfgs


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    print("=== EGP DEMO - full pipeline on synthetic data ===\n")
    truth = gen_ground_truth()
    deals_csv = write_deals_csv(truth, os.path.join(HERE, "synth_deals.csv"))
    feats_csv = write_features_csv(truth, os.path.join(HERE, "synth_features.csv"))
    print(f"[data] {len(truth)} trades -> {os.path.basename(deals_csv)} + {os.path.basename(feats_csv)}")

    # (A) MODEL path from PRICES
    prices, idx, sides = gen_prices_signals()
    cfg = R.default_config()
    cfg["montecarlo"]["paths"] = 300
    try:
        model_from_prices = R.run_pipeline(prices, idx, sides, cfg)
        mfp_gate = model_from_prices.get("manifest", {}).get("gate_decision", "?")
        print(f"[A] run_pipeline (prices->model): {len(idx)} signals -> gate={mfp_gate}")
    except Exception as e:
        model_from_prices = {"error": str(e)}
        print(f"[A] run_pipeline failed: {e}")

    # (B+C+D) unified report: deals + optimization + features
    opt = gen_opt_configs()
    report = FR.full_report(deals_csv, is_path=True, initial_deposit=DEPOSIT,
                            features_csv=feats_csv, opt_config_trades=opt)

    s = report["summary"]
    print(f"[B] deals -> net={s.get('net_total')} | DSR={s.get('dsr')} | "
          f"risk_of_ruin={s.get('risk_of_ruin')} | {s.get('deals_decision')}")
    print(f"[C] optim -> {s.get('opt_n_configs')} configs | PBO={s.get('opt_pbo')} | "
          f"RC={s.get('opt_rc_pvalue')} | SPA={s.get('opt_spa_pvalue')} | {s.get('opt_decision')}")
    print(f"[D] model -> adds value: {s.get('model_adds_value')}")

    # markdown report
    md = FR.render_markdown(report)
    # prepend path (A) verdict
    md = (f"<!-- reproducible demo, fixed seeds -->\n\n"
          f"_Path (A) prices->model: gate = "
          f"{model_from_prices.get('manifest', {}).get('gate_decision', model_from_prices.get('error', '?'))}_\n\n") + md
    out = os.path.join(HERE, "REPORT.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n[ok] report written -> demo/REPORT.md ({len(md)} chars)")
    return report


if __name__ == "__main__":
    main()
