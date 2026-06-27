#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_full_battery.py - run EVERY pure-Python-runnable AFML check on a REAL MT5 deals
export, in ONE consolidated report.

Tiers and the inputs they need:

  ALWAYS (deals CSV only):
    - deflated gate (DSR, MinTRL) + Monte-Carlo (risk of ruin) + cost stress   [egp_full_report]
    - REGIME / SESSION fragility - *where* the edge lives                       [egp_regime]
  IF --configs FOLDER (one deals CSV per config, e.g. Common\\Files\\EGP_optpass):
    - PBO/CSCV + White's Reality Check + Hansen SPA over ALL configs            [egp_full_report]
  IF --features FILE (CLEAN per-position features joined to deals):
    - ML model validation (sized vs raw gate)                                  [egp_full_report]

NOT run here - they require MT5 to RE-EVALUATE each config/window (see README "Roadmap"):
    - walk-forward IS->OOS (egp_wfo)
    - CPCV at optimization (egp_cpcv_pipeline)
    - global sensitivity (egp_sobol / egp_bohb)

Pure-Python, standard library only. No PYTHONPATH needed (adds tools/ to sys.path).

Usage:
    python demo/run_full_battery.py                              # auto-locate the deals export
    python demo/run_full_battery.py path/to/EGP_deals.csv
    python demo/run_full_battery.py --configs "...\\EGP_optpass" # add the overfitting battery
    python demo/run_full_battery.py --features "...\\EGP_features.csv"   # add the model battery
"""
import argparse
import datetime
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def _locate_deals():
    """Best-effort: MT5 Common\\Files\\EGP_deals.csv, then the current directory."""
    cands = []
    appdata = os.environ.get("APPDATA")
    if appdata:
        cands.append(os.path.join(appdata, "MetaQuotes", "Terminal", "Common", "Files", "EGP_deals.csv"))
    cands.append(os.path.join(os.getcwd(), "EGP_deals.csv"))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def _to_dt(v):
    """MT5 timestamps are strings ('YYYY.MM.DD HH:MM:SS'); regime needs a datetime."""
    if isinstance(v, datetime.datetime):
        return v
    if isinstance(v, str):
        for f in ("%Y.%m.%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H:%M"):
            try:
                return datetime.datetime.strptime(v.strip(), f)
            except ValueError:
                pass
    return v


def _fmt(v):
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v in (float("inf"), float("-inf")):
            return "inf" if v > 0 else "-inf"
        return f"{v:.4g}"
    return str(v)


def _regime_section(deals_text, signed=True):
    """REGIME/SESSION section (not produced by egp_full_report). Adapts the trade schema:
    egp_regime expects {'time': datetime, 'pnl': float}; trades_from_deals gives
    {'exit_time': str, 'net_pnl': float}."""
    import egp_mt5_deals as D
    import egp_regime as RG
    trades = D.trades_from_deals(D.parse_deals_csv(deals_text), signed=signed)
    tr = [{"time": _to_dt(t.get("exit_time")), "pnl": float(t["net_pnl"])}
          for t in trades if t.get("exit_time") is not None]
    if not tr:
        return "## Regime / session\n(no timestamped trades to analyze)\n"
    by = RG.performance_by_session(tr)
    ec = RG.edge_concentration(by)
    out = ["## Regime / session - where does the edge live?", "",
           "| Session | Trades | Net | PF | Win% | Sharpe |",
           "|---|---:|---:|---:|---:|---:|"]
    for k, v in by.items():
        out.append(f"| {k} | {v['count']} | {_fmt(v['net'])} | {_fmt(v['profit_factor'])} "
                   f"| {_fmt(v['win_rate'])} | {_fmt(v['sharpe'])} |")
    flag = "**FRAGILE**" if ec.get("fragile") else "ok"
    out += ["", f"- edge concentration: {flag} - {ec.get('reason', '')} "
                f"(HHI={_fmt(ec.get('hhi'))}, best={ec.get('best_segment')} "
                f"share={_fmt(ec.get('best_share'))})"]
    return "\n".join(out) + "\n"


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)   # show progress immediately, even when piped (Tee)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Full AFML battery on a real MT5 deals export (one report).")
    ap.add_argument("deals", nargs="?", default=None,
                    help="Deals CSV (default: auto-locate Common\\Files\\EGP_deals.csv).")
    ap.add_argument("--configs", default=None,
                    help="Folder with one deals CSV per config -> enables PBO/RC/SPA.")
    ap.add_argument("--max-configs", type=int, default=0,
                    help="If the folder holds more configs than this, randomly subsample this many "
                         "(seeded) so PBO/CSCV/RC/SPA stay tractable. 0 = use all (pure-Python PBO "
                         "over more than a few hundred configs takes many minutes).")
    ap.add_argument("--features", default=None,
                    help="CLEAN features CSV (per position) -> enables ML model validation.")
    ap.add_argument("--deposit", type=float, default=10000.0, help="Initial deposit (default 10000).")
    ap.add_argument("--unsigned", action="store_true", help="Treat P&L as UNSIGNED (default: signed).")
    args = ap.parse_args()
    signed = not args.unsigned

    deals_path = args.deals or _locate_deals()
    if not deals_path or not os.path.isfile(deals_path):
        print("ERROR: deals CSV not found. Pass a path, or run a backtest first so the EA writes it.")
        return 2
    deals_text = open(deals_path, encoding="utf-8-sig", errors="replace").read()
    print(f"[full-battery] deals   : {deals_path}")

    from egp_full_report import full_report, render_markdown

    # optional: configs folder -> PBO/RC/SPA over all configs
    opt = None
    if args.configs:
        paths = sorted(glob.glob(os.path.join(args.configs, "*.csv")))
        n_found = len(paths)
        if args.max_configs and n_found > args.max_configs:
            import random as _random
            _random.Random(0).shuffle(paths)                 # seeded -> reproducible subsample
            paths = sorted(paths[:args.max_configs])
            print(f"[full-battery] configs : {n_found} found -> randomly subsampled to {len(paths)} "
                  f"(seeded) so PBO stays tractable")
        elif n_found > 800:
            print(f"[full-battery] configs : {n_found} found -> PBO over this many is SLOW in pure "
                  f"Python; consider --max-configs 300 (or re-optimize on fewer parameters)")
        if len(paths) >= 2:
            from egp_mt5_collect import build_opt_config_trades
            opt, _ = build_opt_config_trades(paths, signed=signed)
            print(f"[full-battery] configs : {len(opt)} -> PBO/RC/SPA enabled")
        else:
            print(f"[full-battery] configs : <2 CSVs in {args.configs} -> PBO skipped")

    features_text = None
    if args.features:
        if not os.path.isfile(args.features):
            print(f"[full-battery] features: {args.features} NOT FOUND -> model skipped")
        else:
            features_text = open(args.features, encoding="utf-8-sig", errors="replace").read()
            print(f"[full-battery] features: {args.features} -> model validation enabled "
                  f"(use a CLEAN single-backtest export)")

    # 1) deals gate + Monte-Carlo + costs (always); + PBO (if configs); + model (if features)
    rep = full_report(deals_text, is_path=False, initial_deposit=args.deposit, signed=signed,
                      features_csv=features_text, opt_config_trades=opt)
    print()
    print(render_markdown(rep))

    # 2) regime/session fragility (NOT produced by full_report)
    print()
    print(_regime_section(deals_text, signed=signed))

    # 3) explicit notice for the tiers that need MT5
    print("## Not run here (require MT5 to re-evaluate each config/window)")
    print("- walk-forward IS->OOS (egp_wfo)")
    print("- CPCV at optimization (egp_cpcv_pipeline)")
    print("- global sensitivity (egp_sobol / egp_bohb)")
    if not args.configs:
        print("- overfitting battery (PBO/RC/SPA): pass --configs with one CSV per config")
    if not args.features:
        print("- ML model validation: pass --features with a CLEAN features CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
