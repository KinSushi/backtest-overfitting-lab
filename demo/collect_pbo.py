#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the TxC optimization matrix from N MT5 deals CSVs (one per config) and run the
overfitting battery: PBO/CSCV + White's Reality Check + Hansen SPA + deflated DSR.

This is the REAL-DATA path C. The MT5 optimizer truncates the deals CSV at each pass, so the
workflow is: run each config as a SEPARATE backtest, export its EGP_deals.csv under a distinct
name into one folder, then point this script at that folder.

    # one CSV per config in a folder (e.g. cfg_001.csv, cfg_002.csv, ...):
    python demo/collect_pbo.py "C:/path/to/configs_folder"
    python demo/collect_pbo.py "C:/path/to/configs_folder" --champion cfg_042

Thin CLI: the reusable logic lives in tools/egp_mt5_collect.py (build_opt_config_trades).
No PYTHONPATH needed (the script adds tools/ to sys.path).
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def main() -> int:
    ap = argparse.ArgumentParser(description="PBO/RC/SPA over N MT5 deals CSVs (one per config).")
    ap.add_argument("folder", help="Folder containing one deals CSV per config.")
    ap.add_argument("--pattern", default="*.csv", help="Glob pattern (default *.csv).")
    ap.add_argument("--champion", default=None,
                    help="Config name (filename stem) for the deals-gate section; default = best net.")
    ap.add_argument("--deposit", type=float, default=10000.0, help="Initial deposit (default 10000).")
    ap.add_argument("--unsigned", action="store_true", help="Treat P&L as UNSIGNED (default: signed).")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.folder, args.pattern)))
    if len(paths) < 2:
        print(f"ERROR: need >=2 deals CSVs (one per config) in {args.folder}; found {len(paths)}.")
        print("       PBO/CSCV is meaningless with a single configuration.")
        return 2

    from egp_mt5_collect import build_opt_config_trades
    from egp_full_report import full_report, render_markdown

    opt, src = build_opt_config_trades(paths, signed=not args.unsigned)

    # champion for the deals-gate section: explicit, else the best total net
    if args.champion and args.champion in src:
        champ = args.champion
    else:
        champ = max(opt, key=lambda k: sum(t.get("net_pnl", 0.0) for t in opt[k]))
    print(f"[collect] {len(opt)} configs | champion (deals-gate) = {champ}\n")

    rep = full_report(src[champ], is_path=True, initial_deposit=args.deposit,
                      signed=not args.unsigned, opt_config_trades=opt)
    print(render_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
