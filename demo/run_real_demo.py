#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REAL-DATA reproducible demo.

Runs the EGP validation battery on the CSVs an EA exported to the MT5 shared folder
(`...\\Terminal\\Common\\Files`), found AUTOMATICALLY. No path to type:

    python demo/run_real_demo.py                      # auto-detects Common\\Files\\EGP_deals.csv
    python demo/run_real_demo.py --deals MyEA_deals.csv   # any other EA's export
    python demo/run_real_demo.py --common-files "C:\\...\\Terminal\\Common\\Files"  # explicit folder

EA-agnostic: the pipeline only needs the MT5 deals format, so this works for ANY EA that exports
its deals (via EGP_MHO_DealsExport.mqh). Thin CLI: all logic lives in tools/ (egp_mt5_collect,
egp_full_report).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproducible EGP demo on REAL MT5 exports (auto-located).")
    ap.add_argument("--common-files", default=None,
                    help="MT5 'Common\\Files' folder (auto-detected if omitted).")
    ap.add_argument("--deals", default="EGP_deals.csv",
                    help="Deals CSV filename (default EGP_deals.csv; change for another EA).")
    ap.add_argument("--features", default="EGP_features.csv",
                    help="Features CSV filename (optional; default EGP_features.csv).")
    ap.add_argument("--deposit", type=float, default=10000.0, help="Initial deposit (default 10000).")
    ap.add_argument("--unsigned", action="store_true", help="Treat P&L as UNSIGNED (default: signed).")
    args = ap.parse_args()

    from egp_mt5_collect import mt5_common_files_dir, features_rows_per_position
    from egp_full_report import full_report, render_markdown

    folder = args.common_files or mt5_common_files_dir()
    if not folder:
        print("ERROR: could not locate the MT5 'Common\\Files' folder automatically.")
        print('  Pass it explicitly: --common-files "C:\\...\\Terminal\\Common\\Files"')
        return 2

    deals = os.path.join(folder, args.deals)
    if not os.path.isfile(deals):
        print(f"ERROR: deals CSV not found: {deals}")
        print("  Run a backtest first (the EA exports it via EGP_MHO_DealsExport.mqh), or pass --deals NAME.")
        return 2

    print(f"[real demo] Common\\Files : {folder}")
    print(f"[real demo] deals        : {args.deals}")

    # features are optional; refuse them if they accumulated across runs (ambiguous join)
    features_arg = None
    feats = os.path.join(folder, args.features)
    if os.path.isfile(feats):
        n, npos, mx = features_rows_per_position(feats)
        if mx > 1:
            print(f"[real demo] features     : ACCUMULATED ({mx} rows for one Position) -> model path SKIPPED")
            print("             (re-run ONE clean backtest with the fixed EGP_MHO_FeaturesExample.mqh)")
        else:
            features_arg = feats
            print(f"[real demo] features     : {args.features} (clean) -> model path ON")
    else:
        print("[real demo] features     : none -> model path off")

    rep = full_report(deals, is_path=True, initial_deposit=args.deposit,
                      signed=not args.unsigned, features_csv=features_arg)
    print()
    print(render_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
