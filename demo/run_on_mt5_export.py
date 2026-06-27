#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run the EGP validation battery on CSVs EXPORTED BY MT5
(EGP_deals.csv and, optionally, EGP_features.csv).

Designed to be launched simply, including on Windows PowerShell:

    cd C:\\path\\to\\EGP
    python demo\\run_on_mt5_export.py "C:\\...\\Common\\Files\\EGP_deals.csv"

    # with features (model path):
    python demo\\run_on_mt5_export.py "C:\\...\\EGP_deals.csv" --features "C:\\...\\EGP_features.csv"

Thin CLI: the reusable logic lives in tools/ (egp_full_report, egp_mt5_collect).
The script adds tools/ to sys.path automatically: no PYTHONPATH needed.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(os.path.dirname(HERE), "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)


def main() -> int:
    ap = argparse.ArgumentParser(description="EGP validation on an MT5 export.")
    ap.add_argument("deals", help="Path to the deals CSV (EGP_deals.csv).")
    ap.add_argument("--features", default=None, help="Path to the features CSV (optional).")
    ap.add_argument("--deposit", type=float, default=10000.0, help="Initial deposit (default 10000).")
    ap.add_argument("--unsigned", action="store_true",
                    help="Treat P&L as UNSIGNED (default: signed).")
    ap.add_argument("--force-features", action="store_true",
                    help="Pass features to the model even if they look accumulated.")
    args = ap.parse_args()

    if not os.path.isfile(args.deals):
        print(f"ERROR: deals file not found: {args.deals}")
        return 2

    from egp_full_report import full_report, render_markdown
    from egp_mt5_collect import features_rows_per_position

    features_arg = None
    if args.features:
        if not os.path.isfile(args.features):
            print(f"WARNING: features not found ({args.features}) -> ignored.")
        else:
            n, npos, mx = features_rows_per_position(args.features)
            if mx > 1 and not args.force_features:
                print("=" * 72)
                print("WARNING: the features file looks ACCUMULATED across several runs")
                print(f"  ({n} rows, {npos} positions, up to {mx} rows for a single Position).")
                print("  The features<->deals join by Position would be AMBIGUOUS: model path SKIPPED.")
                print("  -> Regenerate clean features (single run) or use --force-features.")
                print("=" * 72)
            else:
                features_arg = args.features

    rep = full_report(
        args.deals, is_path=True,
        initial_deposit=args.deposit, signed=not args.unsigned,
        features_csv=features_arg,
    )
    print(render_markdown(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
