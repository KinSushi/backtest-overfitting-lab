# -*- coding: utf-8 -*-
"""
egp_mt5_collect.py - Collect / inspect MT5 export CSVs for the pipeline (no formula; pure plumbing).

Reusable LIBRARY logic that used to live in the demo CLIs:
  - build_opt_config_trades : read N deals CSVs (one per optimization config) -> {config: [trades]},
    ready for egp_opt_validation.align_to_grid / egp_full_report(opt_config_trades=...).
  - features_rows_per_position : detect a features CSV accumulated across runs (rows per Position),
    so callers can refuse an ambiguous features<->deals join.

Pure-Python. Depends on egp_mt5_deals (parse_deals_csv, trades_from_deals).
"""
from __future__ import annotations

import csv
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

from egp_mt5_deals import parse_deals_csv, trades_from_deals


def build_opt_config_trades(paths: Sequence[str], *,
                            signed: bool = True) -> Tuple[Dict[str, List[Dict]], Dict[str, str]]:
    """N MT5 deals CSV paths -> ({config_name: [trade dicts]}, {config_name: source_path}).

    config_name = filename stem. Each trade dict carries `net_pnl` and `exit_time`, the keys that
    egp_opt_validation.align_to_grid expects. Balance/deposit rows are dropped by parse_deals_csv.
    """
    opt: Dict[str, List[Dict]] = {}
    src: Dict[str, str] = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        deals = parse_deals_csv(p, is_path=True)
        opt[name] = trades_from_deals(deals, signed=signed)
        src[name] = p
    return opt, src


def features_rows_per_position(features_csv: str) -> Tuple[int, int, int]:
    """Return (n_rows, n_positions, max_rows_per_position) for a features CSV.

    max_rows_per_position > 1 means the file accumulated across runs (FILE_COMMON + append) -> the
    features<->deals join by Position is ambiguous and the model path should be skipped.
    """
    with open(features_csv, newline="", encoding="utf-8-sig") as f:
        c = Counter(r.get("Position", "") for r in csv.DictReader(f))
    if not c:
        return 0, 0, 0
    return sum(c.values()), len(c), max(c.values())


def mt5_common_files_dir() -> Optional[str]:
    """Best-effort path to MT5 'Common\\Files' (where FILE_COMMON writes), or None if not found.

    Windows:  %APPDATA%\\MetaQuotes\\Terminal\\Common\\Files
    macOS/Linux (MT5 under Wine): a couple of common Wine-prefix locations.
    Pass an explicit folder to the CLI to override this.
    """
    appdata = os.environ.get("APPDATA")                      # Windows
    if appdata:
        p = os.path.join(appdata, "MetaQuotes", "Terminal", "Common", "Files")
        if os.path.isdir(p):
            return p
    home = os.path.expanduser("~")
    user = os.environ.get("USER", os.environ.get("USERNAME", "user"))
    candidates = [
        os.path.join(home, ".wine", "drive_c", "users", user, "AppData", "Roaming",
                     "MetaQuotes", "Terminal", "Common", "Files"),
        os.path.join(home, "Library", "Application Support",
                     "net.metaquotes.wine.metatrader5", "drive_c", "users", "user",
                     "AppData", "Roaming", "MetaQuotes", "Terminal", "Common", "Files"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return None
