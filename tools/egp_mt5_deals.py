"""
egp_mt5_deals.py - Extraction of the MT5 DEAL SERIES -> per-trade P&L series.

This is the P0 LOCK: the deflated gate (DSR/PSR/RC/SPA), the Monte-Carlo (drawdown/ruin), the
CPCV and the meta-labeling all consume a TIME SERIES of per-trade results, which the current
parser (egp_mt5_report_parser) does not produce (it only exposes aggregates).

This module separates TWO things:
  1) the reconstruction LOGIC (group deals by position, compute net P&L, build the equity curve)
     - INDEPENDENT of the file format, hence testable offline;
  2) a format ADAPTER (parse_deals_csv) - specific to the export, to be CONFIRMED on a real
     report (header/locale varies by MT5 build).

Verified MT5 facts (official docs + MQL5 CDeal class):
  - Deal fields: ticket, time, entry (in/out/inout/reverse), pos_id, volume, price,
    commission, swap, fee, profit, balance.
  - "profit" = result of the EXIT; ENTRY deals have profit = 0.
  - NET result of a deal = profit - commission - fee - swap.
  - The "balance" column gives the cumulative equity (balance sense).
Source: MetaTrader 5 Help (Testing Visualization / Testing Report); CDeal (MQL5).
Pure-Python.
"""
from __future__ import annotations

import csv
import io
from typing import Dict, List, Optional, Sequence

# Codes ENUM_DEAL_ENTRY (MQL5)
ENTRY_IN, ENTRY_OUT, ENTRY_INOUT, ENTRY_OUT_BY = 0, 1, 2, 3


def net_result(deal: Dict, signed: bool = True) -> float:
    """NET result of a deal.
    signed=True (default, MQL5 HistoryDealGetDouble API convention): commission/swap/fee are
      already SIGNED (negative when charged) -> net = profit + commission + swap + fee.
    signed=False (some report tables display positive MAGNITUDES):
      net = profit - commission - swap - fee.
    The distinction MUST be confirmed on a real export (cf. the module warning)."""
    g = lambda k: float(deal.get(k, 0.0) or 0.0)
    if signed:
        return g("profit") + g("commission") + g("swap") + g("fee")
    return g("profit") - g("commission") - g("swap") - g("fee")


def trades_from_deals(deals: Sequence[Dict], signed: bool = True) -> List[Dict]:
    """Groups deals by position (pos_id) -> one TRADE per closed position.
    net_pnl = sum of net results; gross_pnl = sum of 'profit';
    costs = gross_pnl - net_pnl (total amount charged, always consistent with the convention).
    Ordered by EXIT instant. 'signed': see net_result()."""
    by_pos: Dict[object, List[Dict]] = {}
    order = []
    for d in deals:
        pid = d.get("pos_id", d.get("ticket"))
        if pid not in by_pos:
            by_pos[pid] = []
            order.append(pid)
        by_pos[pid].append(d)
    trades = []
    for pid in order:
        ds = by_pos[pid]
        ds_sorted = sorted(ds, key=lambda x: x.get("time", 0))
        net = sum(net_result(d, signed) for d in ds)
        gross = sum(float(d.get("profit", 0.0) or 0.0) for d in ds)
        entry_deal = ds_sorted[0]
        exit_deal = ds_sorted[-1]
        trades.append({
            "pos_id": pid,
            "entry_time": entry_deal.get("time"),
            "exit_time": exit_deal.get("time"),
            "volume": float(entry_deal.get("volume", 0.0) or 0.0),
            "gross_pnl": gross,
            "costs": gross - net,
            "net_pnl": net,
        })
    trades.sort(key=lambda t: (t["exit_time"] is None, t["exit_time"]))
    return trades


def pnl_series(trades: Sequence[Dict], net: bool = True) -> List[float]:
    """Per-trade P&L series (NET by default, or gross). Direct input to gate / MC / CPCV."""
    key = "net_pnl" if net else "gross_pnl"
    return [float(t[key]) for t in trades]


def equity_from_deals(deals: Sequence[Dict], initial_deposit: float = 0.0,
                      use_balance_field: bool = False) -> List[float]:
    """Equity curve. If use_balance_field and the 'balance' column exists, it is used
    as is (ordered by time); otherwise we accumulate the NET results of the deals."""
    ds = sorted(deals, key=lambda x: x.get("time", 0))
    if use_balance_field and ds and ds[0].get("balance") is not None:
        return [initial_deposit] + [float(d["balance"]) for d in ds]
    eq = [initial_deposit]
    s = initial_deposit
    for d in ds:
        s += net_result(d)
        eq.append(s)
    return eq


# --------------------------------------------------------------------------- #
# Format adapter (to CONFIRM on a real MT5 export)                           #
# --------------------------------------------------------------------------- #
_DEFAULT_COLMAP = {
    "time": "Time", "deal": "Deal", "symbol": "Symbol", "type": "Type",
    "entry": "Direction", "volume": "Volume", "price": "Price",
    "commission": "Commission", "swap": "Swap", "profit": "Profit",
    "balance": "Balance", "pos_id": "Position", "fee": "Fee",
}
_ENTRY_WORDS = {"in": ENTRY_IN, "out": ENTRY_OUT, "in/out": ENTRY_INOUT,
                "out by": ENTRY_OUT_BY}


def _num(x):
    if x is None:
        return 0.0
    s = str(x).strip().replace("\u00a0", "").replace(" ", "")
    if s == "":
        return 0.0
    s = s.replace(",", ".") if s.count(",") == 1 and s.count(".") == 0 else s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def parse_deals_csv(text_or_path: str, colmap: Optional[Dict] = None,
                    delimiter: Optional[str] = None, is_path: bool = False,
                    drop_non_trades: bool = True) -> List[Dict]:
    """Parses an MT5 deals CSV export into a list of canonical deals.
    WARNING: the exact header, the separator (tab/;/,) and the locale (decimal, date)
    MUST be confirmed on a REAL MT5 report; colmap is adjustable accordingly.
    'entry' accepts a word (in/out/in-out) or an integer (ENUM_DEAL_ENTRY code)."""
    cmap = dict(_DEFAULT_COLMAP)
    if colmap:
        cmap.update(colmap)
    text = open(text_or_path, encoding="utf-8", errors="replace").read() if is_path else text_or_path
    if delimiter is None:
        first = text.splitlines()[0] if text.splitlines() else ""
        delimiter = "\t" if "\t" in first else (";" if ";" in first else ",")
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    deals = []
    for row in reader:
        def col(canon):
            h = cmap.get(canon)
            return row.get(h) if h in row else None
        raw_entry = col("entry")
        entry = raw_entry
        if raw_entry is not None:
            w = str(raw_entry).strip().lower()
            entry = _ENTRY_WORDS.get(w, None)
            if entry is None:
                try:
                    entry = int(float(w))
                except ValueError:
                    entry = None
        # type de deal (ENUM_DEAL_TYPE) : 0=buy, 1=sell, 2=balance(depot), 3+=credit/charge/...
        raw_type = col("type")
        dtype = None
        if raw_type is not None and str(raw_type).strip() != "":
            try:
                dtype = int(float(str(raw_type).strip()))
            except ValueError:
                dtype = None
        pid = col("pos_id") if col("pos_id") not in (None, "") else col("deal")
        # filter NON-trades (deposits/credits): would distort P&L, equity and risk_of_ruin
        if drop_non_trades:
            if dtype is not None:
                if dtype not in (0, 1):
                    continue
            elif str(pid).strip() in ("0", "0.0", ""):  # fallback if the Type column is absent
                continue
        deals.append({
            "type": dtype,
            "time": col("time"),
            "ticket": col("deal"),
            "pos_id": col("pos_id") if col("pos_id") not in (None, "") else col("deal"),
            "entry": entry,
            "volume": _num(col("volume")),
            "price": _num(col("price")),
            "commission": _num(col("commission")),
            "swap": _num(col("swap")),
            "fee": _num(col("fee")),
            "profit": _num(col("profit")),
            "balance": _num(col("balance")) if col("balance") not in (None, "") else None,
        })
    return deals
