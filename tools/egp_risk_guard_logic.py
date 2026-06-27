"""
egp_risk_guard_logic.py - Logic of the 4 EA guards, in TESTABLE pure-Python.

Goal: prove via golden vectors the LOGIC of the protections added to the EA, independently of the
MQL5 compilation (impossible here). The codegen `egp_risk_guard_codegen` emits the MQL5 implementing
EXACTLY this logic with the verified API. Thus the computation is proven on the Python side, and the MQL5
is the faithful translation.

Garde-fous (audit EA) :
      1. RISK-normalized Sizing (P1) : lots = risk_money / (SL_distance * value_per_price_unit),
     value_per_price_unit = TickValue / TickSize. VERIFIED formula (MQL5 forum consensus + EarnForex).
     >>> On the MQL5 side (gold), TickValue is often WRONG: use OrderCalcProfit() (cf. codegen).
  2. DRAWDOWN circuit breaker (P0): halt if total or daily drawdown exceeds a threshold.
  3. Cap de POSITIONS concurrentes (P0).
  4. BREAK-EVEN (P2/P3): moves the SL to entry (+offset) once +trigger*ATR is reached, NEVER
     in the wrong direction.
  5. NEWS blackout (P2): time windows (day, start, end) where no entry is taken.

No invented formula: the sizing formula and the DD/BE logic are standard and verified.
Pur-Python (math).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


# --------------------------------------------------------------------------- 1) risk-based sizing
def value_per_price_unit(tick_value: float, tick_size: float) -> float:
    """Account-currency value of ONE price unit, for 1 lot = TickValue / TickSize."""
    return tick_value / tick_size if tick_size > 0 else 0.0


def normalize_volume(lots: float, vol_min: float, vol_max: float, vol_step: float) -> float:
    """Snaps to the broker step then clamps to [min, max]."""
    if vol_step > 0:
        lots = round(lots / vol_step) * vol_step
    lots = max(vol_min, min(vol_max, lots))
    # re-round to avoid floating-point errors (e.g. 0.30000000004)
    if vol_step > 0:
        lots = round(lots / vol_step) * vol_step
    return round(lots, 8)


def risk_based_lot(risk_money: float, sl_distance_price: float, tick_value: float, tick_size: float,
                   vol_min: float = 0.01, vol_max: float = 100.0, vol_step: float = 0.01,
                   money_per_lot: Optional[float] = None) -> float:
    """Lots to risk `risk_money` if the SL (at `sl_distance_price` from price) is hit.
    If `money_per_lot` is provided (e.g. via OrderCalcProfit on the MQL5 side), it takes precedence over the
    TickValue/TickSize. Returns 0 if undeterminable."""
    if money_per_lot is None:
        vppu = value_per_price_unit(tick_value, tick_size)
        money_per_lot = sl_distance_price * vppu        # loss of 1 lot over the SL distance
    if money_per_lot <= 0 or risk_money <= 0:
        return 0.0
    lots = risk_money / money_per_lot
    return normalize_volume(lots, vol_min, vol_max, vol_step)


# --------------------------------------------------------------------------- 2) coupe-circuit drawdown
def drawdown_pct(equity: float, reference: float) -> float:
    """Relative drawdown = (reference - equity) / reference (0 if reference<=0)."""
    if reference <= 0:
        return 0.0
    return max(0.0, (reference - equity) / reference)


def should_halt(equity: float, peak_equity: float, day_start_equity: float,
                max_total_dd_pct: float = 0.20, max_daily_dd_pct: float = 0.05) -> Dict:
    """Should NEW entries be stopped? Halt if total DD (from the peak) or daily DD
    (from the day start) exceeds its threshold. Returns {halt, reason, total_dd, daily_dd}."""
    total_dd = drawdown_pct(equity, peak_equity)
    daily_dd = drawdown_pct(equity, day_start_equity)
    halt = False
    reason = ""
    if max_total_dd_pct > 0 and total_dd >= max_total_dd_pct:
        halt, reason = True, f"DD_total {total_dd:.4f} >= {max_total_dd_pct:.4f}"
    elif max_daily_dd_pct > 0 and daily_dd >= max_daily_dd_pct:
        halt, reason = True, f"DD_journalier {daily_dd:.4f} >= {max_daily_dd_pct:.4f}"
    return {"halt": halt, "reason": reason, "total_dd": total_dd, "daily_dd": daily_dd}


# --------------------------------------------------------------------------- 3) cap de positions
def position_cap_ok(n_open: int, max_positions: int) -> bool:
    """True si on PEUT encore ouvrir (n_open < max). max_positions<=0 => illimite."""
    if max_positions <= 0:
        return True
    return n_open < max_positions


# --------------------------------------------------------------------------- 4) break-even
def breakeven_new_sl(entry: float, current: float, side: int, atr: float, current_sl: float,
                     trigger_mult: float = 1.0, offset_mult: float = 0.1) -> Optional[float]:
    """New SL if break-even triggers, otherwise None. side=+1 long, -1 short.
    Triggers if profit (in price) >= trigger_mult*ATR. Places the SL at entry +/- offset_mult*ATR.
    Returns an SL only if it IMPROVES the current SL (never in the wrong direction)."""
    if atr <= 0:
        return None
    trigger = trigger_mult * atr
    offset = offset_mult * atr
    if side > 0:                                        # long
        if (current - entry) >= trigger:
            new_sl = entry + offset
            if current_sl is None or new_sl > current_sl:
                return new_sl
    else:                                               # short
        if (entry - current) >= trigger:
            new_sl = entry - offset
            if current_sl is None or new_sl < current_sl:
                return new_sl
    return None


# --------------------------------------------------------------------------- 5) blackout news
def in_news_window(weekday: int, hour: int, minute: int,
                   windows: Sequence[Dict]) -> bool:
    """True if (weekday, hour:minute) falls in a blackout window. weekday 0=Sunday..6=Saturday
    (MqlDateTime.day_of_week convention). Each window = {weekday:int(-1=all), start_min, end_min}
    in minutes since midnight. Interval [start_min, end_min)."""
    t = hour * 60 + minute
    for w in windows:
        wd = w.get("weekday", -1)
        if wd != -1 and wd != weekday:
            continue
        if w["start_min"] <= t < w["end_min"]:
            return True
    return False
