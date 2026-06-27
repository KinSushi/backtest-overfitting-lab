"""
egp_costs.py - Transaction cost model and cost-robustness stress test.

Many "profitable" backtests die once real costs are applied. This module
(1) models the per-trade cost (spread + commission + slippage, + Almgren-Chriss impact
optionally) and (2) stress-tests: does the edge survive +50% (or +X%) of costs?

Sources:
  - Almgren & Chriss (2000), "Optimal Execution of Portfolio Transactions", Journal of Risk
    3(2):5-39. Execution price S^exec = S0 + eta*v (TEMPORARY impact, instantaneous)
    + gamma*(x - x0) (PERMANENT impact, accumulated), linear impacts; spread term
    sign(v)*S/2. Permanent cost in closed form = 0.5*gamma*X^2 (independent of the
    trajectory). For RETAIL lots, the impact (proportional to Q^2) is NEGLIGIBLE
    next to spread/commission/slippage (proportional to Q); it is provided for completeness
    and for large-size sensitivity analysis.

HONESTY: the default XAUUSD values below are ILLUSTRATIVE retail ORDERS OF MAGNITUDE
(to be calibrated on the real broker, e.g. Admiral Markets: spread, commission,
contract size, account currency). They are not facts.

Pure-Python. Input = series of GROSS per-trade P&L (before costs) + sizes in lots.
The gross P&L must be extracted from MT5 (P0 dependency, like the deflated gate).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


@dataclass
class CostModel:
    """Costs in ACCOUNT CURRENCY. spread_price/slippage_price are in PRICE units
    (e.g. USD per ounce for XAUUSD); converted to money via contract_size."""
    spread_price: float = 0.30          # full spread in price (e.g. ~30 XAUUSD points)
    commission_per_lot: float = 7.0     # round-trip commission per lot (account currency)
    slippage_price: float = 0.10        # slippage in price, per side
    contract_size: float = 100.0        # ounces per lot (standard XAUUSD)
    slippage_sides: int = 2             # 2 = entry + exit
    # Almgren-Chriss impact (OPTIONAL; 0 => disabled, retail case)
    ac_eta: float = 0.0                 # temporary impact coef (price per (lot/time))
    ac_gamma: float = 0.0               # permanent impact coef (price per lot)
    ac_exec_time: float = 1.0           # execution duration (same units as eta)

    def linear_cost(self, lots: float) -> float:
        """Cost proportional to size: spread + commission + slippage."""
        spread_c = self.spread_price * self.contract_size * lots
        comm_c = self.commission_per_lot * lots
        slip_c = self.slippage_price * self.contract_size * lots * self.slippage_sides
        return spread_c + comm_c + slip_c

    def impact_cost(self, lots: float) -> float:
        """Almgren-Chriss impact cost (linear): permanent 0.5*gamma*Q^2
        + temporary eta*Q^2/T. Proportional to Q^2 -> negligible at small lots."""
        if self.ac_gamma == 0.0 and self.ac_eta == 0.0:
            return 0.0
        Q = lots * self.contract_size
        perm = 0.5 * self.ac_gamma * Q * Q
        temp = (self.ac_eta * Q * Q / self.ac_exec_time) if self.ac_exec_time > 0 else 0.0
        return perm + temp

    def per_trade_cost(self, lots: float) -> float:
        return self.linear_cost(lots) + self.impact_cost(lots)


def _lots_list(n: int, lots) -> List[float]:
    if isinstance(lots, (int, float)):
        return [float(lots)] * n
    L = [float(x) for x in lots]
    if len(L) != n:
        raise ValueError("len(lots) != len(gross_pnls)")
    return L


def apply_costs(gross_pnls: Sequence[float], lots, model: CostModel,
                cost_factor: float = 1.0) -> List[float]:
    """NET P&L = gross P&L - cost_factor * per_trade_cost."""
    g = [float(x) for x in gross_pnls]
    L = _lots_list(len(g), lots)
    return [g[i] - cost_factor * model.per_trade_cost(L[i]) for i in range(len(g))]


def net_metrics(gross_pnls: Sequence[float], lots, model: CostModel,
                cost_factor: float = 1.0) -> dict:
    g = [float(x) for x in gross_pnls]
    L = _lots_list(len(g), lots)
    cost_total = cost_factor * sum(model.per_trade_cost(x) for x in L)
    gross_total = sum(g)
    net = apply_costs(g, L, model, cost_factor)
    net_total = sum(net)
    wins = sum(x for x in net if x > 0)
    losses = -sum(x for x in net if x < 0)
    pf = (wins / losses) if losses > 0 else float("inf")
    return {"gross_total": gross_total, "cost_total": cost_total,
            "net_total": net_total, "net_profit_factor": pf,
            "n_trades": len(g), "cost_factor": cost_factor}


def breakeven_cost_multiplier(gross_pnls: Sequence[float], lots,
                              model: CostModel) -> float:
    """Cost factor f* such that total net P&L = 0 (net_total(f)=gross-f*cost).
    f* = gross_total / total_cost_at_f1. >1 => margin; <=1 => already unprofitable net."""
    g = [float(x) for x in gross_pnls]
    L = _lots_list(len(g), lots)
    cost1 = sum(model.per_trade_cost(x) for x in L)
    gross_total = sum(g)
    if cost1 <= 0:
        return float("inf")
    if gross_total <= 0:
        return 0.0
    return gross_total / cost1


def cost_stress(gross_pnls: Sequence[float], lots, model: CostModel,
                factors: Sequence[float] = (1.0, 1.25, 1.5, 2.0)
                ) -> List[Tuple[float, float, bool]]:
    """For each cost factor: (factor, net_total, survives=net_total>0)."""
    out = []
    for f in factors:
        nt = net_metrics(gross_pnls, lots, model, cost_factor=f)["net_total"]
        out.append((f, nt, nt > 0.0))
    return out


def survives_plus_pct(gross_pnls: Sequence[float], lots, model: CostModel,
                      pct: float = 50.0) -> bool:
    """Does the edge stay net-profitable with +pct% of costs? (default +50%)."""
    f = 1.0 + pct / 100.0
    return net_metrics(gross_pnls, lots, model, cost_factor=f)["net_total"] > 0.0


# =========================================================================== #
# P2 extension: overnight costs (swap, triple Wednesday) + execution realism  #
# =========================================================================== #
# Sources:
#   - MetaTrader 5, symbol properties: SYMBOL_SWAP_LONG / SYMBOL_SWAP_SHORT
#     (swap rate per night and per lot); SYMBOL_SWAP_ROLLOVER3DAYS = day when a
#     TRIPLE swap is applied (default WEDNESDAY, to cover the T+2 settlement of the
#     weekend). FX/metal markets are closed Saturday/Sunday: no swap, but the
#     Wednesday triple compensates. (MetaTrader 5 Help / MQL5 SymbolInfoInteger/Double.)
# Convention: swap rate per night and per lot, in ACCOUNT CURRENCY and SIGNED (negative = cost).
from datetime import datetime as _datetime, time as _time, timedelta as _timedelta
from dataclasses import dataclass as _dataclass


@_dataclass
class SwapModel:
    swap_long_per_lot: float = -3.0     # per night and per lot, account currency (sign: <0 = cost)
    swap_short_per_lot: float = -1.5
    rollover_hour: int = 22             # server hour of the rollover
    triple_weekday: int = 2             # 0=Monday .. 2=WEDNESDAY (MT5 default)
    trading_days: tuple = (0, 1, 2, 3, 4)   # Mon-Fri (no weekend rollover)


def count_rollovers(entry_dt: "_datetime", exit_dt: "_datetime",
                    rollover_hour: int = 22, triple_weekday: int = 2,
                    trading_days=(0, 1, 2, 3, 4)) -> dict:
    """Counts the rollovers crossed between entry and exit. A rollover occurs at
    rollover_hour each trading day; it weighs 3 nights on triple_weekday (Wednesday),
    0 on weekends. Returns {effective, calendar, triple}. Intraday (no rollover crossed)
    => effective=0 (scalps pay no swap)."""
    if exit_dt <= entry_dt:
        return {"effective": 0, "calendar": 0, "triple": 0}
    d = entry_dt.date()
    end_d = exit_dt.date()
    eff = cal = trip = 0
    while d <= end_d:
        r = _datetime.combine(d, _time(hour=rollover_hour))
        if entry_dt < r <= exit_dt and d.weekday() in trading_days:
            cal += 1
            if d.weekday() == triple_weekday:
                eff += 3
                trip += 1
            else:
                eff += 1
        d = d + _timedelta(days=1)
    return {"effective": eff, "calendar": cal, "triple": trip}


def position_swap(entry_dt: "_datetime", exit_dt: "_datetime", lots: float,
                  direction: str, model: "SwapModel") -> dict:
    """Total swap cost of a position = effective_nights * lots * rate (signed).
    direction: 'long' or 'short'. Returns {swap, rollovers}."""
    rate = model.swap_long_per_lot if direction == "long" else model.swap_short_per_lot
    roll = count_rollovers(entry_dt, exit_dt, model.rollover_hour,
                           model.triple_weekday, model.trading_days)
    return {"swap": roll["effective"] * lots * rate, "rollovers": roll}


def weekend_crossings(entry_dt: "_datetime", exit_dt: "_datetime") -> int:
    """Number of weekends covered by the position (weekend-gap exposure).
    Counts the Saturdays in [entry_date, exit_date]."""
    if exit_dt <= entry_dt:
        return 0
    d = entry_dt.date()
    end_d = exit_dt.date()
    c = 0
    while d <= end_d:
        if d.weekday() == 5:        # Saturday
            c += 1
        d = d + _timedelta(days=1)
    return c


def slippage_vol_scaled(base_slippage_price: float, k_vol: float,
                        realized_vol: float) -> float:
    """Regime-dependent slippage: base + k * realized volatility (more slippage in
    volatile markets). To inject into CostModel.slippage_price before per_trade_cost()."""
    return base_slippage_price + k_vol * max(0.0, realized_vol)


def total_position_cost(entry_dt: "_datetime", exit_dt: "_datetime", lots: float,
                        direction: str, cost_model: "CostModel",
                        swap_model: "SwapModel") -> dict:
    """TOTAL cost of a position = linear (spread+commission+slippage) + impact + swap.
    Returns the detail for audit. The signed swap is added (negative = cost)."""
    linear = cost_model.linear_cost(lots)
    impact = cost_model.impact_cost(lots)
    sw = position_swap(entry_dt, exit_dt, lots, direction, swap_model)
    weekends = weekend_crossings(entry_dt, exit_dt)
    # positive total cost = costs deducted; the signed swap is subtracted if negative
    total = linear + impact - sw["swap"]
    return {"linear": linear, "impact": impact, "swap": sw["swap"],
            "rollovers": sw["rollovers"], "weekend_crossings": weekends,
            "total_cost": total}
