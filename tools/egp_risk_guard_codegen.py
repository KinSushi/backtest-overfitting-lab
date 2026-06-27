"""
egp_risk_guard_codegen.py - Generator of the MQL5 module `EGP_MHO_RiskGuard.mqh` (4 guards).

Implements in MQL5, with VERIFIED API, the logic proven on the Python side in egp_risk_guard_logic:
  P0  DRAWDOWN circuit breaker (total + daily)           -> EGP_RG_Update / EGP_RG_ShouldHalt
  P0  concurrent POSITION cap                             -> EGP_RG_PositionCapOK
  P1  normalized RISK sizing (gold-safe)                  -> EGP_RG_RiskLots (via OrderCalcProfit)
  P2  NEWS blackout (time windows)                        -> EGP_RG_InNewsBlackout
  P2/P3 BREAK-EVEN (SL -> entry +offset)                  -> EGP_RG_ApplyBreakeven

MQL5 API VERIFIED (official doc + forum consensus):
  - AccountInfoDouble(ACCOUNT_EQUITY); TimeToStruct/MqlDateTime (day, hour, day_of_week).
  - PositionsTotal / PositionGetTicket(i) / PositionGetString(POSITION_SYMBOL) /
    PositionGetInteger(POSITION_MAGIC,POSITION_TYPE) / PositionGetDouble(POSITION_PRICE_OPEN,SL,TP).
  - CTrade.PositionModify(ticket, sl, tp) (#include <Trade\\Trade.mqh>).
  - **GOLD-SAFE sizing**: OrderCalcProfit(type,_Symbol,1.0,open,sl,profit) -> loss of 1 lot; fallback
    on SYMBOL_TRADE_TICK_VALUE/SYMBOL_TRADE_TICK_SIZE. VERIFIED: TickValue is often WRONG on
    XAUUSD at some brokers (lots x10), hence OrderCalcProfit first.
  - SymbolInfoDouble : SYMBOL_VOLUME_MIN/MAX/STEP, SYMBOL_TRADE_TICK_SIZE/VALUE, SYMBOL_BID/ASK.

CRITICAL CAVEAT: emits MQL5 TEXT, NOT compilable here. The tests verify the API calls and
the inputs. #include in the EA and WIRE it (see the generated file header), then compile.
All inputs are RG_-prefixed to avoid collisions. RG_Magic must equal EA_MAGIC.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def generate_code(magic_default: int = 1111) -> str:
    """Returns the MQL5 source of the RiskGuard module (4 guards, RG_-prefixed inputs)."""
    code = f'''//+------------------------------------------------------------------+
//| EGP_MHO_RiskGuard.mqh   (auto-generated)                         |
//| 4 guards: drawdown circuit breaker, position cap, risk-based     |
//| sizing (gold-safe), break-even, news blackout.                   |
//| Verified API: AccountInfoDouble / PositionGet* / OrderCalcProfit |
//| / CTrade.PositionModify / TimeToStruct.                          |
//|                                                                  |
//| WIRING in the EA:                                                |
//|  - at the very top of OnTick()    : EGP_RG_Update();             |
//|  - before any NEW entry           : if(EGP_RG_ShouldHalt()) return;|
//|                                     if(!EGP_RG_PositionCapOK()) return;|
//|                                     if(EGP_RG_InNewsBlackout()) return;|
//|  - lot calc (if RG_RiskSizing_On):                               |
//|        lot = EGP_RG_RiskLots(ORDER_TYPE_BUY, ASK, stop_loss);    |
//|  - in ManageOpenPositions()       : EGP_RG_ApplyBreakeven(atr);  |
//| RG_Magic must equal EA_MAGIC.                                    |
//+------------------------------------------------------------------+
#ifndef EGP_MHO_RISKGUARD_MQH
#define EGP_MHO_RISKGUARD_MQH

#include <Trade\\Trade.mqh>

//==================== INPUTS (prefixe RG_) ====================
input group "===== EGP RiskGuard =====";
input long   RG_Magic              = {magic_default};  // must equal EA_MAGIC
//--- P0 drawdown circuit breaker
input bool   RG_DD_On              = true;
input double RG_MaxTotalDD_Pct     = 20.0;   // halt if total DD from peak >= X%
input double RG_MaxDailyDD_Pct     = 5.0;    // halt if daily DD >= X%
//--- P0 concurrent position cap
input int    RG_MaxPositions       = 3;      // <=0 => unlimited
//--- P1 risk-based sizing (gold-safe)
input bool   RG_RiskSizing_On      = true;
input double RG_RiskPct            = 0.5;    // % of equity risked per trade
//--- P2/P3 break-even
input bool   RG_BE_On              = true;
input double RG_BE_TriggerATR      = 1.0;    // triggers at +trigger*ATR of profit
input double RG_BE_OffsetATR       = 0.1;    // places the SL at entry +/- offset*ATR
//--- P2 news blackout (2 time windows; weekday 0=Sun..6=Sat, -1=all)
input bool   RG_News_On            = false;
input bool   RG_News1_On           = true;  input int RG_News1_WD=-1; input int RG_News1_SH=14; input int RG_News1_SM=25; input int RG_News1_EH=14; input int RG_News1_EM=45;
input bool   RG_News2_On           = false; input int RG_News2_WD=-1; input int RG_News2_SH=0;  input int RG_News2_SM=0;  input int RG_News2_EH=0;  input int RG_News2_EM=0;

//==================== INTERNAL STATE ====================
CTrade  egp_rg_trade;
double  egp_rg_peak_equity     = 0.0;
double  egp_rg_day_start_equity = 0.0;
int     egp_rg_day             = -1;

//==================== P0: drawdown ====================
//--- Call at the top of OnTick: tracks the equity peak and the day-start equity.
void EGP_RG_Update()
  {{
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   if(egp_rg_peak_equity <= 0.0) egp_rg_peak_equity = eq;
   if(eq > egp_rg_peak_equity)   egp_rg_peak_equity = eq;
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   if(dt.day != egp_rg_day) {{ egp_rg_day = dt.day; egp_rg_day_start_equity = eq; }}
  }}

//--- Should NEW entries be stopped? (true = halt)
bool EGP_RG_ShouldHalt()
  {{
   if(!RG_DD_On) return(false);
   double eq = AccountInfoDouble(ACCOUNT_EQUITY);
   double total_dd = (egp_rg_peak_equity     > 0.0) ? (egp_rg_peak_equity - eq) / egp_rg_peak_equity : 0.0;
   double daily_dd = (egp_rg_day_start_equity > 0.0) ? (egp_rg_day_start_equity - eq) / egp_rg_day_start_equity : 0.0;
   if(RG_MaxTotalDD_Pct > 0.0 && total_dd >= RG_MaxTotalDD_Pct/100.0)
     {{ PrintFormat("[RiskGuard] HALT total DD=%.4f >= %.4f", total_dd, RG_MaxTotalDD_Pct/100.0); return(true); }}
   if(RG_MaxDailyDD_Pct > 0.0 && daily_dd >= RG_MaxDailyDD_Pct/100.0)
     {{ PrintFormat("[RiskGuard] HALT daily DD=%.4f >= %.4f", daily_dd, RG_MaxDailyDD_Pct/100.0); return(true); }}
   return(false);
  }}

//==================== P0: position cap ====================
int EGP_RG_CountPositions()
  {{
   int c = 0;
   for(int i=PositionsTotal()-1; i>=0; i--)
     {{
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         PositionGetInteger(POSITION_MAGIC) == RG_Magic) c++;
     }}
   return(c);
  }}

bool EGP_RG_PositionCapOK()
  {{
   if(RG_MaxPositions <= 0) return(true);
   return(EGP_RG_CountPositions() < RG_MaxPositions);
  }}

//==================== P1: risk-based sizing (gold-safe) ====================
//--- Loss of 1 lot if price moves from open_price to sl_price (OrderCalcProfit first; gold-safe).
double EGP_RG_MoneyPerLot(ENUM_ORDER_TYPE type, double open_price, double sl_price)
  {{
   double profit = 0.0;
   if(OrderCalcProfit(type, _Symbol, 1.0, open_price, sl_price, profit) && profit != 0.0)
      return(MathAbs(profit));
   double ts = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tv = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double dist = MathAbs(open_price - sl_price);
   return( (ts > 0.0) ? dist / ts * tv : 0.0 );
  }}

//--- Lots to risk RG_RiskPct% of equity if the SL is hit. 0 if RG_RiskSizing_On=false.
double EGP_RG_RiskLots(ENUM_ORDER_TYPE type, double open_price, double sl_price)
  {{
   if(!RG_RiskSizing_On) return(0.0);
   double risk_money = AccountInfoDouble(ACCOUNT_EQUITY) * (RG_RiskPct/100.0);
   double mpl = EGP_RG_MoneyPerLot(type, open_price, sl_price);
   if(mpl <= 0.0 || risk_money <= 0.0) return(0.0);
   double lots = risk_money / mpl;
   double vmin = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double vmax = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double vstep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   if(vstep > 0.0) lots = MathRound(lots/vstep) * vstep;
   lots = MathMax(vmin, MathMin(vmax, lots));
   return(lots);
  }}

//==================== P2/P3: break-even ====================
//--- Moves the SL of the EA positions to entry (+/- offset) once +trigger*ATR is reached.
void EGP_RG_ApplyBreakeven(double atr)
  {{
   if(!RG_BE_On || atr <= 0.0) return;
   double trigger = RG_BE_TriggerATR * atr;
   double offset  = RG_BE_OffsetATR * atr;
   for(int i=PositionsTotal()-1; i>=0; i--)
     {{
      ulong tk = PositionGetTicket(i);
      if(tk == 0) continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol ||
         PositionGetInteger(POSITION_MAGIC) != RG_Magic) continue;
      long   type  = PositionGetInteger(POSITION_TYPE);
      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double sl    = PositionGetDouble(POSITION_SL);
      double tp    = PositionGetDouble(POSITION_TP);
      if(type == POSITION_TYPE_BUY)
        {{
         double cur = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         if(cur - entry >= trigger)
           {{ double nsl = entry + offset; if(sl == 0.0 || nsl > sl) egp_rg_trade.PositionModify(tk, nsl, tp); }}
        }}
      else if(type == POSITION_TYPE_SELL)
        {{
         double cur = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         if(entry - cur >= trigger)
           {{ double nsl = entry - offset; if(sl == 0.0 || nsl < sl) egp_rg_trade.PositionModify(tk, nsl, tp); }}
        }}
     }}
  }}

//==================== P2: news blackout ====================
bool egp_rg_in_window(int wd, int hh, int mm, bool on, int w_wd, int sH, int sM, int eH, int eM)
  {{
   if(!on) return(false);
   if(w_wd >= 0 && w_wd != wd) return(false);
   int t = hh*60 + mm, s = sH*60 + sM, e = eH*60 + eM;
   return(t >= s && t < e);
  }}

bool EGP_RG_InNewsBlackout()
  {{
   if(!RG_News_On) return(false);
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   if(egp_rg_in_window(dt.day_of_week, dt.hour, dt.min, RG_News1_On, RG_News1_WD, RG_News1_SH, RG_News1_SM, RG_News1_EH, RG_News1_EM)) return(true);
   if(egp_rg_in_window(dt.day_of_week, dt.hour, dt.min, RG_News2_On, RG_News2_WD, RG_News2_SH, RG_News2_SM, RG_News2_EH, RG_News2_EM)) return(true);
   return(false);
  }}

#endif // EGP_MHO_RISKGUARD_MQH
'''
    return code


def generate(root: str, out: Optional[str] = None, magic_default: int = 1111) -> Path:
    """Writes EGP_MHO_RiskGuard.mqh (default: MQL5/Experts/) and returns the path."""
    root = Path(root)
    out_path = Path(out) if out else root / "MQL5" / "Experts" / "EGP_MHO_RiskGuard.mqh"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generate_code(magic_default), encoding="utf-8")
    return out_path
