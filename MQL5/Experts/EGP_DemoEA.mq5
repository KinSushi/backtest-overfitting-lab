//+------------------------------------------------------------------+
//|  EGP_DemoEA.mq5                                                  |
//|  DEMONSTRATION EA for the EGP pipeline.                          |
//|                                                                  |
//|  WARNING - GENERIC, PUBLIC EXAMPLE STRATEGY:                     |
//|     fast/slow EMA crossover + RSI filter + ATR stops.           |
//|     This is NOT a production strategy. Its only purpose is to    |
//|     PRODUCE deals and features to exercise the Python pipeline   |
//|     (export -> validation) end to end.                          |
//|                                                                  |
//|  Wires the bundled .mqh modules:                                |
//|    - EGP_MHO_RiskGuard.mqh      (DD circuit-breaker, cap, sizing,|
//|                                  break-even, news)               |
//|    - EGP_MHO_DealsExport.mqh    (EGP_deals.csv at end of test)    |
//|    - EGP_MHO_FeaturesExample.mqh(EGP_features.csv at each trade)  |
//|                                                                  |
//|  MT5 API verified (mql5.com): CTrade.Buy/Sell(vol,sym,price,sl,  |
//|  tp,comment), PositionModify, iMA/iRSI/iATR + CopyBuffer.        |
//|                                                                  |
//|  WARNING - NOT COMPILED HERE (no MetaEditor). Compile on your    |
//|  machine.                                                        |
//+------------------------------------------------------------------+
#property copyright "EGP demo"
#property version   "1.00"
#property description "EGP Demo EA - EXAMPLE strategy (EMA cross + RSI + ATR). NOT a real strategy."
#property strict

#include <Trade\Trade.mqh>
#include "EGP_MHO_RiskGuard.mqh"        // provides the RG_* inputs + EGP_RG_* functions
#include "EGP_MHO_DealsExport.mqh"      // EGP_ExportDeals()
#include "EGP_MHO_OptCollect.mqh"       // EGP_ExportDealsForOptPass() (per-pass capture in optimization)
#include "EGP_MHO_FeaturesExample.mqh"  // EGP_FeaturesInit/Deinit/ComputeAndLogFeatures

//--- example-strategy inputs ---
input group "===== EGP Demo strategy (example) ====="
input long   EA_MAGIC        = 1111;   // MUST equal RG_Magic (RiskGuard module)
input int    InpFastEMA      = 12;     // fast EMA
input int    InpSlowEMA      = 26;     // slow EMA
input int    InpRSIPeriod    = 14;     // RSI period
input double InpRSI_Hi       = 55.0;   // buy if RSI > Hi on a bullish cross
input double InpRSI_Lo       = 45.0;   // sell if RSI < Lo on a bearish cross
input int    InpATRPeriod    = 14;     // ATR period (stops)
input double InpSL_ATRmult   = 1.8;    // SL = entry -/+ mult * ATR
input double InpTP_ATRmult   = 2.6;    // TP = entry +/- mult * ATR
input double InpFallbackLot  = 0.10;   // lot if RG_RiskSizing_On = false
input bool   InpLogFeatures  = true;   // log features at each open

CTrade  trade;
int     h_fast = INVALID_HANDLE, h_slow = INVALID_HANDLE;
int     h_rsi  = INVALID_HANDLE, h_atr  = INVALID_HANDLE;
datetime last_bar_time = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber((ulong)EA_MAGIC);
   if(EA_MAGIC != RG_Magic)
      PrintFormat("[EGP_DemoEA] WARNING: EA_MAGIC(%d) != RG_Magic(%d) - align them.", EA_MAGIC, RG_Magic);

   h_fast = iMA (_Symbol, _Period, InpFastEMA, 0, MODE_EMA, PRICE_CLOSE);
   h_slow = iMA (_Symbol, _Period, InpSlowEMA, 0, MODE_EMA, PRICE_CLOSE);
   h_rsi  = iRSI(_Symbol, _Period, InpRSIPeriod, PRICE_CLOSE);
   h_atr  = iATR(_Symbol, _Period, InpATRPeriod);
   if(h_fast==INVALID_HANDLE || h_slow==INVALID_HANDLE || h_rsi==INVALID_HANDLE || h_atr==INVALID_HANDLE)
     {
      PrintFormat("[EGP_DemoEA] handle creation failed (err=%d)", GetLastError());
      return(INIT_FAILED);
     }
   if(InpLogFeatures && !EGP_FeaturesInit())   // feature handles (example module)
      return(INIT_FAILED);
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| OnTester: called once per pass. During OPTIMIZATION, dump this    |
//| pass's deals to Common\Files\EGP_optpass\cfg_<params>.csv so the  |
//| AFML battery (demo/collect_pbo.py) can judge EVERY configuration  |
//| -- worst to best. OnDeinit still writes EGP_deals.csv on EVERY    |
//| run (single backtest, and the last optimization pass).            |
//| LOCAL agents only: remote/cloud agents need the frames mechanism. |
//+------------------------------------------------------------------+
double OnTester()
  {
   //--- SHORT, unique id from the OPTIMIZED inputs only. Add ONLY the
   //--- parameters you actually vary in the optimizer, else filenames
   //--- explode. '.' becomes 'p' via the sanitizer (1.80 -> 1p80).
   string tag = StringFormat("fE%d_sE%d_slm%.2f_tpm%.2f",
                             InpFastEMA, InpSlowEMA, InpSL_ATRmult, InpTP_ATRmult);
   EGP_ExportDealsForOptPass(tag);             // no-op unless MQL_OPTIMIZATION

   return(0.0);   // neutral criterion; the AFML battery does the real judging offline
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   EGP_ExportDeals();                          // ALWAYS -> Common\Files\EGP_deals.csv (single backtest
                                               // AND the last pass of an optimization). The per-pass
                                               // files are written separately by OnTester (EGP_optpass\).
   if(InpLogFeatures) EGP_FeaturesDeinit();
   if(h_fast!=INVALID_HANDLE) IndicatorRelease(h_fast);
   if(h_slow!=INVALID_HANDLE) IndicatorRelease(h_slow);
   if(h_rsi !=INVALID_HANDLE) IndicatorRelease(h_rsi);
   if(h_atr !=INVALID_HANDLE) IndicatorRelease(h_atr);
  }

//--- read one buffer value at the given shift (0 on failure) ---
double Buf(const int handle, const int shift)
  {
   double b[];
   if(CopyBuffer(handle, 0, shift, 1, b) <= 0) return(0.0);
   return(b[0]);
  }

//+------------------------------------------------------------------+
void OnTick()
  {
   //--- risk guards on EVERY tick ---
   EGP_RG_Update();
   if(RG_DD_On && EGP_RG_ShouldHalt()) return;            // drawdown circuit-breaker

   double atr = Buf(h_atr, 1);                            // ATR on last closed bar
   if(RG_BE_On && atr > 0.0) EGP_RG_ApplyBreakeven(atr);  // break-even every tick

   //--- ENTRY decision only once per bar (anti over-trading) ---
   datetime t = iTime(_Symbol, _Period, 0);
   if(t == last_bar_time) return;
   last_bar_time = t;

   if(!EGP_RG_PositionCapOK()) return;                    // position cap
   if(RG_News_On && EGP_RG_InNewsBlackout()) return;      // news blackout
   if(atr <= 0.0) return;

   //--- signal on the LAST CLOSED bar (shift 1/2): no look-ahead ---
   double f1 = Buf(h_fast, 1), s1 = Buf(h_slow, 1);
   double f2 = Buf(h_fast, 2), s2 = Buf(h_slow, 2);
   double rsi = Buf(h_rsi, 1);

   bool crossUp   = (f2 <= s2 && f1 > s1);
   bool crossDown = (f2 >= s2 && f1 < s1);

   if(crossUp && rsi > InpRSI_Hi)        OpenTrade(ORDER_TYPE_BUY,  atr);
   else if(crossDown && rsi < InpRSI_Lo) OpenTrade(ORDER_TYPE_SELL, atr);
  }

//+------------------------------------------------------------------+
//| Open a position: ATR stops, lot via RiskGuard, log features.     |
//+------------------------------------------------------------------+
void OpenTrade(const ENUM_ORDER_TYPE type, const double atr)
  {
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double price = (type==ORDER_TYPE_BUY ? ask : bid);
   int    dg    = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

   double sl = (type==ORDER_TYPE_BUY) ? price - InpSL_ATRmult*atr : price + InpSL_ATRmult*atr;
   double tp = (type==ORDER_TYPE_BUY) ? price + InpTP_ATRmult*atr : price - InpTP_ATRmult*atr;
   sl = NormalizeDouble(sl, dg);
   tp = NormalizeDouble(tp, dg);

   //--- sizing: risk % via RiskGuard (gold-safe) else fixed lot ---
   double lot = RG_RiskSizing_On ? EGP_RG_RiskLots(type, price, sl) : InpFallbackLot;
   if(lot <= 0.0) { Print("[EGP_DemoEA] computed lot <= 0, trade skipped."); return; }

   bool ok = (type==ORDER_TYPE_BUY)
             ? trade.Buy (lot, _Symbol, 0.0, sl, tp, "EGP demo")
             : trade.Sell(lot, _Symbol, 0.0, sl, tp, "EGP demo");
   if(!ok)
     {
      PrintFormat("[EGP_DemoEA] order rejected retcode=%d (%s)", trade.ResultRetcode(), trade.ResultRetcodeDescription());
      return;
     }

   //--- log features for THIS position (same Position id as EGP_deals.csv) ---
   if(InpLogFeatures && PositionSelect(_Symbol))
     {
      long pid  = (long)PositionGetInteger(POSITION_TICKET);
      int  side = (type==ORDER_TYPE_BUY ? +1 : -1);
      EGP_ComputeAndLogFeatures(pid, side);
     }
  }
//+------------------------------------------------------------------+
