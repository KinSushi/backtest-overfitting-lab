//+------------------------------------------------------------------+
//|  EGP_MHO_FeaturesExample.mqh                                     |
//|  FILLED (compilable) EXAMPLE of feature computation + logging.   |
//|                                                                  |
//|  Fills the ".mqh gap": EGP_MHO_FeaturesExport.mqh provides only  |
//|  the WRITER (EGP_LogFeatures). Here we COMPUTE 8 real features   |
//|  and log them at each position open.                            |
//|                                                                  |
//|  MT5 API VERIFIED (mql5.com/en/docs/indicators):                |
//|   - iATR(symbol, period, ma_period)            -> handle (3 args)|
//|   - iRSI(symbol, period, ma_period, price)     -> handle         |
//|   - iMA (symbol, period, period, shift, meth, price) -> handle   |
//|   - CopyBuffer(handle, buf, start, count, arr) -> n copied / -1  |
//|   - handles created in OnInit (data not ready otherwise),       |
//|     released in OnDeinit via IndicatorRelease.                  |
//|                                                                  |
//|  ANTI-LOOK-AHEAD: every feature is read on the last CLOSED bar  |
//|  (shift = 1), never the forming bar (0).                        |
//|                                                                  |
//|  WARNING: code NOT COMPILED here (no MetaEditor). Compile on     |
//|  your machine. Adapt the feature list to your strategy.         |
//+------------------------------------------------------------------+
#ifndef EGP_MHO_FEATURESEXAMPLE_MQH
#define EGP_MHO_FEATURESEXAMPLE_MQH

#include "EGP_MHO_FeaturesExport.mqh"   // quotes = search next to the .mq5 (verified)

//--- indicator handles (global; created once in OnInit) ---
int egp_h_rsi      = INVALID_HANDLE;
int egp_h_atr      = INVALID_HANDLE;
int egp_h_ema_fast = INVALID_HANDLE;
int egp_h_ema_slow = INVALID_HANDLE;

//--- feature parameters (align with your logic if needed) ---
int    EGP_FEAT_RSI_PERIOD  = 14;
int    EGP_FEAT_ATR_PERIOD  = 14;
int    EGP_FEAT_EMA_FAST    = 20;
int    EGP_FEAT_EMA_SLOW    = 50;
int    EGP_FEAT_MOM_LOOKBACK= 20;   // momentum over 20 bars
string EGP_FEAT_FILENAME    = "EGP_features.csv";

//+------------------------------------------------------------------+
//| CALL IN OnInit(): creates the handles. Returns false on failure. |
//+------------------------------------------------------------------+
bool EGP_FeaturesInit()
  {
   egp_h_rsi      = iRSI(_Symbol, _Period, EGP_FEAT_RSI_PERIOD, PRICE_CLOSE);
   egp_h_atr      = iATR(_Symbol, _Period, EGP_FEAT_ATR_PERIOD);
   egp_h_ema_fast = iMA (_Symbol, _Period, EGP_FEAT_EMA_FAST, 0, MODE_EMA, PRICE_CLOSE);
   egp_h_ema_slow = iMA (_Symbol, _Period, EGP_FEAT_EMA_SLOW, 0, MODE_EMA, PRICE_CLOSE);
   if(egp_h_rsi==INVALID_HANDLE || egp_h_atr==INVALID_HANDLE ||
      egp_h_ema_fast==INVALID_HANDLE || egp_h_ema_slow==INVALID_HANDLE)
     {
      PrintFormat("[EGP_Features] handle creation failed (err=%d)", GetLastError());
      return(false);
     }
   // Start each run from a CLEAN features file: consistency with EGP_deals.csv (truncated each
   // run). Otherwise FILE_COMMON + append-mode writes ACCUMULATE features across all runs
   // -> the features<->deals join by Position becomes ambiguous on the Python side.
   FileDelete(EGP_FEAT_FILENAME, FILE_COMMON);
   return(true);
  }

//+------------------------------------------------------------------+
//| CALL IN OnDeinit(): releases the handles.                        |
//+------------------------------------------------------------------+
void EGP_FeaturesDeinit()
  {
   if(egp_h_rsi      != INVALID_HANDLE) IndicatorRelease(egp_h_rsi);
   if(egp_h_atr      != INVALID_HANDLE) IndicatorRelease(egp_h_atr);
   if(egp_h_ema_fast != INVALID_HANDLE) IndicatorRelease(egp_h_ema_fast);
   if(egp_h_ema_slow != INVALID_HANDLE) IndicatorRelease(egp_h_ema_slow);
   egp_h_rsi = egp_h_atr = egp_h_ema_fast = egp_h_ema_slow = INVALID_HANDLE;
  }

//--- read ONE indicator buffer value at the given shift (0 on failure) ---
double EGP_Buf1(const int handle, const int shift)
  {
   double b[];
   if(CopyBuffer(handle, 0, shift, 1, b) <= 0)
     {
      PrintFormat("[EGP_Features] CopyBuffer(handle=%d) failed (err=%d)", handle, GetLastError());
      return(0.0);
     }
   return(b[0]);
  }

//+------------------------------------------------------------------+
//| Computes 8 features on the last CLOSED bar (shift=1) and logs    |
//| them. `side` = +1 (buy) / -1 (sell).                            |
//| Call right AFTER opening a position, with its ticket.           |
//+------------------------------------------------------------------+
bool EGP_ComputeAndLogFeatures(const long position_id, const int side)
  {
   if(egp_h_rsi==INVALID_HANDLE)            // safety: init is mandatory in OnInit
     {
      Print("[EGP_Features] handles not initialized (call EGP_FeaturesInit in OnInit).");
      return(false);
     }
   const int S = 1;                         // last closed bar (anti look-ahead)

   double rsi   = EGP_Buf1(egp_h_rsi, S);
   double atr   = EGP_Buf1(egp_h_atr, S);
   double emaf  = EGP_Buf1(egp_h_ema_fast, S);
   double emas  = EGP_Buf1(egp_h_ema_slow, S);

   double c1    = iClose(_Symbol, _Period, S);
   double cN    = iClose(_Symbol, _Period, S + EGP_FEAT_MOM_LOOKBACK);
   double c5    = iClose(_Symbol, _Period, S + 5);
   double atr_s = (atr > 0.0 ? atr : _Point);     // division guard
   double c1_s  = (c1  > 0.0 ? c1  : 1.0);

   //--- features (normalized / dimensionless as much as possible) ---
   double f_rsi      = (rsi - 50.0) / 50.0;                  // RSI centered and scaled ~[-1,1]
   double f_mom      = (cN > 0.0 ? (c1 / cN - 1.0) : 0.0);   // momentum over N bars
   double f_ret5     = (c5 > 0.0 ? (c1 / c5 - 1.0) : 0.0);   // 5-bar return
   double f_emadist  = (c1 - emaf) / atr_s;                  // price-to-EMA distance in ATR units
   double f_emaslope = (emaf - emas) / atr_s;                // fast/slow EMA gap in ATR
   double f_atrpct   = atr / c1_s;                           // ATR as % of price (volatility)
   double f_spread   = (double)SymbolInfoInteger(_Symbol, SYMBOL_SPREAD); // spread (points)
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   double f_hour     = (double)dt.hour;                      // hour (intraday seasonality)

   string names[]  = {"rsi","mom","ret5","emadist","emaslope","atrpct","spread","hour"};
   double values[] = { f_rsi, f_mom, f_ret5, f_emadist, f_emaslope, f_atrpct, f_spread, f_hour };

   return EGP_LogFeatures(position_id, TimeCurrent(), side, names, values, EGP_FEAT_FILENAME);
  }

//+------------------------------------------------------------------+
//|  INTEGRATION (3 lines in your EA):                              |
//|   1) OnInit()   :  if(!EGP_FeaturesInit()) return(INIT_FAILED);  |
//|   2) OnDeinit() :  EGP_FeaturesDeinit();                         |
//|   3) right after the OrderSend that OPENS a position:           |
//|        long pid = (long)PositionGetInteger(POSITION_TICKET);     |
//|        // or retrieve the ticket of the opened position          |
//|        EGP_ComputeAndLogFeatures(pid, (buy_side ? +1 : -1));     |
//|                                                                  |
//|  The CSV (Common\Files\EGP_features.csv) has the SAME Position   |
//|  column as EGP_deals.csv -> join by Position on the Python side  |
//|  (run_real_pipeline). This is what wires the MODEL path onto     |
//|  your REAL data.                                                |
//+------------------------------------------------------------------+
#endif // EGP_MHO_FEATURESEXAMPLE_MQH
