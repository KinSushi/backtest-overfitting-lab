"""
egp_deals_export_codegen.py - Generator of the MQL5 module `EGP_MHO_DealsExport.mqh`.

THIS IS THE P0 LOCK. The EA runs in the Strategy Tester; to apply the AFML arsenal (deflated
gate, Monte-Carlo, CPCV) on REAL and not synthetic data, the EA must EXPORT
its deals series. This module GENERATES the MQL5 code that writes the full deals history to a
CSV whose columns match EXACTLY the Python adapter `egp_mt5_deals.parse_deals_csv`
(writer<->reader co-design). The downstream Python then consumes this CSV (egp_real_gate).

MQL5 API VERIFIED against the official doc (mql5.com/en/docs/trading + .../constants/.../dealproperties):
  - HistorySelect(from, to) -> bool : must be called BEFORE any read (otherwise empty lists).
  - HistoryDealsTotal() -> int : number of deals in the selected list.
  - HistoryDealGetTicket(i) -> ulong : ticket of the i-th deal (0 on failure).
  - HistoryDealGetInteger(ticket, prop) : DEAL_TIME(datetime), DEAL_TYPE, DEAL_ENTRY, DEAL_MAGIC,
    DEAL_POSITION_ID, DEAL_TICKET, DEAL_ORDER  (ENUM_DEAL_PROPERTY_INTEGER).
  - HistoryDealGetDouble(ticket, prop)  : DEAL_VOLUME, DEAL_PRICE, DEAL_COMMISSION, DEAL_SWAP,
    DEAL_PROFIT, DEAL_FEE  (ENUM_DEAL_PROPERTY_DOUBLE).
  - HistoryDealGetString(ticket, prop)  : DEAL_SYMBOL.
  - ENUM_DEAL_ENTRY : DEAL_ENTRY_IN=0, DEAL_ENTRY_OUT=1, DEAL_ENTRY_INOUT=2, DEAL_ENTRY_OUT_BY=3.
  - Net result of a deal = DEAL_PROFIT + DEAL_COMMISSION + DEAL_SWAP + DEAL_FEE (commission/swap/fee
    signed by the API; confirmed by the official example: totalPL += profit + commission + swap).

CRITICAL CAVEAT: this module emits MQL5 TEXT; it CANNOT be compiled or executed here (no
MetaEditor/Windows). The tests verify that the emitted code contains the right verified API calls and
the exact header expected by the adapter. The actual compilation remains to be done in MetaEditor.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

# Columns EXACTLY aligned with egp_mt5_deals._DEFAULT_COLMAP
#   time->Time, deal->Deal, pos_id->Position, symbol->Symbol, type->Type, entry->Direction,
#   volume->Volume, price->Price, commission->Commission, swap->Swap, fee->Fee, profit->Profit
HEADER = ["Time", "Deal", "Position", "Symbol", "Type", "Direction", "Volume", "Price",
          "Commission", "Swap", "Fee", "Profit"]


def generate_code(default_filename: str = "EGP_deals.csv", common_folder: bool = True) -> str:
    """Returns the MQL5 source of the deals-export module."""
    file_flags = "FILE_WRITE|FILE_CSV|FILE_ANSI"
    if common_folder:
        file_flags += "|FILE_COMMON"
    header_args = ",".join(f'"{h}"' for h in HEADER)
    common_note = ("the terminal COMMON folder (Terminal\\Common\\Files), accessible to all agents"
                   if common_folder else "the test agent sandbox (MQL5\\Files)")
    code = f'''//+------------------------------------------------------------------+
//| EGP_MHO_DealsExport.mqh   (auto-generated)                       |
//| Exports the full deals history to a CSV for the offline          |
//| AFML arsenal (deflated gate, Monte-Carlo, CPCV).                 |
//| Verified API: HistorySelect / HistoryDealsTotal /                |
//| HistoryDealGetTicket / HistoryDealGet{{Double,Integer,String}}.  |
//| File written to {common_note}.                                   |
//| Call EGP_ExportDeals() once at the end of the test: at the start of  |
//| OnTester() (before the return) or in OnDeinit().                 |
//+------------------------------------------------------------------+
#ifndef EGP_MHO_DEALSEXPORT_MQH
#define EGP_MHO_DEALSEXPORT_MQH

//--- Writes all history deals to `filename` (CSV, ',' separator). Returns true on success.
bool EGP_ExportDeals(const string filename="{default_filename}")
  {{
   //--- 1) select the full history (MANDATORY before any read)
   if(!HistorySelect(0, TimeCurrent()))
     {{
      PrintFormat("[EGP_ExportDeals] HistorySelect failed (err=%d)", GetLastError());
      return(false);
     }}
   int total = HistoryDealsTotal();

   //--- 2) open the CSV
   int h = FileOpen(filename, {file_flags}, ',');
   if(h == INVALID_HANDLE)
     {{
      PrintFormat("[EGP_ExportDeals] FileOpen('%s') failed (err=%d)", filename, GetLastError());
      return(false);
     }}

   //--- 3) header (must match egp_mt5_deals._DEFAULT_COLMAP on the Python side)
   FileWrite(h, {header_args});

   //--- 4) one line per deal
   for(int i=0; i<total; i++)
     {{
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;                                   // unreadable deal -> skip

      datetime d_time   = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      long     d_type   = HistoryDealGetInteger(ticket, DEAL_TYPE);          // ENUM_DEAL_TYPE
      long     d_entry  = HistoryDealGetInteger(ticket, DEAL_ENTRY);         // ENUM_DEAL_ENTRY (0..3)
      long     d_pos    = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);   // position identifier
      string   d_symbol = HistoryDealGetString (ticket, DEAL_SYMBOL);
      double   d_volume = HistoryDealGetDouble (ticket, DEAL_VOLUME);
      double   d_price  = HistoryDealGetDouble (ticket, DEAL_PRICE);
      double   d_comm   = HistoryDealGetDouble (ticket, DEAL_COMMISSION);
      double   d_swap   = HistoryDealGetDouble (ticket, DEAL_SWAP);
      double   d_fee    = HistoryDealGetDouble (ticket, DEAL_FEE);
      double   d_profit = HistoryDealGetDouble (ticket, DEAL_PROFIT);

      //--- column order = HEADER above
      FileWrite(h,
                TimeToString(d_time, TIME_DATE|TIME_SECONDS),
                (long)ticket,        // Deal (ticket)
                d_pos,               // Position
                d_symbol,            // Symbol
                d_type,              // Type
                d_entry,             // Direction (ENUM_DEAL_ENTRY)
                d_volume,            // Volume
                d_price,             // Price
                d_comm,              // Commission
                d_swap,              // Swap
                d_fee,               // Fee
                d_profit);           // Profit
     }}

   FileClose(h);
   PrintFormat("[EGP_ExportDeals] %d deals written to '%s'", total, filename);
   return(true);
  }}

#endif // EGP_MHO_DEALSEXPORT_MQH
'''
    return code


def generate(root: str, out: Optional[str] = None, default_filename: str = "EGP_deals.csv",
             common_folder: bool = True) -> Path:
    """Writes EGP_MHO_DealsExport.mqh (default: MQL5/Experts/) and returns the path."""
    root = Path(root)
    out_path = Path(out) if out else root / "MQL5" / "Experts" / "EGP_MHO_DealsExport.mqh"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generate_code(default_filename, common_folder), encoding="utf-8")
    return out_path
