//+------------------------------------------------------------------+
//| EGP_MHO_DealsExport.mqh   (auto-generated)                       |
//| Exports the full deal history to a CSV for the offline AFML       |
//| arsenal (deflated gate, Monte-Carlo, CPCV).                      |
//| API verified: HistorySelect / HistoryDealsTotal /                |
//| HistoryDealGetTicket / HistoryDealGet{Double,Integer,String}.    |
//| File written to the terminal's COMMON folder (Terminal\Common\Files), accessible to all agents.                                       |
//| Call EGP_ExportDeals() once at end of test: at the start of      |
//| OnTester() (before the return) or in OnDeinit().                 |
//+------------------------------------------------------------------+
#ifndef EGP_MHO_DEALSEXPORT_MQH
#define EGP_MHO_DEALSEXPORT_MQH

//--- Writes every history deal to `filename` (CSV, ',' separator). Returns true on success.
bool EGP_ExportDeals(const string filename="EGP_deals.csv")
  {
   //--- 1) select the full history (MANDATORY before any read)
   if(!HistorySelect(0, TimeCurrent()))
     {
      PrintFormat("[EGP_ExportDeals] HistorySelect failed (err=%d)", GetLastError());
      return(false);
     }
   int total = HistoryDealsTotal();

   //--- 2) open the CSV
   int h = FileOpen(filename, FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON, ',');
   if(h == INVALID_HANDLE)
     {
      PrintFormat("[EGP_ExportDeals] FileOpen('%s') failed (err=%d)", filename, GetLastError());
      return(false);
     }

   //--- 3) header (must match egp_mt5_deals._DEFAULT_COLMAP on the Python side)
   FileWrite(h, "Time","Deal","Position","Symbol","Type","Direction","Volume","Price","Commission","Swap","Fee","Profit");

   //--- 4) one row per deal
   for(int i=0; i<total; i++)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket == 0) continue;                                   // unreadable deal -> skip

      datetime d_time   = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
      long     d_type   = HistoryDealGetInteger(ticket, DEAL_TYPE);          // ENUM_DEAL_TYPE
      long     d_entry  = HistoryDealGetInteger(ticket, DEAL_ENTRY);         // ENUM_DEAL_ENTRY (0..3)
      long     d_pos    = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);   // position id
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
     }

   FileClose(h);
   PrintFormat("[EGP_ExportDeals] %d deals written to '%s'", total, filename);
   return(true);
  }

#endif // EGP_MHO_DEALSEXPORT_MQH
