//+------------------------------------------------------------------+
//| EGP_MHO_FeaturesExport.mqh   (auto-generated TEMPLATE)           |
//| Logs the feature vector at each signal, with the position id      |
//| (join key with the deals on the Python side).                     |
//| TO COMPLETE: pass YOUR real features to EGP_LogFeatures().         |
//| File API verified: FileOpen/FileSize/FileSeek/FileWrite.          |
//+------------------------------------------------------------------+
#ifndef EGP_MHO_FEATURESEXPORT_MQH
#define EGP_MHO_FEATURESEXPORT_MQH

//--- Appends a feature row to the CSV. `names`/`values` same size. side = +1/-1.
//--- Call when a signal OPENS a position, with its position ticket.
bool EGP_LogFeatures(const long position_id, const datetime t, const int side,
                     const string &names[], const double &values[],
                     const string filename="EGP_features.csv")
  {
   int n = ArraySize(values);
   if(n != ArraySize(names) || n == 0)
     {
      PrintFormat("[EGP_LogFeatures] names/values size mismatch (%d vs %d)", ArraySize(names), n);
      return(false);
     }

   bool existed = FileIsExist(filename, FILE_COMMON);
   int h = FileOpen(filename, FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI|FILE_COMMON, ',');
   if(h == INVALID_HANDLE)
     {
      PrintFormat("[EGP_LogFeatures] FileOpen('%s') failed (err=%d)", filename, GetLastError());
      return(false);
     }

   //--- header once (file missing or empty): Position,Time,Side,<names>
   if(!existed || FileSize(h) == 0)
     {
      string header = "Position,Time,Side";
      for(int j=0; j<n; j++) header += "," + names[j];
      FileWrite(h, header);
     }

   //--- seek to end of file (append) then write the row
   FileSeek(h, 0, SEEK_END);
   string row = (string)position_id + "," + TimeToString(t, TIME_DATE|TIME_SECONDS) + "," + (string)side;
   for(int j=0; j<n; j++) row += "," + DoubleToString(values[j], 8);
   FileWrite(h, row);

   FileClose(h);
   return(true);
  }

//--- Example call (adapt with YOUR features):
//    string  names[]  = {"rsi","atr","mom20","spread"};
//    double  values[] = {rsi_val, atr_val, mom20_val, spread_val};
//    EGP_LogFeatures(pos_id, TimeCurrent(), +1, names, values);

#endif // EGP_MHO_FEATURESEXPORT_MQH
