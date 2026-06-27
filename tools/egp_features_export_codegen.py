"""
egp_features_export_codegen.py - Generator (TEMPLATE) of the MQL5 module `EGP_MHO_FeaturesExport.mqh`.

Fills the SECONDARY GAP after P0: to validate the ML MODEL (forest/calibration/MDA/CPCV) on
REAL data, exporting the deals (output) is not enough; the FEATURE VECTOR is also needed
at each signal (input), join key = the opened POSITION identifier.

This module emits an MQL5 module that logs, at each signal, a `Position,Time,Side,f0..fk` line.
TEMPLATE: the feature EXPRESSIONS are EA-specific and CANNOT be guessed here.
The user calls `EGP_LogFeatures(position_id, time, side, names, values)` where their EA computes
its features, passing THEIR variables. On the Python side, `egp_real_pipeline` joins this CSV to the deals
(by Position) and runs the full pipeline on real data.

MQL5 file API used (official doc mql5.com/en/docs/files):
  - FileOpen(name, FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI[|FILE_COMMON], ',') -> handle.
  - FileSize(handle), FileSeek(handle, 0, SEEK_END) (append), FileWrite(handle, ...), FileClose.
  - ArraySize() for the length of the features array.

CRITICAL CAVEAT: emits MQL5 TEXT, not compilable/executable here. The tests verify the presence
of the right API calls and the header. The user MUST wire the calls and compile in MetaEditor.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def generate_code(default_filename: str = "EGP_features.csv", common_folder: bool = True) -> str:
    """Returns the MQL5 source (template) of the features logger."""
    file_flags = "FILE_READ|FILE_WRITE|FILE_CSV|FILE_ANSI"
    if common_folder:
        file_flags += "|FILE_COMMON"
    exists_arg = "FILE_COMMON" if common_folder else "0"
    code = f'''//+------------------------------------------------------------------+
//| EGP_MHO_FeaturesExport.mqh   (TEMPLATE auto-generated)           |
//| Logs the feature vector at each signal, with the position        |
//| identifier (join key on the Python side with the deals).         |
//| TO COMPLETE: pass YOUR real features to EGP_LogFeatures().       |
//| Verified file API: FileOpen/FileSize/FileSeek/FileWrite.         |
//+------------------------------------------------------------------+
#ifndef EGP_MHO_FEATURESEXPORT_MQH
#define EGP_MHO_FEATURESEXPORT_MQH

//--- Adds a feature line to the CSV. `names`/`values` same size. side = +1/-1.
//--- Call when a signal OPENS a position, with its position ticket.
bool EGP_LogFeatures(const long position_id, const datetime t, const int side,
                     const string &names[], const double &values[],
                     const string filename="{default_filename}")
  {{
   int n = ArraySize(values);
   if(n != ArraySize(names) || n == 0)
     {{
      PrintFormat("[EGP_LogFeatures] inconsistent names/values sizes (%d vs %d)", ArraySize(names), n);
      return(false);
     }}

   bool existed = FileIsExist(filename, {exists_arg});
   int h = FileOpen(filename, {file_flags}, ',');
   if(h == INVALID_HANDLE)
     {{
      PrintFormat("[EGP_LogFeatures] FileOpen('%s') failed (err=%d)", filename, GetLastError());
      return(false);
     }}

   //--- header once (file missing or empty): Position,Time,Side,<names>
   if(!existed || FileSize(h) == 0)
     {{
      string header = "Position,Time,Side";
      for(int j=0; j<n; j++) header += "," + names[j];
      FileWrite(h, header);
     }}

   //--- seek to end of file (append) then write the line
   FileSeek(h, 0, SEEK_END);
   string row = (string)position_id + "," + TimeToString(t, TIME_DATE|TIME_SECONDS) + "," + (string)side;
   for(int j=0; j<n; j++) row += "," + DoubleToString(values[j], 8);
   FileWrite(h, row);

   FileClose(h);
   return(true);
  }}

//--- Example call (adapt with YOUR features):
//    string  names[]  = {{"rsi","atr","mom20","spread"}};
//    double  values[] = {{rsi_val, atr_val, mom20_val, spread_val}};
//    EGP_LogFeatures(pos_id, TimeCurrent(), +1, names, values);

#endif // EGP_MHO_FEATURESEXPORT_MQH
'''
    return code


def generate(root: str, out: Optional[str] = None, default_filename: str = "EGP_features.csv",
             common_folder: bool = True) -> Path:
    """Writes EGP_MHO_FeaturesExport.mqh (default: MQL5/Experts/) and returns the path."""
    root = Path(root)
    out_path = Path(out) if out else root / "MQL5" / "Experts" / "EGP_MHO_FeaturesExport.mqh"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(generate_code(default_filename, common_folder), encoding="utf-8")
    return out_path
