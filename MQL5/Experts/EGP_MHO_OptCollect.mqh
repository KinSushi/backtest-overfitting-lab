//+------------------------------------------------------------------+
//| EGP_MHO_OptCollect.mqh                                           |
//| Per-pass deal capture during OPTIMIZATION, so the offline AFML    |
//| battery (demo/collect_pbo.py: PBO/CSCV + Reality Check + SPA +    |
//| deflated DSR) can judge EVERY configuration -- worst to best --   |
//| instead of a single residual EGP_deals.csv.                       |
//|                                                                  |
//| HOW IT WORKS                                                      |
//|   OnTester() fires once per pass. When MQLInfoInteger(            |
//|   MQL_OPTIMIZATION) is true, this writes ONE distinct file per    |
//|   config into the COMMON sandbox:                                 |
//|       Terminal\Common\Files\EGP_optpass\cfg_<params>.csv          |
//|   Distinct filenames (derived from the optimized inputs) mean     |
//|   parallel LOCAL agents never write the same file -> no race.     |
//|                                                                  |
//| SCOPE / LIMITATION  (verified on the MQL5 docs)                   |
//|   FILE_COMMON is shared between the terminal and its LOCAL agents |
//|   only. REMOTE / MQL5-Cloud agents do NOT have access to the      |
//|   common folder, so their files never come back. For a remote     |
//|   farm you must use the frames mechanism (FrameAdd in OnTester +  |
//|   OnTesterPass/FrameNext on the terminal) instead.                |
//|   Sources: mql5.com/en/book/automation/tester ;                   |
//|            mql5.com/en/articles/2720 ; forum 484272.              |
//|                                                                  |
//| NOTE: this header could not be compiled in the authoring          |
//|       environment (no MetaEditor). Compile and test on your       |
//|       machine; the optimization-pass behavior is reference code.  |
//+------------------------------------------------------------------+
#ifndef EGP_MHO_OPTCOLLECT_MQH
#define EGP_MHO_OPTCOLLECT_MQH

#include "EGP_MHO_DealsExport.mqh"   // EGP_ExportDeals() -- reused as-is (same CSV format)

//--- Make a string safe to use as a filename.
//--- Keeps [A-Za-z0-9_-]; turns '.' into 'p' (so 1.80 -> 1p80) and any other
//--- character into '_'. Caps the length so the full path stays well under MAX_PATH.
string EGP_SanitizeTag(const string tag_in)
  {
   string s = tag_in;
   int n = StringLen(s);
   for(int i=0; i<n; i++)
     {
      ushort c = StringGetCharacter(s, i);
      bool keep = (c>='0' && c<='9') ||
                  (c>='A' && c<='Z') ||
                  (c>='a' && c<='z') ||
                  (c=='_') || (c=='-');
      if(keep)            continue;
      if(c=='.')          StringSetCharacter(s, i, 'p');   // decimal point -> 'p'
      else                StringSetCharacter(s, i, '_');   // anything else -> '_'
     }
   int cap = 80;                                           // keep filenames sane
   if(StringLen(s) > cap)
      s = StringSubstr(s, 0, cap);
   return(s);
  }

//--- Capture the CURRENT pass's deals during optimization.
//--- `params_tag` should encode ONLY the optimized inputs (keep it short!), e.g.
//---     StringFormat("fE%d_sE%d_slm%.2f", InpFastEMA, InpSlowEMA, InpSL_ATRmult)
//--- Does nothing outside optimization (returns false) so single backtests keep
//--- exporting normally (e.g. EGP_ExportDeals() in OnDeinit).
//--- Returns true when a per-pass file was written.
bool EGP_ExportDealsForOptPass(const string params_tag, const string folder="EGP_optpass")
  {
   if(!MQLInfoInteger(MQL_OPTIMIZATION))
      return(false);                       // not optimizing: caller handles export

   //--- ensure the subfolder exists in the COMMON sandbox (true if it already exists)
   if(!FolderCreate(folder, FILE_COMMON))
      PrintFormat("[EGP_OptCollect] FolderCreate('%s') err=%d (continuing)",
                  folder, GetLastError());

   string fname = folder + "\\cfg_" + EGP_SanitizeTag(params_tag) + ".csv";
   return(EGP_ExportDeals(fname));         // same writer/format, -> Common\Files\<folder>\...
  }

#endif // EGP_MHO_OPTCOLLECT_MQH
