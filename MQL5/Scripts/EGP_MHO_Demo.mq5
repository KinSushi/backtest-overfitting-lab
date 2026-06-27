//+------------------------------------------------------------------+
//|  EGP_MHO_Demo.mq5  (Script)                                      |
//|  OFFLINE demonstration of the MHO module (Differential Evolution).|
//|  Optimizes a standard test function (sphere) - NO market         |
//|  operation, no orders. Used to verify the header compiles and    |
//|  converges.                                                      |
//|                                                                  |
//|  Place in MQL5/Scripts/ and run it on a chart.                  |
//+------------------------------------------------------------------+
#property copyright "EGP demo"
#property version   "1.00"
#property script_show_inputs

#include "..\\Experts\\EGP_MHO_Optimizer.mqh"

input int    InpDim         = 5;     // problem dimension
input int    InpPop         = 30;    // population size
input int    InpGenerations = 60;    // generations
input double InpF           = 0.7;   // mutation factor
input double InpCR          = 0.9;   // crossover rate
input int    InpSeed        = 42;    // RNG seed

//--- test function: shifted sphere. Optimum (MAX) reached at 2.0 everywhere.
double SphereFitness(const double &p[])
  {
   double s = 0.0;
   for(int j=0; j<ArraySize(p); j++)
     {
      double d = p[j] - 2.0;
      s += d * d;
     }
   return(-s);                 // we MAXIMIZE -> optimum = 0 when p[j]=2 everywhere
  }

//+------------------------------------------------------------------+
void OnStart()
  {
   double low[]; double high[];
   ArrayResize(low, InpDim);
   ArrayResize(high, InpDim);
   for(int j=0; j<InpDim; j++) { low[j] = -5.0; high[j] = 5.0; }

   double best[]; double bf = 0.0;
   int evals = EGP_MHO_DifferentialEvolution(low, high, InpDim,
                  InpPop, InpGenerations, InpF, InpCR, InpSeed,
                  SphereFitness, best, bf);

   PrintFormat("[EGP_MHO_Demo] evaluations=%d  best_fitness=%.8f  (theoretical optimum=0)", evals, bf);
   string s = "[EGP_MHO_Demo] best_params =";
   for(int j=0; j<ArraySize(best); j++) s += StringFormat(" %.4f", best[j]);
   Print(s, "   (target: 2.0 everywhere)");
   PrintFormat("[EGP_MHO_Demo] REMINDER: report evaluations=%d as n_trials to the Python battery (DSR).", evals);
  }
//+------------------------------------------------------------------+
