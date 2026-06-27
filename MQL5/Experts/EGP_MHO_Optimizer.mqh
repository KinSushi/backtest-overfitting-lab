//+------------------------------------------------------------------+
//|  EGP_MHO_Optimizer.mqh                                           |
//|  Metaheuristic Optimizer (MHO) - Differential Evolution.         |
//|                                                                  |
//|  >>> POLICY (NON-NEGOTIABLE) <<<                                  |
//|   - OFFLINE use only (script / OnInit of a dedicated script).    |
//|   - NEVER call from OnTick().                                   |
//|   - NO live mutation of a running EA's parameters.              |
//|   - No automatic promotion: the output must pass the Python      |
//|     validation battery (DSR/PBO/RC/SPA) BEFORE any use.          |
//|                                                                  |
//|  Algorithm: Differential Evolution DE/rand/1/bin (Storn & Price, |
//|  1997). Search over a bounded parameter vector. MAXIMIZES the    |
//|  user-supplied fitness (e.g. out-of-sample Sharpe on saved       |
//|  data). The fitness MUST be deterministic and computed offline   |
//|  (no network call, no market state).                            |
//|                                                                  |
//|  Reference: R. Storn, K. Price, "Differential Evolution",        |
//|  Journal of Global Optimization 11:341-359, 1997.                |
//+------------------------------------------------------------------+
#ifndef EGP_MHO_OPTIMIZER_MQH
#define EGP_MHO_OPTIMIZER_MQH

//--- signature of the user-supplied fitness function ---
//    Returns a score to MAXIMIZE for the parameter vector `params`.
typedef double (*EGP_FitnessFunc)(const double &params[]);

//--- clamp a value into [lo, hi] ---
double EGP_MHO_Clamp(const double x, const double lo, const double hi)
  {
   if(x < lo) return(lo);
   if(x > hi) return(hi);
   return(x);
  }

//--- uniform draw [0,1) from MathRand() (0..32767) ---
double EGP_MHO_Rand01()
  {
   return((double)MathRand() / 32768.0);
  }

//+------------------------------------------------------------------+
//| Differential Evolution DE/rand/1/bin.                            |
//|                                                                  |
//|  low[], high[]    : per-dimension bounds (size = dim)           |
//|  dim              : number of parameters                        |
//|  pop_size         : population size (>= 4)                      |
//|  generations      : number of generations                      |
//|  F                : mutation factor (typically 0.5..0.9)        |
//|  CR               : crossover rate (typically 0.8..0.95)        |
//|  seed             : RNG seed (reproducibility)                  |
//|  fitness          : function to MAXIMIZE (pointer)             |
//|  best_params[]    : (out) best vector found                    |
//|  best_fitness     : (out) best fitness                         |
//|                                                                  |
//|  Returns the number of fitness evaluations performed            |
//|  ( = pop_size * (generations + 1) ), or 0 on invalid params.    |
//|  IMPORTANT: this number = n_trials to report to the Python DSR.  |
//+------------------------------------------------------------------+
int EGP_MHO_DifferentialEvolution(
        const double &low[], const double &high[], const int dim,
        const int pop_size, const int generations,
        const double F, const double CR, const int seed,
        EGP_FitnessFunc fitness,
        double &best_params[], double &best_fitness)
  {
   if(dim <= 0 || pop_size < 4 || generations < 0 ||
      ArraySize(low) != dim || ArraySize(high) != dim || fitness == NULL)
     {
      Print("[EGP_MHO] invalid parameters (dim/pop/bounds/fitness).");
      return(0);
     }

   MathSrand(seed);

   double pop[];   ArrayResize(pop,   pop_size * dim);   // flattened population
   double fit[];   ArrayResize(fit,   pop_size);
   double cand[];  ArrayResize(cand,  dim);              // trial vector
   double indiv[]; ArrayResize(indiv, dim);
   int    evals = 0;

   //--- random initialization within bounds ---
   for(int i=0; i<pop_size; i++)
     {
      for(int j=0; j<dim; j++)
         pop[i*dim + j] = low[j] + EGP_MHO_Rand01() * (high[j] - low[j]);
      for(int j=0; j<dim; j++) indiv[j] = pop[i*dim + j];
      fit[i] = fitness(indiv);
      evals++;
     }

   //--- evolution loop ---
   for(int g=0; g<generations; g++)
     {
      for(int i=0; i<pop_size; i++)
        {
         //--- pick a, b, c distinct and != i ---
         int a, b, c;
         do { a = MathRand() % pop_size; } while(a == i);
         do { b = MathRand() % pop_size; } while(b == i || b == a);
         do { c = MathRand() % pop_size; } while(c == i || c == a || c == b);

         int R = MathRand() % dim;                       // at least one dimension mutated
         for(int j=0; j<dim; j++)
           {
            if(EGP_MHO_Rand01() < CR || j == R)
               cand[j] = EGP_MHO_Clamp(pop[a*dim+j] + F * (pop[b*dim+j] - pop[c*dim+j]),
                                       low[j], high[j]);
            else
               cand[j] = pop[i*dim + j];
           }

         double cf = fitness(cand);
         evals++;
         if(cf >= fit[i])                                // greedy selection (maximization)
           {
            for(int j=0; j<dim; j++) pop[i*dim + j] = cand[j];
            fit[i] = cf;
           }
        }
     }

   //--- extract the best ---
   int bi = 0;
   for(int i=1; i<pop_size; i++)
      if(fit[i] > fit[bi]) bi = i;

   ArrayResize(best_params, dim);
   for(int j=0; j<dim; j++) best_params[j] = pop[bi*dim + j];
   best_fitness = fit[bi];

   return(evals);
  }

//+------------------------------------------------------------------+
//|  USAGE EXAMPLE (offline, in a dedicated script):                 |
//|                                                                  |
//|   // fitness to MAXIMIZE (here: -sphere, optimum at 0)           |
//|   double MyFitness(const double &p[])                            |
//|     { double s=0; for(int j=0;j<ArraySize(p);j++) s+=p[j]*p[j];  |
//|       return(-s); }                                             |
//|                                                                  |
//|   void OnStart()                                                |
//|     {                                                           |
//|      double lo[] = {-5,-5,-5};                                  |
//|      double hi[] = { 5, 5, 5};                                  |
//|      double best[]; double bf;                                  |
//|      int ev = EGP_MHO_DifferentialEvolution(lo,hi,3, 20,40,      |
//|                    0.7,0.9, 42, MyFitness, best, bf);            |
//|      PrintFormat("evals=%d best_fit=%.6f", ev, bf);             |
//|      // -> report ev as n_trials to the Python battery.          |
//|     }                                                           |
//+------------------------------------------------------------------+
#endif // EGP_MHO_OPTIMIZER_MQH
