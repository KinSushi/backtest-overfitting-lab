#!/usr/bin/env python3
"""EGP offline metaheuristic kernels.

No MT5 calls, no trading calls. Deterministic when seeded.
Designed for candidate generation and mathematical tests before any Strategy Tester run.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Sequence, Dict, Any
import math, random, statistics

Vector = List[float]
Objective = Callable[[Vector], float]


def clamp_vec(x: Vector, bounds: Sequence[Tuple[float, float]]) -> Vector:
    return [max(lo, min(hi, v)) for v, (lo, hi) in zip(x, bounds)]


def sphere(x: Vector) -> float:
    return sum(v*v for v in x)


def rastrigin(x: Vector) -> float:
    return 10*len(x)+sum(v*v-10*math.cos(2*math.pi*v) for v in x)


def random_vec(bounds, rng):
    return [rng.uniform(lo, hi) for lo, hi in bounds]


@dataclass
class LSHADEState:
    best: Vector
    best_score: float
    history: List[float]


def lshade_minimize(fn: Objective, bounds: Sequence[Tuple[float,float]], max_evals=1200, pop_size=None, seed=42) -> LSHADEState:
    """L-SHADE style DE/current-to-pbest/1 with archive and parameter memories."""
    rng=random.Random(seed); dim=len(bounds)
    n_init=pop_size or max(18, 6*dim); n_min=max(4, dim)
    H=6; M_F=[0.5]*H; M_CR=[0.5]*H; k_mem=0
    pop=[random_vec(bounds,rng) for _ in range(n_init)]
    fit=[fn(x) for x in pop]
    evals=len(pop); archive=[]; hist=[]
    while evals < max_evals and len(pop)>=n_min:
        n=len(pop); order=sorted(range(n), key=lambda i: fit[i])
        new_pop=[]; new_fit=[]; succ_F=[]; succ_CR=[]; deltas=[]
        for i,x in enumerate(pop):
            r=rng.randrange(H)
            F=max(0.05,min(1.0,rng.gauss(M_F[r],0.1)))
            CR=max(0.0,min(1.0,rng.gauss(M_CR[r],0.1)))
            p=max(2,int(math.ceil(rng.uniform(0.05,0.2)*n)))
            xp=pop[rng.choice(order[:p])]
            idx=list(range(n)); idx.remove(i)
            r1=rng.choice(idx)
            pool=pop+archive
            r2=rng.randrange(len(pool))
            xr1=pop[r1]; xr2=pool[r2]
            v=[x[j] + F*(xp[j]-x[j]) + F*(xr1[j]-xr2[j]) for j in range(dim)]
            v=clamp_vec(v,bounds)
            jrand=rng.randrange(dim)
            u=[v[j] if (rng.random()<CR or j==jrand) else x[j] for j in range(dim)]
            fu=fn(u); evals+=1
            if fu <= fit[i]:
                new_pop.append(u); new_fit.append(fu); archive.append(x[:])
                succ_F.append(F); succ_CR.append(CR); deltas.append(abs(fit[i]-fu)+1e-12)
            else:
                new_pop.append(x); new_fit.append(fit[i])
            if evals>=max_evals: break
        pop,fit=new_pop,new_fit
        if len(archive)>n_init:
            rng.shuffle(archive); archive=archive[:n_init]
        if succ_F:
            sw=sum(deltas); w=[d/sw for d in deltas]
            M_F[k_mem]=sum(wi*fi*fi for wi,fi in zip(w,succ_F))/max(1e-12,sum(wi*fi for wi,fi in zip(w,succ_F)))
            M_CR[k_mem]=sum(wi*ci for wi,ci in zip(w,succ_CR))
            k_mem=(k_mem+1)%H
        # linear population size reduction
        target=round(n_min + (n_init-n_min)*(1-evals/max_evals))
        if len(pop)>target:
            order=sorted(range(len(pop)), key=lambda i: fit[i])[:target]
            pop=[pop[i] for i in order]; fit=[fit[i] for i in order]
        best_i=min(range(len(pop)), key=lambda i: fit[i]); hist.append(fit[best_i])
    bi=min(range(len(pop)), key=lambda i: fit[i])
    return LSHADEState(pop[bi], fit[bi], hist)


@dataclass
class CMAESState:
    best: Vector
    best_score: float
    history: List[float]


def mat_vec(A, x): return [sum(A[i][j]*x[j] for j in range(len(x))) for i in range(len(A))]
def eye(n): return [[1.0 if i==j else 0.0 for j in range(n)] for i in range(n)]
def outer(a,b): return [[ai*bj for bj in b] for ai in a]
def addm(A,B): return [[A[i][j]+B[i][j] for j in range(len(A))] for i in range(len(A))]
def scalem(s,A): return [[s*A[i][j] for j in range(len(A))] for i in range(len(A))]

def chol_sample(C, rng):
    n=len(C); L=[[0.0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1):
            s=C[i][j]-sum(L[i][k]*L[j][k] for k in range(j))
            if i==j: L[i][j]=math.sqrt(max(s,1e-12))
            else: L[i][j]=s/max(L[j][j],1e-12)
    z=[rng.gauss(0,1) for _ in range(n)]
    return [sum(L[i][j]*z[j] for j in range(i+1)) for i in range(n)]

def cmaes_minimize(fn: Objective, bounds: Sequence[Tuple[float,float]], max_evals=900, seed=7) -> CMAESState:
    rng=random.Random(seed); n=len(bounds)
    from egp_mho_hybrid import inv_sqrt_C  # C^{-1/2} for a correct CSA (fixes the omission)
    lam=4+int(3*math.log(n+1)); mu=lam//2
    weights=[math.log(mu+0.5)-math.log(i+1) for i in range(mu)]; sw=sum(weights); weights=[w/sw for w in weights]
    m=[(lo+hi)/2 for lo,hi in bounds]; sigma=sum(hi-lo for lo,hi in bounds)/(len(bounds)*4)
    C=eye(n); pc=[0.0]*n; ps=[0.0]*n
    mueff=1/sum(w*w for w in weights); cc=(4+mueff/n)/(n+4+2*mueff/n); cs=(mueff+2)/(n+mueff+5)
    c1=2/((n+1.3)**2+mueff); cmu=min(1-c1,2*(mueff-2+1/mueff)/((n+2)**2+mueff)); ds=1+2*max(0,math.sqrt((mueff-1)/(n+1))-1)+cs
    evals=0; best=None; best_score=float('inf'); hist=[]; chi=math.sqrt(n)*(1-1/(4*n)+1/(21*n*n))
    while evals<max_evals:
        samples=[]
        for _ in range(lam):
            y=chol_sample(C,rng); x=clamp_vec([m[i]+sigma*y[i] for i in range(n)],bounds); samples.append((fn(x),x,y)); evals+=1
        samples.sort(key=lambda t:t[0])
        if samples[0][0]<best_score: best_score=samples[0][0]; best=samples[0][1]
        old=m[:]
        m=[sum(weights[i]*samples[i][1][j] for i in range(mu)) for j in range(n)]
        y_w=[(m[j]-old[j])/max(sigma,1e-12) for j in range(n)]
        Cinv=inv_sqrt_C(C)                                       # C^{-1/2}
        z_w=[sum(Cinv[j][k]*y_w[k] for k in range(n)) for j in range(n)]
        ps=[(1-cs)*ps[j]+math.sqrt(cs*(2-cs)*mueff)*z_w[j] for j in range(n)]
        normps=math.sqrt(sum(v*v for v in ps)); sigma*=math.exp((cs/ds)*(normps/chi-1))
        hsig=1.0 if normps/math.sqrt(1-(1-cs)**(2*max(1,evals/lam))) < (1.4+2/(n+1))*chi else 0.0
        pc=[(1-cc)*pc[j]+hsig*math.sqrt(cc*(2-cc)*mueff)*y_w[j] for j in range(n)]
        rank_mu=[[0.0]*n for _ in range(n)]
        for i in range(mu):
            y=samples[i][2]; rank_mu=addm(rank_mu, scalem(weights[i], outer(y,y)))
        C=addm(scalem(1-c1-cmu,C), addm(scalem(c1,outer(pc,pc)), scalem(cmu,rank_mu)))
        hist.append(best_score)
        if best_score<1e-10: break
    return CMAESState(best or m, best_score, hist)


# NSGA-II helpers
def dominates(a,b):
    return all(x<=y for x,y in zip(a,b)) and any(x<y for x,y in zip(a,b))

def fast_non_dominated_sort(objs):
    S=[[] for _ in objs]; n=[0]*len(objs); fronts=[[]]
    for p in range(len(objs)):
        for q in range(len(objs)):
            if dominates(objs[p],objs[q]): S[p].append(q)
            elif dominates(objs[q],objs[p]): n[p]+=1
        if n[p]==0: fronts[0].append(p)
    i=0
    while fronts[i]:
        nxt=[]
        for p in fronts[i]:
            for q in S[p]:
                n[q]-=1
                if n[q]==0: nxt.append(q)
        i+=1; fronts.append(nxt)
    return fronts[:-1]

def crowding_distance(front, objs):
    d={i:0.0 for i in front}; m=len(objs[0])
    for k in range(m):
        order=sorted(front,key=lambda i:objs[i][k]); d[order[0]]=d[order[-1]]=float('inf')
        lo,hi=objs[order[0]][k],objs[order[-1]][k]
        if hi==lo: continue
        for a,b,c in zip(order,order[1:],order[2:]): d[b]+= (objs[c][k]-objs[a][k])/(hi-lo)
    return d


def nsga2_sort(objs):
    fronts=fast_non_dominated_sort(objs); result=[]
    for f in fronts:
        cd=crowding_distance(f,objs); result.extend(sorted(f,key=lambda i:(fronts.index(f),-cd[i])))
    return result


@dataclass
class TPESample:
    x: Dict[str,Any]
    y: float

class SimpleTPE:
    def __init__(self, space: Dict[str,Any], gamma=0.25, seed=1):
        self.space=space; self.gamma=gamma; self.rng=random.Random(seed); self.samples=[]
    def random_candidate(self):
        out={}
        for k,v in self.space.items():
            if v['type']=='float': out[k]=self.rng.uniform(v['min'],v['max'])
            elif v['type']=='int': out[k]=self.rng.randint(v['min'],v['max'])
            elif v['type']=='cat': out[k]=self.rng.choice(v['choices'])
        return out
    def observe(self,x,y): self.samples.append(TPESample(dict(x),float(y)))
    def suggest(self,n_candidates=64):
        if len(self.samples)<8: return self.random_candidate()
        s=sorted(self.samples,key=lambda t:t.y); cut=max(1,int(len(s)*self.gamma)); good=s[:cut]; bad=s[cut:] or s[cut-1:]
        def score(x):
            val=0.0
            for k,sp in self.space.items():
                if sp['type'] in ['float','int']:
                    gv=[g.x[k] for g in good]; bv=[b.x[k] for b in bad]
                    sg=max(1e-6,statistics.pstdev(gv) if len(gv)>1 else (sp['max']-sp['min'])/6)
                    sb=max(1e-6,statistics.pstdev(bv) if len(bv)>1 else (sp['max']-sp['min'])/6)
                    mg=sum(gv)/len(gv); mb=sum(bv)/len(bv); xx=x[k]
                    lg=math.exp(-0.5*((xx-mg)/sg)**2)/sg; lb=math.exp(-0.5*((xx-mb)/sb)**2)/sb
                    val+=math.log(lg+1e-12)-math.log(lb+1e-12)
                else:
                    cg=sum(1 for g in good if g.x[k]==x[k])+1; cb=sum(1 for b in bad if b.x[k]==x[k])+1
                    val+=math.log(cg/(len(good)+len(sp['choices'])))-math.log(cb/(len(bad)+len(sp['choices'])))
            return val
        cand=[self.random_candidate() for _ in range(n_candidates)]
        return max(cand,key=score)


def tabu_vns_optimize(start: List[int], eval_fn: Callable[[List[int]],float], max_iter=200, tenure=7):
    best=start[:]; best_score=eval_fn(best); cur=start[:]; tabu=[]
    for it in range(max_iter):
        neigh=[]
        for i in range(len(cur)):
            y=cur[:]; y[i]=1-y[i]; neigh.append(y)
        candidates=[]
        for y in neigh:
            key=tuple(y); sc=eval_fn(y)
            if key not in tabu or sc<best_score: candidates.append((sc,y,key))
        if not candidates: break
        sc,y,key=min(candidates,key=lambda t:t[0]); cur=y; tabu.append(key); tabu=tabu[-tenure:]
        if sc<best_score: best_score=sc; best=y[:]
    return best,best_score


def successive_halving(configs, score_fn, min_budget=1, max_budget=9, eta=3):
    survivors=list(configs); budget=min_budget; history=[]
    while len(survivors)>1 and budget<=max_budget:
        scored=sorted([(score_fn(c,budget),c) for c in survivors], key=lambda t:t[0])
        keep=max(1,len(scored)//eta); survivors=[c for _,c in scored[:keep]]; history.append((budget,keep))
        budget*=eta
    return survivors,history

# --------------------------------------------------------------------------- #
# P3 : versions FIDELES (honnetete documentaire)
#  - tabu_vns_optimize above = SINGLE-NEIGHBORHOOD primitive (1-flip) + tabu.
#  - successive_halving above = ONE bracket of SHA.
# The two functions below genuinely carry the semantics of their name:
# several neighborhood structures (VNS) and several brackets (Hyperband).
# --------------------------------------------------------------------------- #
def _local_search_1flip(x, eval_fn, tabu, budget):
    """Best-improvement 1-flip descent (one neighborhood), bounded by 'budget' evals.
    Accepts only STRICT improvements and avoids tabu solutions.
    Returns (x, score, evals_used)."""
    cur = list(x); cur_s = eval_fn(cur); used = 1
    improved = True
    while improved and used < budget:
        improved = False; best_y = None; best_s = cur_s
        for i in range(len(cur)):
            if used >= budget:
                break
            y = list(cur); y[i] = 1 - y[i]; used += 1
            s = eval_fn(y)
            if s < best_s and tuple(y) not in tabu:
                best_s = s; best_y = y
        if best_y is not None:
            cur = best_y; cur_s = best_s; improved = True
    return cur, cur_s, used


def variable_neighborhood_search(start, eval_fn, k_max=3, max_evals=500, tenure=50, seed=0):
    """FAITHFUL VNS (minimization, binary vectors): SEVERAL neighborhood structures
    N_1..N_kmax = k-bit perturbations (k-flip). Standard scheme of the base VNS:
      1) shake: random draw in N_k (k bits flipped at random),
      2) best-improvement local search (1-flip) on the perturbed point,
      3) if the incumbent is improved -> move there and k returns to 1;
         otherwise widen the neighborhood k <- k+1 (cycle to 1 beyond k_max).
    A tabu list bounds the re-traversal. Unlike a 1-flip-only descent,
    neighborhoods k>1 allow escaping 1-flip local optima.
    Refs : Mladenovic & Hansen (1997) ; Hansen, Mladenovic et al., 'Variable Neighborhood
    Search', Handbook of Metaheuristics."""
    import random as _r
    rng = _r.Random(seed); n = len(start)
    best = list(start); best_s = eval_fn(best); evals = 1
    cur = list(best); cur_s = best_s
    tabu = [tuple(cur)]
    k = 1
    while evals < max_evals:
        kk = min(k, n)
        y = list(cur)
        for i in rng.sample(range(n), kk):  # shake in N_k (k-flip)
            y[i] = 1 - y[i]
        y, y_s, used = _local_search_1flip(y, eval_fn, set(tabu), max_evals - evals)
        evals += used
        if y_s < cur_s:
            cur = y; cur_s = y_s; k = 1
            if y_s < best_s:
                best = list(y); best_s = y_s
        else:
            k += 1
            if k > k_max:
                k = 1
        t = tuple(cur)
        if t not in tabu:
            tabu.append(t); tabu = tabu[-tenure:]
    return best, best_s


def hyperband(get_config, score_fn, max_resource=27, eta=3, seed=0):
    """FAITHFUL HYPERBAND (minimization): wraps SEVERAL Successive Halving brackets
    with varied trade-offs (n configs, r budget). With R = max_resource:
      s_max = floor(log_eta(R)),  B = (s_max + 1) * R.
    For s = s_max .. 0 :
      n = ceil( B/R * eta^s / (s+1) )  configurations initiales,
      r = R * eta^(-s)                 initial budget per configuration,
      then (s+1) halving rounds: at each round keep the best 1/eta at budget x eta.
    The best survivor of each bracket is re-evaluated at full budget R; we keep the
    meilleur global. get_config(rng) -> nouvelle config ; score_fn(config, budget) -> cout
    (smaller = better). Returns (best_config, best_score, brackets).
    Ref : Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar, JMLR 2018, Algorithm 1."""
    import math, random as _r
    rng = _r.Random(seed)
    R = max_resource
    s_max = int(math.floor(math.log(R, eta)))
    B = (s_max + 1) * R
    brackets = []; results = []
    for s in range(s_max, -1, -1):
        n = int(math.ceil(B / R * (eta ** s) / (s + 1)))
        r = R * (eta ** (-s))
        configs = [get_config(rng) for _ in range(n)]
        hist = []
        for i in range(s + 1):
            n_i = int(math.floor(n * (eta ** (-i))))
            r_i = r * (eta ** i)
            scored = sorted(((score_fn(c, r_i), c) for c in configs), key=lambda t: t[0])
            hist.append((len(configs), r_i))
            keep = max(1, int(math.floor(n_i / eta)))
            configs = [c for _, c in scored[:keep]]
        best_c = configs[0]
        results.append((score_fn(best_c, R), best_c))
        brackets.append({'s': s, 'rounds': hist})
    best = min(results, key=lambda t: t[0])
    return best[1], best[0], brackets


if __name__=='__main__':
    # tiny smoke
    print(lshade_minimize(sphere,[(-5,5)]*5,300,seed=1).best_score)
