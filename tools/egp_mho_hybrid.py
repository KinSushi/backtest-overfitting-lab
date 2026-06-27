#!/usr/bin/env python3
"""EGP MHO hybrid self-adaptive optimizer (offline, deterministic when seeded).

Adds what the baseline lacked:
  1. HYBRID CROSSOVER operators (intermediate, SBX, DE/current-to-pbest, ES self-adaptive).
  2. ADAPTIVE OPERATOR SELECTION (AOS): the algorithm learns which operator yields the most
     improvements (credit = improvement, roulette selection) -> "self-optimization"
     of the search's own controls.
  3. Parameter self-adaptation: sigma (ES) and F/CR (SHADE, Lehmer mean).
  4. CMA-ES with CORRECTED CSA: ps path using C^{-1/2} (fixes the M1 defect).
  5. Propose/observe loop WIRED to DOCS/PARAMETER_MAP.csv (fixes MHO-0):
     only the OPTIMIZE_CORE/OPTIMIZE_LATER dimensions vary; the protected ones are
     locked AND checked by a reverse-test.

No MT5/trading call. The real fitness (backtest metrics) plugs in via
`scores` provided to evolve_generation() (from MT5 reports, outside this module).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, List, Tuple, Sequence, Dict, Any, Optional
import re
import math, random, csv
from pathlib import Path

Vector = List[float]
Bounds = Sequence[Tuple[float, float]]
Objective = Callable[[Vector], float]

OPT_STATUSES = {'OPTIMIZE_CORE', 'OPTIMIZE_LATER'}


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def clamp_vec(x: Vector, bounds: Bounds) -> Vector:
    return [max(lo, min(hi, v)) for v, (lo, hi) in zip(x, bounds)]


# --------------------------------------------------------------------------- #
# HYBRID CROSSOVER OPERATORS (the formulas)
# --------------------------------------------------------------------------- #
def op_intermediate(x_a: Vector, x_b: Vector, rng: random.Random) -> Vector:
    """Arithmetic recombination: x' = a*x_a + (1-a)*x_b."""
    a = rng.random()
    return [a * u + (1 - a) * v for u, v in zip(x_a, x_b)]


def op_sbx(x_a: Vector, x_b: Vector, rng: random.Random, eta: float = 15.0) -> Vector:
    """Simulated Binary Crossover. beta from a polynomial distribution."""
    out = []
    for u, v in zip(x_a, x_b):
        r = rng.random()
        if r <= 0.5:
            beta = (2.0 * r) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - r))) ** (1.0 / (eta + 1.0))
        out.append(0.5 * ((1.0 + beta) * u + (1.0 - beta) * v))
    return out


def op_de_current_to_pbest(x_i: Vector, x_pbest: Vector, x_r1: Vector, x_r2: Vector, F: float) -> Vector:
    """DE/current-to-pbest/1 : v = x + F*(pbest - x) + F*(r1 - r2)."""
    return [xi + F * (xp - xi) + F * (a - b)
            for xi, xp, a, b in zip(x_i, x_pbest, x_r1, x_r2)]


def op_es_self_adaptive(x: Vector, sigma: Vector, rng: random.Random) -> Tuple[Vector, Vector]:
    """ES self-adaptive: sigma' = sigma*exp(tau'*N + tau*N_i); x' = x + sigma'*N_i."""
    n = len(x)
    tau_p = 1.0 / math.sqrt(2.0 * n)
    tau = 1.0 / math.sqrt(2.0 * math.sqrt(n))
    g = rng.gauss(0.0, 1.0)
    sigma2 = [max(1e-12, s * math.exp(tau_p * g + tau * rng.gauss(0.0, 1.0))) for s in sigma]
    child = [xi + si * rng.gauss(0.0, 1.0) for xi, si in zip(x, sigma2)]
    return child, sigma2


def binomial_crossover(parent: Vector, mutant: Vector, CR: float, rng: random.Random) -> Vector:
    dim = len(parent)
    jrand = rng.randrange(dim)
    return [mutant[j] if (rng.random() < CR or j == jrand) else parent[j] for j in range(dim)]


# --------------------------------------------------------------------------- #
# HYBRID ENGINE WITH AOS + ES sigma + SHADE F/CR
# --------------------------------------------------------------------------- #
@dataclass
class HybridResult:
    best: Vector
    best_score: float
    history: List[float]
    op_prob: Dict[str, float]
    sigma_mean_start: float
    sigma_mean_end: float


OPERATORS = ['de_pbest', 'sbx', 'intermediate', 'es_sigma']


def hybrid_minimize(fn: Objective, bounds: Bounds, max_evals: int = 1500,
                    pop_size: Optional[int] = None, seed: int = 42,
                    decay: float = 0.9) -> HybridResult:
    """Hybrid optimizer: portfolio of operators selected by AOS,
    self-adaptive ES step, SHADE F/CR memories, archive, LPSR."""
    rng = random.Random(seed)
    dim = len(bounds)
    NP = pop_size or max(16, 5 * dim)
    pop = [[rng.uniform(lo, hi) for lo, hi in bounds] for _ in range(NP)]
    sigma = [[(hi - lo) * 0.3 for lo, hi in bounds] for _ in range(NP)]
    fit = [fn(x) for x in pop]
    evals = NP
    archive: List[Vector] = []
    H = 6
    M_F = [0.5] * H
    M_CR = [0.5] * H
    kmem = 0
    # AOS: credit and usage per operator (ratio = mean reward)
    credit = {o: 1.0 for o in OPERATORS}
    usage = {o: 1.0 for o in OPERATORS}
    sigma_start = sum(sum(s) / dim for s in sigma) / NP
    hist: List[float] = []

    def select_operator() -> str:
        probs = {o: credit[o] / usage[o] for o in OPERATORS}
        tot = sum(probs.values())
        if tot <= 0:
            return rng.choice(OPERATORS)
        r = rng.random() * tot
        acc = 0.0
        for o in OPERATORS:
            acc += probs[o]
            if r <= acc:
                return o
        return OPERATORS[-1]

    while evals < max_evals:
        order = sorted(range(len(pop)), key=lambda i: fit[i])
        succ_F: List[float] = []
        succ_CR: List[float] = []
        deltas: List[float] = []
        p = max(2, int(0.15 * len(pop)))
        for i in range(len(pop)):
            op = select_operator()
            usage[op] += 1.0
            r = rng.randrange(H)
            F = max(0.05, min(1.0, rng.gauss(M_F[r], 0.1)))
            CR = max(0.0, min(1.0, rng.gauss(M_CR[r], 0.1)))
            xp = pop[rng.choice(order[:p])]
            idx = [j for j in range(len(pop)) if j != i]
            r1 = rng.choice(idx)
            pool = pop + archive
            r2 = rng.randrange(len(pool))
            child_sigma = sigma[i]

            if op == 'de_pbest':
                mutant = op_de_current_to_pbest(pop[i], xp, pop[r1], pool[r2], F)
                child = binomial_crossover(pop[i], mutant, CR, rng)
            elif op == 'sbx':
                child = op_sbx(pop[i], xp, rng)
            elif op == 'intermediate':
                child = op_intermediate(pop[i], pop[r1], rng)
            else:  # es_sigma
                child, child_sigma = op_es_self_adaptive(pop[i], sigma[i], rng)

            child = clamp_vec(child, bounds)
            fc = fn(child)
            evals += 1
            if fc <= fit[i]:
                improvement = abs(fit[i] - fc) + 1e-12
                archive.append(pop[i][:])
                pop[i] = child
                fit[i] = fc
                sigma[i] = child_sigma
                credit[op] += improvement          # AOS: reward = improvement
                succ_F.append(F)
                succ_CR.append(CR)
                deltas.append(improvement)
            if evals >= max_evals:
                break

        # AOS credit decay (progressive forgetting -> non-stationarity)
        for o in OPERATORS:
            credit[o] *= decay
            usage[o] = max(1.0, usage[o] * decay)
            credit[o] = max(credit[o], 1e-9)

        # bounded archive
        if len(archive) > NP:
            rng.shuffle(archive)
            archive = archive[:NP]

        # SHADE: update F memories (weighted Lehmer) and CR (weighted arithmetic)
        if succ_F:
            sw = sum(deltas)
            w = [d / sw for d in deltas]
            num = sum(wi * fi * fi for wi, fi in zip(w, succ_F))
            den = sum(wi * fi for wi, fi in zip(w, succ_F))
            M_F[kmem] = num / max(1e-12, den)
            M_CR[kmem] = sum(wi * ci for wi, ci in zip(w, succ_CR))
            kmem = (kmem + 1) % H

        # LPSR: linear population size reduction
        target = round(max(4, dim) + (NP - max(4, dim)) * (1 - evals / max_evals))
        if len(pop) > target:
            keep = sorted(range(len(pop)), key=lambda i: fit[i])[:target]
            pop = [pop[i] for i in keep]
            fit = [fit[i] for i in keep]
            sigma = [sigma[i] for i in keep]

        bi = min(range(len(pop)), key=lambda i: fit[i])
        hist.append(fit[bi])

    bi = min(range(len(pop)), key=lambda i: fit[i])
    probs = {o: credit[o] / usage[o] for o in OPERATORS}
    s = sum(probs.values()) or 1.0
    op_prob = {o: probs[o] / s for o in OPERATORS}
    sigma_end = sum(sum(sg) / dim for sg in sigma) / len(sigma)
    return HybridResult(pop[bi], fit[bi], hist, op_prob, sigma_start, sigma_end)


# --------------------------------------------------------------------------- #
# CMA-ES with CORRECTED CSA (C^{-1/2}) - fixes the M1 defect
# --------------------------------------------------------------------------- #
def jacobi_eig(A: List[List[float]], iters: int = 100, tol: float = 1e-12):
    """Eigendecomposition of a symmetric matrix (Jacobi). Returns (d, V)."""
    n = len(A)
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    M = [row[:] for row in A]
    for _ in range(iters):
        p, q, mx = 0, 1, 0.0
        for i in range(n):
            for j in range(i + 1, n):
                if abs(M[i][j]) > mx:
                    mx = abs(M[i][j]); p, q = i, j
        if mx < tol:
            break
        app, aqq, apq = M[p][p], M[q][q], M[p][q]
        phi = 0.5 * math.atan2(2 * apq, (aqq - app)) if abs(aqq - app) > 1e-30 else math.pi / 4
        c, s = math.cos(phi), math.sin(phi)
        for k in range(n):
            mkp, mkq = M[k][p], M[k][q]
            M[k][p] = c * mkp - s * mkq
            M[k][q] = s * mkp + c * mkq
        for k in range(n):
            mpk, mqk = M[p][k], M[q][k]
            M[p][k] = c * mpk - s * mqk
            M[q][k] = s * mpk + c * mqk
        for k in range(n):
            vkp, vkq = V[k][p], V[k][q]
            V[k][p] = c * vkp - s * vkq
            V[k][q] = s * vkp + c * vkq
    d = [M[i][i] for i in range(n)]
    return d, V


def inv_sqrt_C(C: List[List[float]]) -> List[List[float]]:
    """C^{-1/2} = V diag(1/sqrt(d)) V^T for symmetric positive-definite C."""
    d, V = jacobi_eig(C)
    d = [max(di, 1e-20) for di in d]
    n = len(C)
    inv = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            inv[i][j] = sum(V[i][k] * (1.0 / math.sqrt(d[k])) * V[j][k] for k in range(n))
    return inv


@dataclass
class CMAESCorrectState:
    best: Vector
    best_score: float
    history: List[float]


def cmaes_csa_correct(fn: Objective, bounds: Bounds, max_evals: int = 900, seed: int = 7) -> CMAESCorrectState:
    """CMA-ES with CORRECT conjugate step path: ps uses C^{-1/2}."""
    rng = random.Random(seed)
    n = len(bounds)
    lam = 4 + int(3 * math.log(n + 1))
    mu = lam // 2
    weights = [math.log(mu + 0.5) - math.log(i + 1) for i in range(mu)]
    sw = sum(weights)
    weights = [w / sw for w in weights]
    mueff = 1.0 / sum(w * w for w in weights)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    ds = 1 + 2 * max(0.0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chi = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

    m = [(lo + hi) / 2 for lo, hi in bounds]
    sigma = sum(hi - lo for lo, hi in bounds) / (n * 4)
    C = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    pc = [0.0] * n
    ps = [0.0] * n
    evals = 0
    best = None
    best_score = float('inf')
    hist: List[float] = []

    def chol_sample():
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1):
                s = C[i][j] - sum(L[i][k] * L[j][k] for k in range(j))
                if i == j:
                    L[i][j] = math.sqrt(max(s, 1e-12))
                else:
                    L[i][j] = s / max(L[j][j], 1e-12)
        z = [rng.gauss(0, 1) for _ in range(n)]
        return [sum(L[i][j] * z[j] for j in range(i + 1)) for i in range(n)]

    while evals < max_evals:
        samples = []
        for _ in range(lam):
            y = chol_sample()
            x = clamp_vec([m[i] + sigma * y[i] for i in range(n)], bounds)
            samples.append((fn(x), x, y))
            evals += 1
        samples.sort(key=lambda t: t[0])
        if samples[0][0] < best_score:
            best_score = samples[0][0]
            best = samples[0][1]
        old = m[:]
        m = [sum(weights[i] * samples[i][1][j] for i in range(mu)) for j in range(n)]
        y_w = [(m[j] - old[j]) / max(sigma, 1e-12) for j in range(n)]
        # --- FIX: ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * C^{-1/2} * y_w ---
        Cinv2 = inv_sqrt_C(C)
        Cinv2_yw = [sum(Cinv2[i][j] * y_w[j] for j in range(n)) for i in range(n)]
        ps = [(1 - cs) * ps[j] + math.sqrt(cs * (2 - cs) * mueff) * Cinv2_yw[j] for j in range(n)]
        normps = math.sqrt(sum(v * v for v in ps))
        sigma *= math.exp((cs / ds) * (normps / chi - 1))
        denom = math.sqrt(1 - (1 - cs) ** (2 * max(1, evals / lam)))
        hsig = 1.0 if normps / max(denom, 1e-12) < (1.4 + 2 / (n + 1)) * chi else 0.0
        pc = [(1 - cc) * pc[j] + hsig * math.sqrt(cc * (2 - cc) * mueff) * y_w[j] for j in range(n)]
        rank_mu = [[0.0] * n for _ in range(n)]
        for i in range(mu):
            y = samples[i][2]
            for a in range(n):
                for b in range(n):
                    rank_mu[a][b] += weights[i] * y[a] * y[b]
        for a in range(n):
            for b in range(n):
                C[a][b] = (1 - c1 - cmu) * C[a][b] + c1 * pc[a] * pc[b] + cmu * rank_mu[a][b]
        hist.append(best_score)
        if best_score < 1e-12:
            break
    return CMAESCorrectState(best or m, best_score, hist)


# --------------------------------------------------------------------------- #
# PER-PARAMETER WIRING (fixes MHO-0) + policy
# --------------------------------------------------------------------------- #
@dataclass
class OptDim:
    name: str
    lo: float
    hi: float
    is_int: bool
    kind: str = 'cont'        # 'cont' | 'int' | 'bool' | 'enum'
    choices: tuple = ()       # tokens EMITTED in the .set (enum's integer value if declared)
    choice_names: tuple = ()  # member labels (traceability; same length as choices)


def _parse_choice_spec(r: dict) -> List[Tuple[str, str]]:
    """Specification of an enum's choices, ONLY if declared (column 'choices'
    or notes 'choices=...'). Two formats per token, separated by '|':
      - 'NAME:VALUE'  -> (NAME, VALUE)   ; VALUE = underlying integer written in the .set
      - 'NAME'         -> (NAME, NAME)    ; backward-compatibility (emits the name)
    No invention: if nothing is declared -> []. The VALUE is what MT5 actually
    expects in the .set for an enum input (non-0-based enums, e.g. off_1=3
    or PRICE_CLOSE=1, require the exact value, not the index)."""
    raw = (r.get('choices') or '').strip()
    if not raw:
        m = re.search(r'choices\s*=\s*([^;]+)', r.get('notes') or '')
        raw = (m.group(1).strip() if m else '')
    spec: List[Tuple[str, str]] = []
    for tok in raw.split('|'):
        tok = tok.strip()
        if not tok:
            continue
        if ':' in tok:
            nm, val = tok.split(':', 1)
            spec.append((nm.strip(), val.strip()))
        else:
            spec.append((tok, tok))
    return spec


def _parse_choices(r: dict) -> List[str]:
    """EMITTED tokens of an enum (value if 'NAME:VALUE', otherwise the name). See _parse_choice_spec."""
    return [emitted for _, emitted in _parse_choice_spec(r)]


def load_opt_dims(root: str) -> List[OptDim]:
    """Optimizable dimensions: bounded numerics, booleans (true/false), and enums
    ONLY if their choices are declared (otherwise ignored, no invention)."""
    rows = list(csv.DictReader((Path(root) / 'DOCS' / 'PARAMETER_MAP.csv').open(encoding='utf-8')))
    dims: List[OptDim] = []
    for r in rows:
        if r.get('status') not in OPT_STATUSES:
            continue
        typ = r.get('type', '') or ''
        tl = typ.strip().lower()
        if tl == 'bool':
            dims.append(OptDim(r['name'], 0.0, 1.0, False, kind='bool'))
            continue
        spec = _parse_choice_spec(r)
        if spec:
            names = tuple(n for n, _ in spec)
            emitted = tuple(e for _, e in spec)
            dims.append(OptDim(r['name'], 0.0, float(len(spec)), False,
                               kind='enum', choices=emitted, choice_names=names))
            continue
        lo, hi = r.get('min', ''), r.get('max', '')
        if lo in (None, '') or hi in (None, ''):
            continue
        try:
            lo_f, hi_f = float(lo), float(hi)
        except ValueError:
            continue
        if hi_f <= lo_f:
            continue
        is_int = (tl == 'int') or typ.startswith('ENUM_TIMEFRAMES')
        dims.append(OptDim(r['name'], lo_f, hi_f, is_int, kind='int' if is_int else 'cont'))
    return dims


def decode(vec: Vector, dims: List[OptDim]) -> Dict[str, str]:
    """Vector -> .set values, optimizable dimensions only. Handles cont/int/bool/enum.
    bool: threshold 0.5; enum: index clamped to choices; int: rounded; cont: .6g."""
    out: Dict[str, str] = {}
    for v, d in zip(vec, dims):
        v = max(d.lo, min(d.hi, v))
        k = getattr(d, 'kind', 'cont')
        if k == 'bool':
            out[d.name] = 'true' if v >= 0.5 else 'false'
        elif k == 'enum' and d.choices:
            idx = max(0, min(len(d.choices) - 1, int(v)))
            out[d.name] = str(d.choices[idx])
        elif d.is_int:
            out[d.name] = str(int(round(v)))
        else:
            out[d.name] = f"{v:.6g}"
    return out


def encode(val, d: OptDim) -> float:
    """Inverse of decode: .set value -> real coordinate (history reconstruction
    without info loss on bool/enum)."""
    k = getattr(d, 'kind', 'cont')
    if k == 'bool':
        return 1.0 if str(val).strip().lower() == 'true' else 0.0
    if k == 'enum' and d.choices:
        sval = str(val)
        try:
            return float(d.choices.index(sval))
        except ValueError:
            pass
        names = getattr(d, 'choice_names', ()) or ()
        try:
            return float(names.index(sval))   # tolerates a .set carrying the member NAME
        except ValueError:
            return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return (d.lo + d.hi) / 2.0


def tpe_sample(bounds: Bounds, prev_vectors: List[Vector], prev_scores: List[float],
               n: int = 20, seed: int = 123, gamma: float = 0.25, n_candidates: int = 64) -> List[Vector]:
    """TPE: samples from l(x) (density of the good observations) and ranks by
    l(x)/g(x) (proxy for expected improvement). More sample-efficient than uniform
    sampling: proposals concentrate where the observed scores are good.
    We MINIMIZE: 'good' = lowest scores. Deterministic (seeded)."""
    rng = random.Random(seed)
    dim = len(bounds)
    pts = sorted(zip(prev_scores, prev_vectors), key=lambda t: t[0])   # ascending: best first
    N = len(pts)
    cut = max(1, int(math.ceil(gamma * N)))
    good = [v for _, v in pts[:cut]]
    bad = [v for _, v in pts[cut:]] or [pts[-1][1]]
    span = [hi - lo for lo, hi in bounds]
    b0 = max(0.08, 0.5 / math.sqrt(max(1, len(good))))               # band shrinking with the data
    bw = [max(1e-9, s * b0) for s in span]
    u = 1.0
    for s in span:
        u *= 1.0 / max(s, 1e-12)                                     # uniform density over the box

    def kde(x, sample):
        acc = u                                                     # prior: 1 uniform pseudo-observation
        for c in sample:
            d2 = 0.0
            for j in range(dim):
                d2 += ((x[j] - c[j]) / bw[j]) ** 2
            g = math.exp(-0.5 * d2)
            for j in range(dim):
                g /= (bw[j] * math.sqrt(2 * math.pi))
            acc += g
        return acc / (len(sample) + 1)

    cands: List[Vector] = []
    for _ in range(max(n_candidates, n)):
        if rng.random() < 1.0 / (len(good) + 1):                    # prior component -> uniform
            x = [lo + rng.random() * (hi - lo) for lo, hi in bounds]
        else:                                                        # 'good' component -> Gaussian around
            c = rng.choice(good)
            x = [min(hi, max(lo, rng.gauss(c[j], bw[j]))) for j, (lo, hi) in enumerate(bounds)]
        cands.append(x)
    cands.sort(key=lambda x: -(kde(x, good) / max(kde(x, bad), 1e-300)))
    return cands[:n]


def evolve_generation(dims: List[OptDim],
                      prev_vectors: Optional[List[Vector]],
                      prev_scores: Optional[List[float]],
                      family: str = 'TPE_LSHADE',
                      n: int = 20,
                      seed: int = 123,
                      inner_evals: int = 400) -> List[Vector]:
    """WIRED PROPOSE/OBSERVE loop (fixes MHO-0).

    - generation 0 (prev_*=None): space-filling population (Latin Hypercube).
    - generations 1+: a surrogate is fitted on (prev_vectors, prev_scores),
      then the `family` kernel proposes n new candidates WITHIN the OPT bounds.
      `prev_scores` comes from the real backtest metrics (outside the module).
    """
    rng = random.Random(seed)
    bounds: Bounds = [(d.lo, d.hi) for d in dims]
    dim = len(dims)

    if not prev_vectors or not prev_scores:
        # Latin Hypercube Sampling: deterministic space coverage (better than i.i.d.)
        cand: List[Vector] = []
        perms = [list(range(n)) for _ in range(dim)]
        for col in perms:
            rng.shuffle(col)
        for i in range(n):
            x = []
            for j, (lo, hi) in enumerate(bounds):
                u = (perms[j][i] + rng.random()) / n
                x.append(lo + u * (hi - lo))
            cand.append(x)
        return cand

    # Surrogate: weighted nearest neighbor (no external dependency), deterministic.
    pts = list(zip(prev_vectors, prev_scores))

    def surrogate(x: Vector) -> float:
        num = 0.0; den = 0.0
        for p, s in pts:
            d2 = sum((xi - pi) ** 2 for xi, pi in zip(x, p)) + 1e-9
            wgt = 1.0 / d2
            num += wgt * s; den += wgt
        return num / den

    fam = family.upper()
    if 'CMAES' in fam:
        # directly exploits the corrected engine on the surrogate
        out: List[Vector] = []
        for k in range(n):
            st = cmaes_csa_correct(surrogate, bounds, max_evals=inner_evals, seed=seed + k)
            out.append(st.best)
        return out
    if 'TPE' in fam:
        # TPE sampling from l(x) on the real observations (sample efficiency)
        return tpe_sample(bounds, prev_vectors, prev_scores, n=n, seed=seed)
    # other families (LSHADE/TABU/VNS...): hybrid engine on the surrogate
    out = []
    for k in range(n):
        res = hybrid_minimize(surrogate, bounds, max_evals=inner_evals, seed=seed + k)
        out.append(res.best)
    return out


def reverse_test_protected(root: str, produced: Dict[str, str]) -> List[str]:
    """Reverse-test: no produced key may touch a PROTECTED dimension.
    Returns the list of violations (empty = compliant)."""
    opt_names = {d.name for d in load_opt_dims(root)}
    return [k for k in produced.keys() if k not in opt_names]


from collections import namedtuple as _namedtuple
_LocalResult = _namedtuple('_LocalResult', 'best best_score evals')


def local_search(fn: Objective, x0: Vector, bounds: Bounds,
                 init_step: float = 0.1, iters: int = 40, shrink: float = 0.5):
    """Local pattern search (coordinate/pattern search) with decreasing step.
    Memetic hybrid: refines an (elite) individual proposed by the metaheuristic.
    Monotone: best_score <= fn(x0). Deterministic. Returns (best, best_score, evals)."""
    x = [min(hi, max(lo, v)) for v, (lo, hi) in zip(list(x0), bounds)]
    fx = fn(x)
    steps = [init_step * (hi - lo) for lo, hi in bounds]
    evals = 1
    for _ in range(iters):
        improved = False
        for i in range(len(x)):
            for s in (steps[i], -steps[i]):
                y = list(x)
                y[i] = min(bounds[i][1], max(bounds[i][0], x[i] + s))
                fy = fn(y); evals += 1
                if fy < fx:
                    x, fx, improved = y, fy, True
                    break
        if not improved:
            steps = [st * shrink for st in steps]
            if max(steps) < 1e-12:
                break
    return _LocalResult(best=x, best_score=fx, evals=evals)


def memetic_minimize(fn: Objective, bounds: Bounds, max_evals: int = 1500,
                     seed: int = 0, local_frac: float = 0.3) -> HybridResult:
    """Hybrid + local polishing of the incumbent (memetic). Splits the budget:
    (1-local_frac) to the global hybrid, local_frac to the final local search."""
    from dataclasses import replace
    g = max(1, int(max_evals * (1.0 - local_frac)))
    res = hybrid_minimize(fn, bounds, max_evals=g, seed=seed)
    loc = local_search(fn, res.best, bounds, iters=max(5, int(max_evals * local_frac / max(1, len(bounds)))))
    if loc.best_score <= res.best_score:
        return replace(res, best=loc.best, best_score=loc.best_score)
    return res


def polynomial_mutation(x: Vector, bounds: Bounds, rng: random.Random,
                        eta: float = 20.0, pm: Optional[float] = None) -> Vector:
    """Polynomial mutation (standard NSGA-II operator). pm = probability per variable."""
    n = len(x)
    pm = pm if pm is not None else 1.0 / max(1, n)
    out = list(x)
    for i, (lo, hi) in enumerate(bounds):
        if hi <= lo or rng.random() >= pm:
            continue
        xi = out[i]
        d1, d2 = (xi - lo) / (hi - lo), (hi - xi) / (hi - lo)
        u = rng.random(); mp = 1.0 / (eta + 1.0)
        if u < 0.5:
            val = 2.0 * u + (1.0 - 2.0 * u) * ((1.0 - d1) ** (eta + 1.0))
            dq = val ** mp - 1.0
        else:
            val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * ((1.0 - d2) ** (eta + 1.0))
            dq = 1.0 - val ** mp
        out[i] = min(hi, max(lo, xi + dq * (hi - lo)))
    return out


def pareto_front_size(objs: List[tuple]) -> int:
    """Size of the Pareto front (front 0) of a set of objective vectors."""
    if not objs:
        return 0
    from egp_meta_algos import fast_non_dominated_sort   # lazy import (no cycle)
    fronts = fast_non_dominated_sort([list(o) for o in objs])
    return len(fronts[0]) if fronts else 0


def evolve_generation_mo(dims: List[OptDim],
                         prev_vectors: Optional[List[Vector]],
                         prev_objectives: Optional[List[tuple]],
                         n: int = 20, seed: int = 123,
                         eta_c: float = 15.0, eta_m: float = 20.0) -> List[Vector]:
    """MULTI-OBJECTIVE PROPOSE/OBSERVE loop (NSGA-II) wired to the pipeline.

    - generation 0 (prev_*=None): Latin Hypercube (reuses evolve_generation).
    - generations 1+: non-dominated sort + crowding on the OBSERVED objective vectors,
      selection of the best parents, offspring via SBX + polynomial mutation,
      within the OPT bounds. The objectives come from the real backtest metrics.
    """
    rng = random.Random(seed)
    bounds: Bounds = [(d.lo, d.hi) for d in dims]
    if not prev_vectors or not prev_objectives:
        return evolve_generation(dims, None, None, n=n, seed=seed)   # LHS gen0
    from egp_meta_algos import nsga2_sort                            # lazy import
    ranked = nsga2_sort([list(o) for o in prev_objectives])         # indices: best -> worst
    k = max(2, len(ranked) // 2)
    parents = [prev_vectors[i] for i in ranked[:k]]
    children: List[Vector] = []
    while len(children) < n:
        a, b = rng.choice(parents), rng.choice(parents)
        child = op_sbx(a, b, rng, eta=eta_c)
        child = polynomial_mutation(child, bounds, rng, eta=eta_m)
        children.append(clamp_vec(child, bounds))
    return children[:n]


if __name__ == '__main__':
    print('hybrid sphere:', round(hybrid_minimize(lambda x: sum(v*v for v in x), [(-5, 5)] * 5, 600, seed=1).best_score, 6))
    print('cmaes(csa+) sphere:', round(cmaes_csa_correct(lambda x: sum(v*v for v in x), [(-3, 3)] * 4, 500, seed=2).best_score, 6))
