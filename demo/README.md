# EGP — End-to-end demo (synthetic data)

This demo proves, **without MetaTrader and reproducibly**, that every path of the validation pipeline
runs and returns a coherent verdict. It serves as a **living integration test** and a starting point
before wiring **real backtests**.

## Two demos

- **`run_real_demo.py`** — runs on the REAL CSVs an EA exported to MT5's `...\Terminal\Common\Files`,
  found **automatically** (no path to type). This is the reproducible demo **on real data**.
- **`run_demo.py`** — runs on **synthetic** data (no MetaTrader needed); proves all 4 pipeline paths.

### Real-data demo (auto-located)

```bash
python demo/run_real_demo.py                      # auto-detects Common\Files\EGP_deals.csv
python demo/run_real_demo.py --deals MyEA_deals.csv   # any other EA's export
python demo/run_real_demo.py --common-files "C:\...\Terminal\Common\Files"  # explicit folder
```

EA-agnostic: the pipeline only needs the MT5 deals format, so this works for **any** EA that exports
its deals (via `EGP_MHO_DealsExport.mqh`). It runs the deals→gate path on real data, and adds the
model path automatically when a **clean** features file is present (one row per Position).

### Synthetic demo

```bash
python demo/run_demo.py
```

Expected output (fixed seeds → reproducible):

```
[A] run_pipeline (prices->model): 124 signals -> gate=REJECT
[B] deals -> net=3344.68 | DSR=0.0 | risk_of_ruin=0.0 | REJECT
[C] optim -> 30 configs | PBO=0.33 | RC=0.41 | SPA=0.14 | REJECT
[D] model -> adds value: True
```

The full report is written to `demo/REPORT.md`.

### Full battery in one command (`run_full_battery.py`)

Runs **every pure-Python-runnable check** on a real MT5 deals export, in one consolidated report:

```bash
# auto-locate the EA's export (Common\Files\EGP_deals.csv):
python demo/run_full_battery.py

# or point at a file, and add the heavier tiers when you have their inputs:
python demo/run_full_battery.py path/to/EGP_deals.csv \
    --configs "%APPDATA%\MetaQuotes\Terminal\Common\Files\EGP_optpass" \
    --features "%APPDATA%\MetaQuotes\Terminal\Common\Files\EGP_features.csv"
```

What runs, by input tier:

| Tier | Needs | Checks |
|---|---|---|
| **Always** | the deals CSV | deflated gate (DSR, MinTRL), Monte-Carlo (risk of ruin), cost stress, **regime/session fragility** |
| `--configs FOLDER` | one deals CSV per config | PBO/CSCV + White's Reality Check + Hansen SPA over all configs |
| `--features FILE` | a CLEAN single-run features export | ML model validation (sized vs raw gate) |

> **Large optimizations.** Pure-Python PBO/CSCV + RC/SPA scale linearly with the config count, but
> thousands of configs take minutes (and the output is piped/buffered, so it can look stuck). Pass
> `--max-configs 300` to randomly subsample (seeded) to a tractable, statistically meaningful set —
> or re-optimize on the 2–4 structural parameters so the folder holds tens/hundreds, not thousands.

Three tiers are **not** run here because they need MT5 to re-evaluate each config/window:
walk-forward (`egp_wfo`), CPCV-at-optimization (`egp_cpcv_pipeline`), global sensitivity
(`egp_sobol`/`egp_bohb`). The script prints an explicit notice for each.

> The **regime/session** section answers *where* the edge lives. An edge concentrated in a single
> session (high HHI → `FRAGILE`) usually vanishes when the context changes — an honesty check the
> deals gate alone does not provide.

## The 4 exercised paths

| | Path | Input | What is validated |
|---|---|---|---|
| **A** | Prices → model | synthetic prices + signals | `run_pipeline`: triple-barrier → features → forest → purged CV → calibration → gate |
| **B** | Deals → risk | MT5-format deals CSV | DSR + Monte-Carlo (risk_of_ruin) + costs; **deposit row filtered** |
| **C** | Optimization → overfitting | N configs | **PBO + White's Reality Check + Hansen SPA**, `n_trials = number of configs` |
| **D** | Features → model | features + deals CSV | does the ML filter add value vs the raw signal? |

## Why (almost) everything prints REJECT

This is **intentional and healthy**. The data is near-noise; the battery is a **lie detector**:

- **B** rejects because the DSR, deflated for multiple trials, requires a Sharpe that noise does not reach.
- **C** rejects because a small edge among 30 configs does not survive multiple-testing correction (PBO/RC/SPA).
- **D** still detects that the weakly predictive feature `f0` adds value → `True`.

A pipeline that accepted noise would be **broken**. This one rejects it.

## Generated data (all synthetic, in-memory / local files)

- `demo/synth_deals.csv` — deals in MT5 export format, including a **deposit** row (`Type=2`) that must be filtered.
- `demo/synth_features.csv` — features in `EGP_MHO_FeaturesExport` format (`Position,Time,Side,f0..f3`), with the **same `Position` column** as the deals (join key).

## Moving to REAL data (next steps)

1. **Compile** the EA with the bundled `.mqh` (`MQL5/Experts/`), in particular:
   - `EGP_MHO_DealsExport.mqh` → writes `EGP_deals.csv` (wired in `OnDeinit`; written on **every** run —
     a single backtest, and the last pass of an optimization).
   - `EGP_MHO_FeaturesExample.mqh` → **filled example** (RSI, momentum, ATR, EMA distance…). It logs
     **once per position open** (on the last **closed** bar, no look-ahead) and **deletes the file at
     `OnInit`**, so one backtest yields one clean row per `Position`.
2. **Run ONE backtest** (not the optimizer — see the note) over any window. Prefer a **random
   one-year window**, not only the most recent year, to avoid recency bias. Use **"Every tick"** for
   the final run (open-price modeling inflates the profit factor). This writes a clean `EGP_deals.csv`
   **and** `EGP_features.csv`.
3. **Validate the whole battery** in one command:
   ```bash
   python demo/run_full_battery.py "PATH/EGP_deals.csv" --features "PATH/EGP_features.csv"
   ```
   (or the thinner `run_on_mt5_export.py "…deals.csv" --features "…features.csv"`).

> **Clean features = single backtest only.** During **optimization**, every pass reuses the same
> `Position` ids and writes the **shared** `Common\Files\EGP_features.csv`, so the file **accumulates
> across passes** and the features↔deals join becomes ambiguous. The ML/model path is therefore meant
> for a **single** backtest. The join is look-ahead-safe regardless: if several rows share a
> `Position`, it keeps the **earliest-timestamp** row (the entry), never a later bar.

The demo is the **dress rehearsal**: on real data, you replace the synthetic CSVs with the MT5 CSVs.

## Path C: the overfitting test (PBO/CSCV) on real configs

Path C is **not** produced by a single backtest. PBO/CSCV measures **selection overfitting**: across
many candidate configurations, when you pick the in-sample best, how often is it *below median*
out-of-sample? It therefore needs **one deals file per configuration**.

> **Critical — the overwrite trap.** The EA always writes to the **same** file
> (`Common\Files\EGP_deals.csv`). Each new backtest **overwrites** the previous one, and MT5's
> **optimizer truncates** the deals CSV at every pass. So you cannot run the optimizer and collect
> afterwards — you must run **N separate single backtests** and **copy the file out under a distinct
> name between each run**.

Workflow — a bundled helper does the copy for you. After **each** single backtest, run
[`demo/save_config_deals.ps1`](save_config_deals.ps1): it copies the EA's
`Common\Files\EGP_deals.csv` into a collection folder under an **auto-incremented** name
(`cfg_001.csv`, `cfg_002.csv`, …), so you can never overwrite a previous run.

```powershell
# for EACH configuration: load its .set, run ONE single backtest, then:
powershell -ExecutionPolicy Bypass -File demo\save_config_deals.ps1
#   -> copies EGP_deals.csv to .\pbo_configs\cfg_001.csv  (then 002, 003, ... automatically)

# options:
powershell -ExecutionPolicy Bypass -File demo\save_config_deals.ps1 -Dest H:\pbo_configs   # other folder
powershell -ExecutionPolicy Bypass -File demo\save_config_deals.ps1 -Name baseline         # explicit name

# when the folder holds >= 2 (ideally 10+) distinct CSVs:
python demo\collect_pbo.py ".\pbo_configs"          # one CSV per config -> PBO / RC / SPA / DSR
```

The script prints the running CSV count and the next command to run. `-ExecutionPolicy Bypass` avoids
the default block on local scripts; if Windows still refuses it (file marked "from the internet"), run
`Unblock-File demo\save_config_deals.ps1` once. Without the helper, the equivalent manual copy is just:

```powershell
Copy-Item "$env:APPDATA\MetaQuotes\Terminal\Common\Files\EGP_deals.csv" .\pbo_configs\cfg_001.csv
```

Notes:

- The **filename stem** becomes the **config name** in the report (`cfg_001`, `cfg_002`, …).
- The configurations must **differ** (different parameters): two identical runs carry no information
  for PBO. Keep the **same symbol and the same period** across configs so the matrix stays aligned.
- `--champion cfg_042` selects which config feeds the deals-gate section (default = best total net).
- `>= 2` is the technical minimum; PBO/CSCV is only **statistically meaningful with many** configs
  (realistically 10–20+).

### Capturing every pass automatically during optimization (local agents)

Running configs by hand is fine for 10–30 finalists. To let the battery see **every pass** of an
optimization (worst to best), have the EA dump one deals CSV per pass. The demo EA already does this:
`OnTester()` (one call per pass) invokes `EGP_ExportDealsForOptPass(tag)`, which — **only** when
`MQLInfoInteger(MQL_OPTIMIZATION)` is true — writes `Common\Files\EGP_optpass\cfg_<params>.csv`, one
distinct file per configuration. Distinct filenames mean parallel local agents never collide.

```powershell
# after the optimization finishes, point the battery at the folder:
python demo\collect_pbo.py "$env:APPDATA\MetaQuotes\Terminal\Common\Files\EGP_optpass"
```

What it tells you (and what it does **not**): the battery over all passes answers *“is the best-in-
sample config real, or the luckiest of N?”* — that is **PBO** (a verdict on the *selection*), plus a
**DSR deflated by the number of trials**. With many configs the deflation is brutal and almost
nothing “passes”; that is the **correct** skepticism, not a failure. Scanning more configs does **not**
surface a winner — out-of-sample / walk-forward consistency does. Use this to *disqualify* an
overfit winner, then confirm any survivor on a separate period.

Practical limits:

- **Local agents only.** Remote / MQL5-Cloud agents do **not** share the Common folder, so their
  files never come back. For a remote farm use the MT5 **frames** mechanism (`FrameAdd` in
  `OnTester` + `OnTesterPass`/`FrameNext` on the terminal) instead.
- **Optimize only a few structural parameters.** The filename encodes the varied inputs; optimizing
  a dozen parameters yields thousands of files with very long names. Vary the 2–4 that matter,
  freeze the rest (and edit the `tag` in `OnTester()` to match what you actually vary).
- **`OnTester` returning `0.0`** only affects the “Custom max” criterion. Keep a standard
  optimization criterion (or define your own in `OnTester`).
- **Every tick for finalists.** Open-price modeling inflates the profit factor; re-run the few
  survivors under “Every tick” before believing any number.

## Saving a clean run to a file

`run_real_demo.py`, `run_on_mt5_export.py` and `collect_pbo.py` **print** the report but do not write
a file (only `run_demo.py` writes `demo/REPORT.md`). To keep a clean transcript, redirect the output —
and run **one command at a time**, not a whole pasted block:

```powershell
python demo\run_real_demo.py 2>&1 | Tee-Object result_real.txt      # PowerShell: shows AND saves
```
```bash
python demo/run_real_demo.py 2>&1 | tee result_real.txt             # bash/zsh
```

## Caveats

- No MT5 compilation or backtest here: the `.mqh` are **not compiled** (compile them on your machine).
- Walk-forward (`egp_wfo`) and real optimization require MT5 to evaluate each config.
- Not financial advice.
