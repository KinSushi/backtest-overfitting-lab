# Contributing to EGP

Thanks for your interest. This repository is a quantitative-research tool; rigor and traceability
come first.

## Prerequisites

- Python ≥ 3.10. The pipeline uses **the standard library only** (no third-party dependency).
- `pytest` for the tests: `python -m pip install -r requirements-dev.txt`.

## Running the tests

```bash
python -m pytest -q          # 382 passed, 10 skipped
python demo/run_demo.py      # end-to-end demo
```

## Contribution rules

1. **No third-party dependency** in `tools/` (MQL5/embedded portability). Standard library only.
2. **Every logic change ships with a test.** The suite must stay green.
3. **Verify, do not invent**: every formula/API must be sourced (official MQL5 docs, AFML, papers).
4. No large cosmetic reformatting mixed with substantive changes.

## MQL5 policy (`.mqh` modules)

- **No MHO in `OnTick()`.**
- **No live mutation** of a running EA's parameters.
- **No automatic promotion**: any configuration coming out of optimization must **pass the battery**
  (DSR / PBO / Reality Check / SPA) **before** any use.

## Submitting

Open an issue (templates provided), then a pull request describing the what/why and the evidence
(tests, output). No PR without green tests.
