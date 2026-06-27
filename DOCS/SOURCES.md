# SOURCES — Consolidated bibliography (verified sources used in the code)

> Consolidated list of the sources **actually consulted** to verify formulas and APIs before
> coding. Each module cites its sources in its docstring; this file gathers them by topic. Main
> consultation: 2026-06. Reliability: **A** = primary/official source (vendor doc, reference book,
> peer-reviewed article); **B** = reliable secondary (university course, reference implementation);
> **C** = tertiary (technical blog, official forum) used only for confirmation.

---

## 1. Financial ML methodology (López de Prado)

| Source | Type | Rel. |
|---|---|---|
| M. López de Prado, *Advances in Financial Machine Learning*, Wiley (2018) — ch.3 (triple-barrier), 4 (sample weights, sequential bootstrap), 5 (fractional differentiation), 7 (purged k-fold, embargo), 8 (MDI/MDA), 10 (bet sizing), 12 (CPCV) | Reference book | A |
| M. López de Prado, *Machine Learning for Asset Managers*, Cambridge Univ. Press (2020) — ch.6 (Clustered Feature Importance) | Reference book | A |
| Man, X. & Chan, E.P. (2021), "The best way to select features: Comparing MDA, LIME and SHAP", *J. of Financial Data Science* (cMDA paper) | Article | A |
| mlfinlab (hudson-and-thames) — `feature_importance/importance.py`, `feature_clusters` | Reference impl. | B |
| GitHub emoen/Machine-Learning-for-Asset-Managers (MLAM snippets) | Reference impl. | B |

## 2. Trees, forests, importance (Gini / CART / MDI / OOB)

| Source | Type | Rel. |
|---|---|---|
| Breiman, L. (1996), *Bagging Predictors* — stat.berkeley.edu/~breiman/bagging.pdf | Article | A |
| Breiman, L., *Out-of-bag estimation* — stat.berkeley.edu/~breiman/OOBestimation.pdf | Article | A |
| Breiman, L. (2002), notes on Mean Decrease Gini | Article | A |
| scikit-learn, `DecisionTreeClassifier` doc (weighted impurity decrease) | Official doc | A |
| arxiv 2505.05402 (CART-ELC); 2507.07477, 1911.11901 (MDI/importance) | Preprint | B |
| Penn State STAT 508/857 (bagging, OOB); numberanalytics; scientistcafe; quantinsti | Course/blog | B/C |

## 3. Probability calibration (Platt / isotonic / Brier)

| Source | Type | Rel. |
|---|---|---|
| Platt, J. (2000), sigmoid calibration; Zadrozny & Elkan (2002); Niculescu-Mizil & Caruana (2005) | Articles | A |
| arxiv 2601.19944 (Classifier Calibration at Scale); 1710.08901 (PD calibration, MLE formula) | Preprint | B |
| Mantegna, R. (1999), "Hierarchical structure in financial markets" (correlation distance) | Article | A |
| arxiv 1511.05191 (isotonic ensemble/PAVA) | Preprint | B |
| KDnuggets, MATLAB i-vector (PAVA), Train in Data (Brier) | Blog/doc | C |

## 4. Stationarity & memory (ADF, fractional differentiation, AIC/BIC)

| Source | Type | Rel. |
|---|---|---|
| MacKinnon, J.G. (1994), asymptotic critical values of the ADF test | Article | A |
| statsmodels, `adfuller` doc (AIC/BIC autolag, same sample) | Official doc | A |
| arxiv 1506.02940 (BIC(p)=ln(SSR/T)+(p+1)*lnT/T), 1506.01984 | Preprint | B |
| Introduction to Econometrics with R section 14.6; Minitab; Real Statistics; Statalist | Course/doc | B/C |
| Box & Jenkins (ACF); Fisher-Pearson moments (skewness/kurtosis) | Standard ref. | A |

## 5. Robust evaluation (deflated Sharpe, PBO/CSCV, Monte-Carlo)

| Source | Type | Rel. |
|---|---|---|
| Bailey, D. & López de Prado, M. — Deflated Sharpe Ratio, Probability of Backtest Overfitting (CSCV) | Articles | A |
| López de Prado, *AFML* ch.11-12 (gate, CPCV) | Reference book | A |
| Cawley & Talbot (2010), "On Over-fitting in Model Selection" (nested CV) | Article | A |

## 6. MQL5 API (P0 bridge: deals export)

| Source (mql5.com/en/docs) | Type | Rel. |
|---|---|---|
| Trade Functions: `historyselect`, `historydealstotal`, `historydealgetticket`, `historydealgetdouble`, `historydealgetinteger`, `historydealgetstring` | Official doc | A |
| Constants: `tradingconstants/dealproperties` (ENUM_DEAL_PROPERTY_DOUBLE/INTEGER, ENUM_DEAL_ENTRY) | Official doc | A |
| File Functions: `FileOpen`/`FileWrite`/`FileSeek`/`FileSize` (FILE_CSV, FILE_COMMON) | Official doc | A |
| Official forum mql5.com/en/forum/473015 (net = profit + commission + swap; "select history first") | Official forum | C |

---

## Traceability reservations

- The sources above were **consulted via web search/reading** at coding time; the arxiv identifiers
  and official-doc URLs are **stable and verifiable**. A few blog entries (reliability C) were used
  only to **confirm** a formula already established by an A/B source.
- This file lists the sources **used in the code**; it is not an exhaustive literature review. The
  formulas themselves are validated by **golden vectors** in the tests.
- No value was invented: when a datum could not be verified (e.g. exact MacKinnon response surfaces),
  it is **replaced by a self-verifying method** (simulation) and flagged.
