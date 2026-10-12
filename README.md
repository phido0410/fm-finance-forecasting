# VN30 Foundation-Model Forecasting with LEAF

> **Why do global time-series foundation models fail on Vietnamese equities — and can a lightweight, regime-aware router fix it?**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![TimesFM](https://img.shields.io/badge/Google-TimesFM-4285F4)
![Chronos-2](https://img.shields.io/badge/Amazon-Chronos--2-FF9900)
![Moirai-2](https://img.shields.io/badge/Salesforce-Moirai--2-00A1E0)

Team research project benchmarking **3 frozen time-series foundation models** (TimesFM, Chronos-2, Moirai-2) against classical baselines (ARIMA, LSTM) for **1-day-ahead log-return forecasting** on 15 VN30 stocks, and proposing **LEAF** (*Lightweight Regime-Aware Ensemble Adapter*) — a ~100K-parameter router that combines the frozen models per sample.

> 📌 This is a fork of the team repository [`namhaing/fm-finance-forecasting`](https://github.com/namhaing/fm-finance-forecasting). See [My contributions](#-my-contributions) below.

---

## 👤 My contributions

**Đỗ Ngọc Phi** — research & implementation of the LSTM baseline, data collection, and foundation-model experiments (TimesFM, Chronos-2).

During the project I developed my parts locally and handed them to the repo maintainer, who committed them; the commits therefore carry his name. The original work is preserved in the git history:

| Area | What I did | Original commit | Final script |
|---|---|---|---|
| **LSTM baseline** | Researched and built a multivariate LSTM baseline for daily-return forecasting | [`39d7313`](https://github.com/namhaing/fm-finance-forecasting/commit/39d7313e76d7cb28c3909ff218267f19ab219214) | [`scripts/lstm_baseline_v3.py`](scripts/lstm_baseline_v3.py) |
| **Data collection** | Crawled daily OHLCV for VN30 constituents and indices from the VNDirect API | [`f0eb35e`](https://github.com/namhaing/fm-finance-forecasting/commit/f0eb35e7a36d7ac024ef55335b54414012c1913c), [`e647296`](https://github.com/namhaing/fm-finance-forecasting/commit/e64729648eb9c92c1a86d5c73a2e53ad2c9fee5e) | [`src/data/collecting/collect_price.py`](src/data/collecting/collect_price.py) |
| **TimesFM** | Zero-shot forecasting with Google TimesFM on VN30 | [`9e71de0`](https://github.com/namhaing/fm-finance-forecasting/commit/9e71de00602af75a977152ccd036f49eacf28966) | [`scripts/timesfm_zeroshot_v3.py`](scripts/timesfm_zeroshot_v3.py) |
| **Chronos-2** | Zero-shot vs. fine-tuning experiments with Amazon Chronos-2 | [`79bd2f7`](https://github.com/namhaing/fm-finance-forecasting/commit/79bd2f75ba765863d7bf99a27bc0745baf970578) | [`scripts/chronos2_zeroshot_v3.py`](scripts/chronos2_zeroshot_v3.py) |

The macro-data pipeline, feature selection, Moirai-2 and the LEAF framework were developed by teammates.

---

## 🧩 Problem

Time-series foundation models are pre-trained on large global corpora and are claimed to forecast "zero-shot" on unseen series. We test that claim on an **emerging market** (Vietnam's VN30), where daily returns are noisy and partly driven by macro factors (FX, rates, foreign flows) rather than price history alone.

**Setup**

| Item | Value |
|---|---|
| Universe | 15 VN30 stocks (banking, real estate, consumer, energy, tech, …) |
| Target | `y_return_1d = ln(Close[t+1] / Close[t])` |
| Split (time-based) | Train 2013–2021 · Val 2022–2023 · **Test 2024-01-03 → 2025-12-30** |
| Test size | 15 tickers × 416 trading days = **6,240 rows**, identical for every model |
| Metrics | RMSE, MAE, **IC** (Information Coefficient — rank correlation of forecast vs. realised return), **DA** (directional accuracy) |

---

## 📊 Results (test set)

| Model | Type | RMSE ↓ | MAE ↓ | IC ↑ | DA ↑ |
|---|---|---|---|---|---|
| **LEAF (5-seed ensemble)** | Router over 3 frozen FMs | **0.02107** | **0.01349** | **+0.0429** | 0.4731 |
| ARIMA(1,0,1) | Classical | 0.02130 | 0.01370 | −0.0229 | **0.4739** |
| LSTM-MV | Deep learning | 0.02128 | 0.01375 | −0.0028 | 0.4567 |
| Chronos-2 | FM, zero-shot | 0.02175 | 0.01377 | −0.0557 | 0.4518 |
| TimesFM 1.0-200M | FM, zero-shot | 0.02287 | 0.01399 | −0.0566 | 0.4614 |
| Equal-weight (3 FM) | Naive ensemble | 0.02271 | 0.01411 | −0.0698 | 0.4558 |
| Moirai-2 | FM, zero-shot | 0.02469 | 0.01520 | −0.0707 | 0.4563 |

![Model comparison](outputs/figures/fig1_model_comparison.png)

**Key findings**

1. **All three foundation models have negative IC zero-shot** on VN30 — they do not transfer out-of-the-box to this market.
2. **Naively averaging them makes it worse** (equal-weight IC −0.070).
3. **LEAF is the only model with positive IC** and has the lowest RMSE/MAE, despite being built only from negative-IC components — routing *when* to trust *which* model matters more than any single model.
4. **Directional accuracy stays below 50% for every model** (ARIMA is marginally best). Daily returns remain hard to predict; the signal LEAF recovers is real but weak.

---

## 🍃 LEAF in brief

```
context window (60d) ──► Regime encoder ──┐
                                          ├──► Dynamic router ──► weights w₁,w₂,w₃
17 aux features ──────────────────────────┘          │
(FM disagreement, rolling FM error,                  ▼
 ticker beta/volatility, …)            ŷ = Σ wᵢ · calibᵢ(FMᵢ forecast)
                                       (per-FM affine calibration)
```

- **Trained on FM validation predictions only** → genuinely out-of-sample for the frozen models.
- **Entropy-floor regularisation** prevents the router collapsing onto one model.
- **5-seed ensemble** removes seed noise (single-seed IC σ ≈ 0.028).

**Ablation** (`scripts/leaf_ablation.py`)

| Round | Change | IC |
|---|---|---|
| R0 | Baseline router (collapses) | −0.0580 |
| R1 | + entropy floor | −0.0639 |
| R2 | + tracking aux features | −0.0622 |
| R3 | + interaction / ticker features + calibration | +0.0092 |
| R4 | + 5-seed ensemble | **+0.0429** |

| Regime weights | IC per regime |
|---|---|
| ![Regime weights](outputs/figures/fig3_regime_weights.png) | ![IC per regime](outputs/figures/fig5_ic_per_regime.png) |

---

## 🔬 Methodology

- **No look-ahead leakage** — rolling features use `.shift(1)`; scalers fit without test data; FMs are zero-shot; leakage audit in `scripts/utils/check_and_fix_leakage.py`.
- **Same evaluation window for every model** — 416 days × 15 tickers.
- **Reproducible** — `set_seed()` covers torch / numpy / random / cuDNN; seeds `[42, 123, 2024, 7, 1337]`.
- **Feature selection** — 51 candidate features (12 stock + 39 macro) reduced to 8 by 3-step voting (low variance → correlation → mutual information).

---

## 🗂 Repository layout

```
├── scripts/                     main pipeline (run in order)
│   ├── preprocess_macro.py          clean macro series
│   ├── create_panel_data.py         stock + macro → long panel
│   ├── create_targets_and_split.py  log-return target, time split
│   ├── feature_selection_v3_final.py
│   ├── arima_baseline_v3.py         ARIMA baseline
│   ├── lstm_baseline_v3.py          LSTM baseline
│   ├── timesfm_zeroshot_v3.py       TimesFM zero-shot
│   ├── chronos2_zeroshot_v3.py      Chronos-2 zero-shot (+ past covariates)
│   ├── moirai2_from_notebook.py     Moirai-2 zero-shot
│   ├── leaf_ensemble.py             LEAF (headline result)
│   ├── leaf_ablation.py             4-round ablation
│   ├── visualize_results.py
│   └── utils/                       leakage / validation checks
├── src/data/collecting/         raw-data fetchers (VNDirect, ETF flows)
├── outputs/figures/             result figures
├── docs/                        reports & design notes
└── data/                        not tracked (see below)
```

---

## 🚀 Quick start

```bash
pip install -r requirements.txt

# Foundation models (heavy, install separately; GPU recommended)
pip install timesfm
pip install uni2ts gluonts
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

The processed panel (~300 MB) is not stored in git. Regenerate it with `src/data/collecting/*.py`, then:

```bash
# 1. Data
python scripts/preprocess_macro.py
python scripts/create_panel_data.py
python scripts/create_targets_and_split.py

# 2. Baselines
python scripts/arima_baseline_v3.py
python scripts/lstm_baseline_v3.py

# 3. Foundation models (zero-shot)
python scripts/timesfm_zeroshot_v3.py
python scripts/chronos2_zeroshot_v3.py
python scripts/moirai2_from_notebook.py

# 4. LEAF + ablation + figures
python scripts/leaf_ensemble.py   # → RMSE=0.02107 MAE=0.01349 IC=+0.0429 DA=0.4731
python scripts/leaf_ablation.py
python scripts/visualize_results.py
```

Full pipeline ≈ 30 min on one GPU (LEAF alone ≈ 3–5 min).

---

## ⚠️ Limitations & future work

- **Weak signal** — IC +0.043 is positive but small, and DA < 50% for all models; results are not a trading strategy.
- **Ablation is reported on the test set**, so the design rounds may carry some selection bias. Re-running the ablation on a held-out validation window is planned.
- **No walk-forward re-training** of LEAF yet; the router is trained once on 2022–2023.
- **Not yet done**: adding ARIMA/LSTM as extra experts in the router, hyper-parameter search.

---

## 👥 Team

Research project on *"Foundation Models for Financial Time Series"*. Original repository maintained by [@namhaing](https://github.com/namhaing).

*For research and educational purposes only — not investment advice.*
