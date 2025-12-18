# Calibrated Volatility Forecasting (Quantile Regression + Conformal Prediction)

A small, reproducible **time-series research project**: forecast next-day volatility for an equity index ETF (default: **SPY**) and produce **calibrated prediction intervals** using *Conformalized Quantile Regression (CQR)*.

## What this project demonstrates
- Time-series aware training / evaluation (walk-forward style split; no leakage)
- Quantile regression for predictive uncertainty
- **Finite-sample calibrated** prediction intervals via conformal prediction
- Clear, publishable evaluation: coverage, width, pinball loss, and plots

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt

# 1) Fetch daily OHLCV
python -m src.data_fetch --ticker SPY --start 2005-01-01 --out data/spy.csv

# 2) Train + evaluate + make figures
python -m src.train_eval --data data/spy.csv --out_dir artifacts --test_years 2 --cal_years 1

# 3) Write a short PDF-style report (markdown) you can paste into an application
python -m src.write_note --results artifacts/results.json --out artifacts/research_note.md
```

Outputs land in `artifacts/`:
- `results.json` (metrics + key numbers)
- `predictions.csv` (test-set predictions & intervals)
- `fig_interval.png`, `fig_coverage.png`

## Method (high level)
- Daily volatility proxy: **Parkinson** estimator from High/Low.
- Target: next-day volatility proxy.
- Models: `HistGradientBoostingRegressor(loss="quantile")` fit separately for each quantile.
- Conformal calibration (CQR): widen each nominal interval by an empirical nonconformity quantile computed on a calibration split.

## Notes
This is meant to be finished in a day. If you want a more “research associate” extension:
- multi-asset panel (SPY, QQQ, IWM, TLT)
- regime features (vol-of-vol, drawdowns)
- compare to GARCH(1,1) baseline
- use intraday realized volatility (if you have data)
