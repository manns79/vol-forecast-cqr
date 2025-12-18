"""
Write a short research note (Markdown) from artifacts/results.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", default="artifacts/research_note.md")
    args = p.parse_args()

    results_path = Path(args.results)
    out_path = Path(args.out)

    with open(results_path, "r") as f:
        r = json.load(f)

    note = f"""# Calibrated Volatility Forecasting for SPY (Quantile Regression + Conformal)

**Goal.** Forecast next-day volatility (daily Parkinson proxy computed from High/Low) and produce *calibrated* predictive intervals.

## Data and target
- Daily OHLC for SPY
- Volatility proxy: Parkinson estimator
- Target: next-day volatility proxy (shifted by 1 trading day)

## Method
- Quantile regression via `HistGradientBoostingRegressor(loss="quantile")` for multiple quantiles.
- Split by time into train / calibration / test (no leakage):
  - Train: all data before {r["cal_start"]}
  - Calibration: [{r["cal_start"]}, {r["test_start"]})
  - Test: [{r["test_start"]}, {r["last_date"]}]
- Conformalized Quantile Regression (CQR) to achieve finite-sample calibrated intervals:
  - For each nominal interval, compute nonconformity scores on calibration set
  - Widen predicted intervals on the test set by a single learned adjustment `s*`

## Results (test set)
- n_train = {r["n_train"]:,}, n_cal = {r["n_cal"]:,}, n_test = {r["n_test"]:,}
- Pinball loss (q=0.50): {r["pinball_q50"]:.6f}
- Pinball loss (q=0.10): {r["pinball_q10"]:.6f}
- Pinball loss (q=0.90): {r["pinball_q90"]:.6f}

### Calibrated interval performance
| Interval | Empirical coverage | Avg width | Conformal s* |
|---|---:|---:|---:|
| 50% | {r["coverage_50"]:.3f} | {r["width_50"]:.6f} | {r["s_star_50"]:.6f} |
| 80% | {r["coverage_80"]:.3f} | {r["width_80"]:.6f} | {r["s_star_80"]:.6f} |
| 95% | {r["coverage_95"]:.3f} | {r["width_95"]:.6f} | {r["s_star_95"]:.6f} |

## Artifacts
- `predictions.csv`: test-set y and calibrated intervals
- `fig_interval.png`: interval plot over the last test window
- `fig_coverage.png`: nominal vs empirical coverage

## Next steps
- Multi-asset panel and cross-asset generalization (SPY/QQQ/IWM/TLT)
- Strong baselines (GARCH, HAR-RV with richer realized-vol features)
- Regime-aware features and stress-period analysis
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(note, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
