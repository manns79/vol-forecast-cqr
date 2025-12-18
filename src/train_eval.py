"""
Train quantile regressors and evaluate conformalized prediction intervals.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump

from .features import make_supervised
from .metrics import interval_coverage, interval_width, pinball_loss
from .models import QuantileModelSpec, fit_quantile_model, cqr_calibrate, apply_cqr
from .plots import plot_intervals, plot_coverage


@dataclass(frozen=True)
class RunArgs:
    data: Path
    out_dir: Path
    test_years: int
    cal_years: int
    ticker: str | None = None


def year_split(dates: pd.Series, test_years: int, cal_years: int):
    """
    Split by calendar time:
    - test: last `test_years` years
    - cal: preceding `cal_years` years
    - train: everything before
    """
    dt = pd.to_datetime(dates)
    last_date = dt.max()
    test_start = last_date - pd.DateOffset(years=test_years)
    cal_start = test_start - pd.DateOffset(years=cal_years)

    idx_test = dt >= test_start
    idx_cal = (dt >= cal_start) & (dt < test_start)
    idx_train = dt < cal_start
    return idx_train.values, idx_cal.values, idx_test.values, cal_start, test_start, last_date


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="artifacts")
    p.add_argument("--test_years", type=int, default=2)
    p.add_argument("--cal_years", type=int, default=1)
    args_ns = p.parse_args()
    args = RunArgs(data=Path(args_ns.data), out_dir=Path(args_ns.out_dir), test_years=args_ns.test_years, cal_years=args_ns.cal_years)

    df_raw = pd.read_csv(args.data)
    # yfinance uses "Adj Close"; normalize
    if "Adj Close" in df_raw.columns and "Adj_Close" not in df_raw.columns:
        df_raw = df_raw.rename(columns={"Adj Close": "Adj_Close"})
    if "Close" not in df_raw.columns:
        raise ValueError("Expected a 'Close' column in the data.")
    if "High" not in df_raw.columns or "Low" not in df_raw.columns:
        raise ValueError("Expected 'High' and 'Low' columns in the data for Parkinson vol.")

    X_df, y = make_supervised(df_raw)
    dates = X_df["Date"].copy()
    X = X_df.drop(columns=["Date"]).to_numpy(dtype=float)
    y = y.to_numpy(dtype=float)

    idx_train, idx_cal, idx_test, cal_start, test_start, last_date = year_split(dates, args.test_years, args.cal_years)

    X_train, y_train = X[idx_train], y[idx_train]
    X_cal, y_cal = X[idx_cal], y[idx_cal]
    X_test, y_test = X[idx_test], y[idx_test]
    dates_test = pd.to_datetime(dates[idx_test]).reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Quantiles for 50%, 80%, 95% intervals
    quantiles = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]
    models = {}
    for q in quantiles:
        spec = QuantileModelSpec(quantile=q)
        m = fit_quantile_model(X_train, y_train, spec)
        models[q] = m
        dump(m, args.out_dir / f"model_q{q:.3f}.joblib")

    def pred(q: float, Xp: np.ndarray) -> np.ndarray:
        return models[q].predict(Xp)

    # Base (uncalibrated) quantile predictions
    preds_cal = {q: pred(q, X_cal) for q in quantiles}
    preds_test = {q: pred(q, X_test) for q in quantiles}

    # Build nominal intervals and conformalize them on calibration set
    intervals = {
        "50": (0.25, 0.75, 0.50),
        "80": (0.10, 0.90, 0.20),
        "95": (0.025, 0.975, 0.05),
    }

    out_pred = pd.DataFrame({"Date": dates_test, "y": y_test})

    results = {
        "n_train": int(len(y_train)),
        "n_cal": int(len(y_cal)),
        "n_test": int(len(y_test)),
        "cal_start": str(pd.to_datetime(cal_start).date()),
        "test_start": str(pd.to_datetime(test_start).date()),
        "last_date": str(pd.to_datetime(last_date).date()),
    }

    # Pinball loss on test for a few key quantiles
    results["pinball_q50"] = pinball_loss(y_test, preds_test[0.50], 0.50)
    results["pinball_q10"] = pinball_loss(y_test, preds_test[0.10], 0.10)
    results["pinball_q90"] = pinball_loss(y_test, preds_test[0.90], 0.90)

    for name, (q_lo, q_hi, alpha) in intervals.items():
        lo_cal, hi_cal = preds_cal[q_lo], preds_cal[q_hi]
        s_star = cqr_calibrate(y_cal, lo_cal, hi_cal, alpha=alpha)

        lo_test, hi_test = preds_test[q_lo], preds_test[q_hi]
        lo_c, hi_c = apply_cqr(lo_test, hi_test, s_star)

        out_pred[f"lo_{name}"] = lo_c
        out_pred[f"hi_{name}"] = hi_c

        results[f"s_star_{name}"] = float(s_star)
        results[f"coverage_{name}"] = interval_coverage(y_test, lo_c, hi_c)
        results[f"width_{name}"] = interval_width(lo_c, hi_c)

    out_pred.to_csv(args.out_dir / "predictions.csv", index=False)
    with open(args.out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Figures
    plot_intervals(out_pred, args.out_dir / "fig_interval.png", n=min(250, len(out_pred)))
    plot_coverage(results, args.out_dir / "fig_coverage.png")

    print("Done.")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
