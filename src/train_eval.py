# src/train_eval.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.features import make_supervised
from src.models import QuantileSpec, fit_quantile_model, pinball_loss


def _split_by_years(index: pd.DatetimeIndex, test_years: int, cal_years: int):
    """
    Returns boolean masks for train/cal/test based on calendar time.
    test = last `test_years` years
    cal  = the `cal_years` years immediately before test
    train = everything before cal
    """
    end = index.max()
    test_start = end - pd.DateOffset(years=test_years)
    cal_start = test_start - pd.DateOffset(years=cal_years)

    is_test = index >= test_start
    is_cal = (index >= cal_start) & (index < test_start)
    is_train = index < cal_start

    return is_train, is_cal, is_test, cal_start, test_start, end


def _conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    """
    Conformal quantile with finite-sample correction:
      qhat = k-th order statistic where k = ceil((n+1)*(1-alpha))
    This yields conservative coverage under exchangeability.

    scores should be nonnegative.
    """
    scores = np.asarray(scores, dtype=float)
    scores = scores[~np.isnan(scores)]
    n = scores.size
    if n == 0:
        raise RuntimeError("Calibration scores are empty; cannot compute conformal adjustment.")
    scores = np.maximum(scores, 0.0)

    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)

    return float(np.sort(scores)[k - 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="artifacts")
    ap.add_argument("--test_years", type=int, default=2)
    ap.add_argument("--cal_years", type=int, default=1)
    ap.add_argument(
        "--plot_cov",
        type=float,
        default=0.80,
        help="Which nominal coverage to plot in fig_interval.png (e.g., 0.8).",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read CSV defensively; we standardize Date handling in make_supervised anyway.
    df_raw = pd.read_csv(args.data)
    if "Date" not in df_raw.columns:
        df_raw = pd.read_csv(args.data, index_col=0).reset_index()
        if "Date" not in df_raw.columns and "index" in df_raw.columns:
            df_raw = df_raw.rename(columns={"index": "Date"})

    X_df, y = make_supervised(df_raw)

    idx = X_df.index
    is_train, is_cal, is_test, cal_start, test_start, end = _split_by_years(idx, args.test_years, args.cal_years)

    X_train, y_train = X_df.loc[is_train], y.loc[is_train]
    X_cal, y_cal = X_df.loc[is_cal], y.loc[is_cal]
    X_test, y_test = X_df.loc[is_test], y.loc[is_test]

    if len(X_train) == 0 or len(X_cal) == 0 or len(X_test) == 0:
        raise RuntimeError(
            "One of the splits is empty.\n"
            f"Sizes: train={len(X_train)}, cal={len(X_cal)}, test={len(X_test)}\n"
            f"Index range: {idx.min().date()} .. {idx.max().date()}\n"
            f"cal_start={cal_start.date()}, test_start={test_start.date()}, end={end.date()}\n"
            "Try decreasing --test_years / --cal_years or check that Date parsed correctly."
        )

    # Quantile pairs matched to desired nominal coverages
    pairs: dict[float, tuple[float, float]] = {
        0.50: (0.25, 0.75),
        0.80: (0.10, 0.90),
        0.95: (0.025, 0.975),
    }

    # Optional median model (useful for plotting)
    m_med = fit_quantile_model(X_train, y_train, QuantileSpec(quantile=0.50))
    med_test = m_med.predict(X_test)

    metrics: dict[str, dict[str, float]] = {}
    intervals: dict[str, dict[str, float]] = {}
    pinball_by_q: dict[str, float] = {}

    # Store predictions into one dataframe indexed by test dates
    pred_df = pd.DataFrame({"y_true": y_test.values, "q0.5": med_test}, index=X_test.index)

    # Fit/predict/calibrate per coverage
    for cov, (q_lo, q_hi) in pairs.items():
        cov_key = f"{cov:.2f}"
        alpha = 1.0 - cov

        m_lo = fit_quantile_model(X_train, y_train, QuantileSpec(quantile=q_lo))
        m_hi = fit_quantile_model(X_train, y_train, QuantileSpec(quantile=q_hi))

        lo_cal = m_lo.predict(X_cal)
        hi_cal = m_hi.predict(X_cal)

        lo_test = m_lo.predict(X_test)
        hi_test = m_hi.predict(X_test)

        # CQR conformity scores: how much we need to expand the interval to include y
        # score_i = max(y_i - lo_i, hi_i - y_i)
        scores = np.maximum(y_cal.values - lo_cal, hi_cal - y_cal.values)

        qhat = _conformal_qhat(scores, alpha=alpha)

        lo_c = lo_test - qhat
        hi_c = hi_test + qhat

        emp_cov = float(np.mean((y_test.values >= lo_c) & (y_test.values <= hi_c)))
        avg_width = float(np.mean(hi_c - lo_c))

        metrics[cov_key] = {"empirical_coverage": emp_cov, "avg_width": avg_width}
        intervals[cov_key] = {"alpha": alpha, "qhat": qhat, "q_lo": q_lo, "q_hi": q_hi}

        # Pinball loss for the uncalibrated quantiles on test (informative diagnostics)
        pinball_by_q[f"q{q_lo:g}"] = pinball_loss(y_test.values, lo_test, q_lo)
        pinball_by_q[f"q{q_hi:g}"] = pinball_loss(y_test.values, hi_test, q_hi)

        # Save predictions
        pct = int(round(cov * 100))
        pred_df[f"base_lo_{pct}"] = lo_test
        pred_df[f"base_hi_{pct}"] = hi_test
        pred_df[f"cqr_lo_{pct}"] = lo_c
        pred_df[f"cqr_hi_{pct}"] = hi_c

    results = {
        "split": {
            "train_start": str(X_train.index.min().date()),
            "train_end": str(X_train.index.max().date()),
            "cal_start": str(X_cal.index.min().date()),
            "cal_end": str(X_cal.index.max().date()),
            "test_start": str(X_test.index.min().date()),
            "test_end": str(X_test.index.max().date()),
            "n_train": int(len(X_train)),
            "n_cal": int(len(X_cal)),
            "n_test": int(len(X_test)),
        },
        "metrics": metrics,
        "intervals": intervals,
        "pinball_loss_uncalibrated": pinball_by_q,
    }

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    pred_df.to_csv(out_dir / "predictions.csv")

    # ---------- Figures ----------
    # Figure 1: time plot with one chosen calibrated interval (default 80%)
    plot_cov = float(args.plot_cov)
    plot_pct = int(round(plot_cov * 100))
    lo_col = f"cqr_lo_{plot_pct}"
    hi_col = f"cqr_hi_{plot_pct}"

    if lo_col not in pred_df.columns or hi_col not in pred_df.columns:
        # fall back to 80% if user passed something odd
        plot_pct = 80
        lo_col = "cqr_lo_80"
        hi_col = "cqr_hi_80"

    tail_n = min(250, len(pred_df))
    tail = pred_df.iloc[-tail_n:]

    plt.figure()
    plt.plot(tail.index, tail["y_true"], label="Realized vol (target)")
    plt.plot(tail.index, tail["q0.5"], label="Median pred (q=0.5)")
    plt.fill_between(tail.index, tail[lo_col], tail[hi_col], alpha=0.3, label=f"CQR {plot_pct}%")
    plt.title(f"Next-day volatility forecast with CQR interval ({plot_pct}%)")
    plt.xlabel("Date")
    plt.ylabel("Volatility proxy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig_interval.png", dpi=160)
    plt.close()

    # Figure 2: nominal vs empirical coverage (for CQR)
    covs = np.array([float(k) for k in metrics.keys()], dtype=float)
    emps = np.array([metrics[f"{c:.2f}"]["empirical_coverage"] for c in covs], dtype=float)

    order = np.argsort(covs)
    covs, emps = covs[order], emps[order]

    plt.figure()
    plt.plot(covs, emps, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("CQR coverage: nominal vs empirical")
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical coverage")
    plt.tight_layout()
    plt.savefig(out_dir / "fig_coverage.png", dpi=160)
    plt.close()

    print("Wrote artifacts:")
    print(f" - {out_dir / 'results.json'}")
    print(f" - {out_dir / 'predictions.csv'}")
    print(f" - {out_dir / 'fig_interval.png'}")
    print(f" - {out_dir / 'fig_coverage.png'}")


if __name__ == "__main__":
    main()
