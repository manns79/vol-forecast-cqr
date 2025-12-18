"""
Plot helpers.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_intervals(df_pred: pd.DataFrame, out: Path, n: int = 250) -> None:
    """
    Plot last n days: y, and 80% + 95% conformalized intervals.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    d = df_pred.tail(n).copy()
    x = pd.to_datetime(d["Date"])

    plt.figure()
    plt.plot(x, d["y"], label="Realized (proxy)")
    plt.fill_between(x, d["lo_80"], d["hi_80"], alpha=0.2, label="80% interval")
    plt.fill_between(x, d["lo_95"], d["hi_95"], alpha=0.15, label="95% interval")
    plt.title("Next-day volatility: predictions with calibrated intervals")
    plt.xlabel("Date")
    plt.ylabel("Volatility (Parkinson proxy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()


def plot_coverage(results: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = ["50%", "80%", "95%"]
    nominal = np.array([0.50, 0.80, 0.95])
    empirical = np.array([results["coverage_50"], results["coverage_80"], results["coverage_95"]])

    plt.figure()
    plt.plot(labels, nominal, marker="o", label="Nominal")
    plt.plot(labels, empirical, marker="o", label="Empirical")
    plt.ylim(0.0, 1.0)
    plt.title("Interval coverage (nominal vs empirical)")
    plt.xlabel("Interval level")
    plt.ylabel("Coverage")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
