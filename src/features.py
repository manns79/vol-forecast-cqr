# src/features.py
from __future__ import annotations

import numpy as np
import pandas as pd


def _parkinson_vol(high: pd.Series, low: pd.Series) -> pd.Series:
    # Parkinson variance estimator: (1/(4 ln 2)) * (ln(H/L))^2
    hl = (high / low).replace([np.inf, -np.inf], np.nan)
    log_hl = np.log(hl)
    var = (log_hl ** 2) / (4.0 * np.log(2.0))
    vol = np.sqrt(var)
    return vol


def make_supervised(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Input: raw OHLCV with a Date column (or index).
    Output: X (DataFrame, DateIndex) and y (Series, aligned, next-day vol target).
    """
    df = df_raw.copy()

    # Ensure Date exists and is datetime
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    else:
        # If no Date column, try treating the index as Date
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()].sort_index()

    # Require OHLC columns
    needed = {"Open", "High", "Low", "Close"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing required columns for features: {sorted(missing)}")

    close = df["Close"]

    # Returns
    r1 = close.pct_change()
    r5 = close.pct_change(5)

    # Daily range / volatility proxy
    park_vol = _parkinson_vol(df["High"], df["Low"])

    # Target: next-day volatility proxy
    y = park_vol.shift(-1).rename("y_nextday_vol")

    X = pd.DataFrame(index=df.index)

    # Lagged returns
    for k in [1, 2, 5, 10]:
        X[f"ret_lag_{k}"] = r1.shift(k)

    # Rolling return stats
    for w in [5, 10, 21, 63]:
        X[f"ret_roll_mean_{w}"] = r1.rolling(w).mean()
        X[f"ret_roll_std_{w}"] = r1.rolling(w).std()

    # Volatility features
    for k in [0, 1, 2, 5]:
        X[f"park_vol_lag_{k}"] = park_vol.shift(k)

    for w in [5, 10, 21, 63]:
        X[f"park_vol_roll_mean_{w}"] = park_vol.rolling(w).mean()
        X[f"park_vol_roll_std_{w}"] = park_vol.rolling(w).std()

    # Range feature (log high/low)
    X["log_hl"] = np.log((df["High"] / df["Low"]).replace([np.inf, -np.inf], np.nan))

    # Volume (optional)
    if "Volume" in df.columns:
        X["log_volume"] = np.log(df["Volume"].replace(0, np.nan))

    # Day of week
    X["dow"] = X.index.dayofweek.astype(float)

    # Drop rows with any NA in X or y
    Z = X.join(y, how="inner")
    Z = Z.dropna(axis=0)

    y_out = Z["y_nextday_vol"].copy()
    X_out = Z.drop(columns=["y_nextday_vol"]).copy()

    return X_out, y_out
