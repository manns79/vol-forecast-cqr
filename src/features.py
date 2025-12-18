"""
Feature engineering for next-day volatility forecasting.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def parkinson_vol(df: pd.DataFrame) -> pd.Series:
    """
    Parkinson daily volatility estimator using High/Low.
    sigma = sqrt( (1/(4 ln 2)) * (ln(H/L))^2 )
    """
    hl = np.log(df["High"].astype(float) / df["Low"].astype(float))
    return np.sqrt((hl ** 2) / (4.0 * np.log(2.0)))


def make_supervised(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X (features at time t) and y (target at time t): next-day volatility.
    """
    df = df.copy()
    df = df.sort_values("Date")
    # Ensure numeric
    for col in ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # log returns
    df["log_ret"] = np.log(df["Close"] / df["Close"].shift(1))
    df["abs_ret"] = df["log_ret"].abs()
    df["sq_ret"] = df["log_ret"] ** 2

    # volatility proxy today
    df["vol_pk"] = parkinson_vol(df)

    # rolling features
    for w in [5, 10, 21, 63]:
        df[f"vol_pk_mean_{w}"] = df["vol_pk"].rolling(w).mean()
        df[f"vol_pk_std_{w}"] = df["vol_pk"].rolling(w).std()
        df[f"abs_ret_mean_{w}"] = df["abs_ret"].rolling(w).mean()
        df[f"sq_ret_sum_{w}"] = df["sq_ret"].rolling(w).sum()

    # day-of-week
    dt = pd.to_datetime(df["Date"])
    df["dow"] = dt.dt.dayofweek.astype(int)  # 0=Mon

    # target: next-day volatility
    y = df["vol_pk"].shift(-1).rename("y_next_vol")

    feature_cols = [
        "vol_pk", "log_ret", "abs_ret", "sq_ret",
        "vol_pk_mean_5", "vol_pk_std_5", "abs_ret_mean_5", "sq_ret_sum_5",
        "vol_pk_mean_10", "vol_pk_std_10", "abs_ret_mean_10", "sq_ret_sum_10",
        "vol_pk_mean_21", "vol_pk_std_21", "abs_ret_mean_21", "sq_ret_sum_21",
        "vol_pk_mean_63", "vol_pk_std_63", "abs_ret_mean_63", "sq_ret_sum_63",
        "dow",
    ]
    X = df[["Date"] + feature_cols].copy()

    # one-hot encode dow in a stable way
    X = pd.get_dummies(X, columns=["dow"], prefix="dow", drop_first=False)

    # drop rows with NaNs (rolling, shift)
    mask = X.drop(columns=["Date"]).notna().all(axis=1) & y.notna()
    X = X.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    return X, y
