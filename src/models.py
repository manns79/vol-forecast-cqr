"""
Quantile regression models + conformalized quantile regression (CQR).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class QuantileModelSpec:
    quantile: float
    max_depth: int = 3
    learning_rate: float = 0.05
    max_iter: int = 400
    min_samples_leaf: int = 30
    random_state: int = 0


def fit_quantile_model(X: np.ndarray, y: np.ndarray, spec: QuantileModelSpec) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=spec.quantile,
        max_depth=spec.max_depth,
        learning_rate=spec.learning_rate,
        max_iter=spec.max_iter,
        min_samples_leaf=spec.min_samples_leaf,
        random_state=spec.random_state,
    )
    model.fit(X, y)
    return model


def cqr_calibrate(y_cal: np.ndarray, lo_cal: np.ndarray, hi_cal: np.ndarray, alpha: float) -> float:
    """
    Conformal score for CQR:
        s_i = max(lo_i - y_i, y_i - hi_i, 0)
    Then choose s* as the (1-alpha) empirical quantile.
    """
    y_cal = np.asarray(y_cal)
    lo_cal = np.asarray(lo_cal)
    hi_cal = np.asarray(hi_cal)

    scores = np.maximum.reduce([lo_cal - y_cal, y_cal - hi_cal, np.zeros_like(y_cal)])
    # "higher" interpolation for conservative finite-sample behavior
    q = np.quantile(scores, 1.0 - alpha, method="higher")
    return float(q)


def apply_cqr(lo: np.ndarray, hi: np.ndarray, s_star: float) -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lo) - s_star
    hi = np.asarray(hi) + s_star
    return lo, hi
