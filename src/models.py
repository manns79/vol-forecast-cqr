# src/models.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class QuantileSpec:
    quantile: float
    max_depth: int = 3
    max_iter: int = 300
    learning_rate: float = 0.05
    min_samples_leaf: int = 20
    l2_regularization: float = 0.0
    random_state: int = 42


def fit_quantile_model(X, y, spec: QuantileSpec) -> HistGradientBoostingRegressor:
    q = float(spec.quantile)
    if not (0.0 < q < 1.0):
        raise ValueError(f"Quantile must be in (0,1), got {q}")

    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=q,
        max_depth=spec.max_depth,
        max_iter=spec.max_iter,
        learning_rate=spec.learning_rate,
        min_samples_leaf=spec.min_samples_leaf,
        l2_regularization=spec.l2_regularization,
        random_state=spec.random_state,
    )
    model.fit(X, y)
    return model


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, q: float) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    e = y_true - y_pred
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))
