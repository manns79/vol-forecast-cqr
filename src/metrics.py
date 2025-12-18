"""
Metrics for quantile forecasts and prediction intervals.
"""
from __future__ import annotations

import numpy as np


def pinball_loss(y: np.ndarray, qhat: np.ndarray, q: float) -> float:
    """
    Mean pinball loss for quantile q.
    """
    y = np.asarray(y)
    qhat = np.asarray(qhat)
    u = y - qhat
    return float(np.mean(np.maximum(q * u, (q - 1.0) * u)))


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    y = np.asarray(y)
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    lo = np.asarray(lo)
    hi = np.asarray(hi)
    return float(np.mean(hi - lo))
