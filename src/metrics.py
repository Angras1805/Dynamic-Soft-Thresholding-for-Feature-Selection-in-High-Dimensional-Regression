from __future__ import annotations

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    d = y_true - y_pred
    return float(np.mean(d * d))


def count_nonzero(coef: np.ndarray, zero_tol: float) -> int:
    coef = np.asarray(coef, dtype=np.float64).ravel()
    return int(np.sum(np.abs(coef) > float(zero_tol)))

