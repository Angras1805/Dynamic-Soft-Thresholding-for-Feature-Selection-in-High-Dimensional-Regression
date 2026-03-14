from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from sklearn.linear_model import Lasso, Ridge


@dataclass(frozen=True)
class FitResult:
    coef_: np.ndarray
    intercept_: float
    runtime_s: float


def fit_ridge(X, y: np.ndarray, alpha: float, *, random_seed: int) -> FitResult:
    t0 = perf_counter()
    model = Ridge(alpha=alpha, random_state=random_seed)
    model.fit(X, y)
    rt = perf_counter() - t0
    return FitResult(
        coef_=np.asarray(model.coef_, dtype=np.float64).ravel(),
        intercept_=float(model.intercept_),
        runtime_s=float(rt),
    )


def fit_lasso(X, y: np.ndarray, lam: float, *, random_seed: int, max_iter: int = 10_000) -> FitResult:
    # sklearn uses alpha for the L1 coefficient in:
    # (1/2n)||y-Xw||^2 + alpha||w||_1
    t0 = perf_counter()
    model = Lasso(alpha=lam, max_iter=max_iter, random_state=random_seed, selection="cyclic")
    model.fit(X, y)
    rt = perf_counter() - t0
    return FitResult(
        coef_=np.asarray(model.coef_, dtype=np.float64).ravel(),
        intercept_=float(model.intercept_),
        runtime_s=float(rt),
    )

