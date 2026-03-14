from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse


def _matvec(X: Any, v: np.ndarray) -> np.ndarray:
    out = X @ v
    if sparse.issparse(out):
        out = out.A.ravel()
    return np.asarray(out).ravel()


def _rmatvec(X: Any, v: np.ndarray) -> np.ndarray:
    out = X.T @ v
    if sparse.issparse(out):
        out = out.A.ravel()
    return np.asarray(out).ravel()


def estimate_lipschitz_constant(X: Any, n: int, n_iter: int, seed: int = 0) -> float:
    """
    Estimate L for grad f(beta) = (1/n) X^T (X beta - y) where f=(1/2n)||Xb-y||^2.
    L = ||X||_2^2 / n.
    """
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    v = rng.normal(size=p)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(n_iter):
        Xv = _matvec(X, v)
        w = _rmatvec(X, Xv)
        norm_w = np.linalg.norm(w)
        if norm_w == 0:
            return 1.0
        v = w / norm_w
    # Rayleigh quotient approximation for ||X||_2^2
    Xv = _matvec(X, v)
    num = float(np.dot(Xv, Xv))
    return max(num / float(n), 1e-12)


def soft_threshold(x: np.ndarray, thresh: np.ndarray | float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0.0)


@dataclass(frozen=True)
class DynProxResult:
    coef_: np.ndarray
    n_iter_: int
    runtime_s: float
    converged: bool
    history: dict[str, list[float]]


def dynamic_proximal_gradient_lasso(
    X: Any,
    y: np.ndarray,
    lam: float,
    *,
    gamma0: float,
    gamma1: float,
    eps: float,
    anneal_strength: float,
    anneal_power: float,
    max_iter: int,
    tol: float,
    power_iter: int,
    seed: int = 0,
    trace_every: int = 50,
) -> DynProxResult:
    """
    Minimize (1/2n)||Xb - y||^2 + lam||b||_1 using prox-grad with dynamic thresholds:
      g_i = |b_i|/(|b_i|+eps)
      s(k)=1 + anneal_strength/(1+k)^anneal_power
      tau_i = alpha*lam*(gamma0 + gamma1*(1-g_i))*s(k)
      b <- SoftThresh(b - alpha*grad, tau)
    """
    t0 = perf_counter()
    y = np.asarray(y, dtype=np.float64).ravel()
    n, p = X.shape
    L = estimate_lipschitz_constant(X, n=n, n_iter=power_iter, seed=seed)
    alpha = 1.0 / L

    b = np.zeros(p, dtype=np.float64)
    history: dict[str, list[float]] = {"obj": [], "nnz": [], "step_norm": []}

    def obj(beta: np.ndarray) -> float:
        r = _matvec(X, beta) - y
        return 0.5 * float(np.dot(r, r)) / float(n) + float(lam) * float(np.sum(np.abs(beta)))

    prev_obj = obj(b)
    converged = False

    for k in range(1, max_iter + 1):
        r = _matvec(X, b) - y
        grad = _rmatvec(X, r) / float(n)
        z = b - alpha * grad

        g = np.abs(b) / (np.abs(b) + eps)
        s_k = 1.0 + float(anneal_strength) / float((1.0 + k) ** float(anneal_power))
        tau = (alpha * lam) * (gamma0 + gamma1 * (1.0 - g)) * s_k

        b_next = soft_threshold(z, tau)
        step = float(np.linalg.norm(b_next - b))
        b = b_next

        if (k % trace_every) == 0 or k == 1:
            cur_obj = obj(b)
            history["obj"].append(cur_obj)
            history["nnz"].append(float(np.count_nonzero(b)))
            history["step_norm"].append(step)
            if abs(prev_obj - cur_obj) <= tol * max(1.0, abs(prev_obj)):
                converged = True
                break
            prev_obj = cur_obj
        else:
            if step <= tol * max(1.0, float(np.linalg.norm(b))):
                converged = True
                break

    runtime_s = perf_counter() - t0
    return DynProxResult(
        coef_=b,
        n_iter_=k,
        runtime_s=runtime_s,
        converged=converged,
        history=history,
    )

