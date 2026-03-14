from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    random_seed: int = 42
    n_splits: int = 5

    # Target transform (House Prices is highly skewed)
    log_target: bool = True

    # Baseline grids
    ridge_alphas: tuple[float, ...] = (
        1e-3,
        1e-2,
        1e-1,
        1.0,
        10.0,
        100.0,
        1_000.0,
    )
    lasso_lambdas: tuple[float, ...] = (
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
    )

    # Our dynamic proximal method grids
    dyn_lambdas: tuple[float, ...] = (
        1e-4,
        3e-4,
        1e-3,
        3e-3,
        1e-2,
        3e-2,
        1e-1,
    )
    gamma0: float = 1.0
    gamma1_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
    eps: float = 1e-6

    # Annealing: s(k)=1 + anneal_strength/(1+k)^anneal_power
    anneal_strength: float = 2.0
    anneal_power: float = 1.0

    # Solver
    max_iter: int = 5_000
    tol: float = 1e-6
    power_iter: int = 40
    trace_every: int = 50

    # Reporting
    coef_zero_tol: float = 1e-8

