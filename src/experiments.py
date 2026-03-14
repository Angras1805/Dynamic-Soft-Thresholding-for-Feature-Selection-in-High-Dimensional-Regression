from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import sparse

from src.baselines import fit_lasso, fit_ridge
from src.config import ExperimentConfig
from src.data import load_house_prices_train, make_cv_splits, make_dataset_bundle
from src.metrics import count_nonzero, mean_squared_error
from src.optimizers.prox_dynamic import dynamic_proximal_gradient_lasso


def _subset_rows(X, idx: np.ndarray):
    if sparse.issparse(X):
        return X[idx]
    return np.asarray(X)[idx]


def _predict(X, coef: np.ndarray, intercept: float = 0.0) -> np.ndarray:
    if sparse.issparse(X):
        y = X @ coef
        return np.asarray(y).ravel() + intercept
    return (np.asarray(X) @ coef).ravel() + intercept


def _ensure_dirs(out_dir: Path) -> Path:
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    return plots_dir


def run_all_experiments(data_dir: Path, out_dir: Path, cfg: ExperimentConfig) -> pd.DataFrame:
    """
    Runs CV tuning for Ridge, LASSO, and Dynamic LASSO (ours) using identical folds.
    Writes plots and a detailed JSON of the full grid.
    Returns a DataFrame of aggregated results (one row per configuration).
    """
    plots_dir = _ensure_dirs(out_dir)

    df = load_house_prices_train(data_dir)
    bundle = make_dataset_bundle(df, log_target=cfg.log_target)

    X = bundle.X
    y = bundle.y
    n = len(y)
    splits = make_cv_splits(n_samples=n, n_splits=cfg.n_splits, random_seed=cfg.random_seed)

    rows: list[dict] = []

    def eval_ridge(alpha: float) -> None:
        fold_mse: list[float] = []
        fold_nnz: list[int] = []
        fold_rt: list[float] = []
        for tr, va in splits:
            Xtr, Xva = _subset_rows(X, tr), _subset_rows(X, va)
            ytr, yva = y[tr], y[va]
            fit = fit_ridge(Xtr, ytr, alpha=alpha, random_seed=cfg.random_seed)
            pred = _predict(Xva, fit.coef_, fit.intercept_)
            fold_mse.append(mean_squared_error(yva, pred))
            fold_nnz.append(count_nonzero(fit.coef_, cfg.coef_zero_tol))
            fold_rt.append(fit.runtime_s)
        rows.append(
            {
                "method": "ridge",
                "param": alpha,
                "mean_mse": float(np.mean(fold_mse)),
                "std_mse": float(np.std(fold_mse)),
                "mean_nnz": float(np.mean(fold_nnz)),
                "mean_runtime_s": float(np.mean(fold_rt)),
            }
        )

    def eval_lasso(lam: float) -> None:
        fold_mse: list[float] = []
        fold_nnz: list[int] = []
        fold_rt: list[float] = []
        for tr, va in splits:
            Xtr, Xva = _subset_rows(X, tr), _subset_rows(X, va)
            ytr, yva = y[tr], y[va]
            fit = fit_lasso(Xtr, ytr, lam=lam, random_seed=cfg.random_seed)
            pred = _predict(Xva, fit.coef_, fit.intercept_)
            fold_mse.append(mean_squared_error(yva, pred))
            fold_nnz.append(count_nonzero(fit.coef_, cfg.coef_zero_tol))
            fold_rt.append(fit.runtime_s)
        rows.append(
            {
                "method": "lasso",
                "param": lam,
                "mean_mse": float(np.mean(fold_mse)),
                "std_mse": float(np.std(fold_mse)),
                "mean_nnz": float(np.mean(fold_nnz)),
                "mean_runtime_s": float(np.mean(fold_rt)),
            }
        )

    def eval_dynamic(lam: float, gamma1: float) -> None:
        fold_mse: list[float] = []
        fold_nnz: list[int] = []
        fold_rt: list[float] = []
        fold_iter: list[int] = []
        for tr, va in splits:
            Xtr, Xva = _subset_rows(X, tr), _subset_rows(X, va)
            ytr, yva = y[tr], y[va]
            t0 = perf_counter()
            res = dynamic_proximal_gradient_lasso(
                Xtr,
                ytr,
                lam=lam,
                gamma0=cfg.gamma0,
                gamma1=gamma1,
                eps=cfg.eps,
                anneal_strength=cfg.anneal_strength,
                anneal_power=cfg.anneal_power,
                max_iter=cfg.max_iter,
                tol=cfg.tol,
                power_iter=cfg.power_iter,
                seed=cfg.random_seed,
                trace_every=cfg.trace_every,
            )
            rt = perf_counter() - t0
            pred = _predict(Xva, res.coef_, 0.0)
            fold_mse.append(mean_squared_error(yva, pred))
            fold_nnz.append(count_nonzero(res.coef_, cfg.coef_zero_tol))
            fold_rt.append(float(rt))
            fold_iter.append(int(res.n_iter_))

        rows.append(
            {
                "method": "dynamic_lasso",
                "param": lam,
                "gamma1": gamma1,
                "mean_mse": float(np.mean(fold_mse)),
                "std_mse": float(np.std(fold_mse)),
                "mean_nnz": float(np.mean(fold_nnz)),
                "mean_runtime_s": float(np.mean(fold_rt)),
                "mean_iters": float(np.mean(fold_iter)),
            }
        )

    # Baselines
    for a in cfg.ridge_alphas:
        eval_ridge(a)
    for lam in cfg.lasso_lambdas:
        eval_lasso(lam)

    # Ours
    for lam in cfg.dyn_lambdas:
        for g1 in cfg.gamma1_grid:
            eval_dynamic(lam, g1)

    results = pd.DataFrame(rows)
    (out_dir / "grid_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "preprocess_info.json").write_text(
        json.dumps(
            {
                "n_samples": int(n),
                "n_features_after_preprocess": int(X.shape[1]),
                "n_numeric": len(bundle.artifacts.numeric_features),
                "n_categorical": len(bundle.artifacts.categorical_features),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _write_plots(results, plots_dir)
    _write_best_summary(results, out_dir, cfg)
    return results


def _write_best_summary(results: pd.DataFrame, out_dir: Path, cfg: ExperimentConfig) -> None:
    best_rows = []
    for method in sorted(results["method"].unique()):
        sub = results[results["method"] == method].copy()
        # prioritize MSE; if tie-ish, pick smaller nnz; then runtime
        sub = sub.sort_values(["mean_mse", "mean_nnz", "mean_runtime_s"], ascending=[True, True, True])
        best_rows.append(sub.iloc[0].to_dict())
    payload = {"config": asdict(cfg), "best_by_method": best_rows}
    (out_dir / "best_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_plots(results: pd.DataFrame, plots_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    # MSE vs sparsity
    plt.figure(figsize=(9, 6))
    ax = sns.scatterplot(
        data=results,
        x="mean_nnz",
        y="mean_mse",
        hue="method",
        style="method",
        s=80,
    )
    ax.set_title("CV Mean MSE vs Sparsity (nonzeros)")
    ax.set_xlabel("Mean #Nonzero Coefficients (CV)")
    ax.set_ylabel("Mean MSE (CV)")
    plt.tight_layout()
    plt.savefig(plots_dir / "mse_vs_sparsity.png", dpi=180)
    plt.close()

    # Runtime bars
    plt.figure(figsize=(9, 6))
    ax = sns.barplot(data=results, x="method", y="mean_runtime_s", errorbar=None)
    ax.set_title("Mean Fit Runtime per CV Fold")
    ax.set_xlabel("Method")
    ax.set_ylabel("Seconds (mean per fold)")
    plt.tight_layout()
    plt.savefig(plots_dir / "runtime_by_method.png", dpi=180)
    plt.close()

    # For dynamic method, show mse vs nnz colored by gamma1
    dyn = results[results["method"] == "dynamic_lasso"].copy()
    if len(dyn) > 0 and "gamma1" in dyn.columns:
        plt.figure(figsize=(9, 6))
        ax = sns.scatterplot(
            data=dyn,
            x="mean_nnz",
            y="mean_mse",
            hue="gamma1",
            palette="viridis",
            s=90,
        )
        ax.set_title("Dynamic LASSO: MSE vs Sparsity colored by gamma1")
        ax.set_xlabel("Mean #Nonzero Coefficients (CV)")
        ax.set_ylabel("Mean MSE (CV)")
        plt.tight_layout()
        plt.savefig(plots_dir / "dynamic_mse_vs_sparsity_gamma1.png", dpi=180)
        plt.close()

