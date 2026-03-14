from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from src.preprocess import PreprocessArtifacts, fit_preprocessor


TARGET_COL = "SalePrice"


@dataclass(frozen=True)
class DatasetBundle:
    X: object  # scipy sparse or numpy array
    y: np.ndarray
    feature_names: list[str]
    artifacts: PreprocessArtifacts


def load_house_prices_train(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Expected Kaggle train.csv at data/raw/train.csv")
    df = pd.read_csv(path)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in {path}")
    return df


def make_dataset_bundle(
    df: pd.DataFrame,
    log_target: bool,
) -> DatasetBundle:
    artifacts = fit_preprocessor(df, target_col=TARGET_COL)
    X = artifacts.preprocessor.transform(df.drop(columns=[TARGET_COL]))
    y = df[TARGET_COL].to_numpy(dtype=np.float64)
    if log_target:
        y = np.log1p(y)
    return DatasetBundle(X=X, y=y, feature_names=artifacts.feature_names, artifacts=artifacts)


def make_cv_splits(
    n_samples: int,
    n_splits: int,
    random_seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    idx = np.arange(n_samples)
    return [(train_idx, val_idx) for train_idx, val_idx in kf.split(idx)]


def make_holdout_split(
    n_samples: int,
    random_seed: int,
    test_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.arange(n_samples)
    tr, va = train_test_split(idx, test_size=test_size, random_state=random_seed, shuffle=True)
    return tr, va

