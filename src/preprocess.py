from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PreprocessArtifacts:
    preprocessor: ColumnTransformer
    feature_names: list[str]
    numeric_features: list[str]
    categorical_features: list[str]


def infer_feature_types(df: pd.DataFrame, target_col: str) -> tuple[list[str], list[str]]:
    cols = [c for c in df.columns if c != target_col]
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in cols if c not in numeric]
    return numeric, categorical


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            # with_mean=False supports sparse matrices after ColumnTransformer
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        sparse_threshold=0.3,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def fit_preprocessor(df: pd.DataFrame, target_col: str) -> PreprocessArtifacts:
    numeric, categorical = infer_feature_types(df, target_col=target_col)
    preprocessor = build_preprocessor(numeric_features=numeric, categorical_features=categorical)

    X = preprocessor.fit_transform(df.drop(columns=[target_col]))
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback: just create positional names
        n_features = X.shape[1]
        feature_names = [f"x{i}" for i in range(n_features)]

    return PreprocessArtifacts(
        preprocessor=preprocessor,
        feature_names=feature_names,
        numeric_features=numeric,
        categorical_features=categorical,
    )


def transform_with_preprocessor(preprocessor: ColumnTransformer, df: pd.DataFrame) -> np.ndarray:
    return preprocessor.transform(df)

