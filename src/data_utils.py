"""Data loading and inspection helpers for the credit card fraud dataset."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


def load_dataset(csv_path: Path) -> pd.DataFrame:
    """Reads the raw transaction CSV into a DataFrame."""
    return pd.read_csv(csv_path)


def basic_inspection(df: pd.DataFrame) -> dict:
    """Returns shape, dtypes, missing values, duplicate rows and constant columns."""
    constant_cols = [c for c in df.columns if df[c].nunique() <= 1]
    return {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_values_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "constant_columns": constant_cols,
    }


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the raw `Time` column (seconds elapsed since the first
    transaction) into hour-of-day plus a cyclical sin/cos encoding, since
    hour-of-day is periodic and a raw integer hour would imply a false
    discontinuity between hour 23 and hour 0.
    """
    df = df.copy()
    seconds_in_day = 24 * 60 * 60
    df["hour_of_day"] = (df["Time"] % seconds_in_day) // 3600
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    return df


def train_test_split_stratified(df: pd.DataFrame, feature_cols, target_col="Class",
                                 test_size=0.2, seed=RANDOM_SEED):
    """Stratified train/test split used by the supervised models (LR, RF)."""
    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def normal_only_train_test_split(df: pd.DataFrame, feature_cols, target_col="Class",
                                  test_size=0.2, seed=RANDOM_SEED):
    """Reproduces the original blog's split: the training set retains only
    normal (Class == 0) transactions so the autoencoder never sees fraud
    during training; the test set keeps the natural class mix.
    """
    normal_df = df[df[target_col] == 0]
    fraud_df = df[df[target_col] == 1]

    normal_train, normal_test = train_test_split(
        normal_df, test_size=test_size, random_state=seed
    )
    test_df = pd.concat([normal_test, fraud_df]).sample(frac=1, random_state=seed)

    X_train = normal_train[feature_cols]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    return X_train, X_test, y_test


def chronological_train_test_split(df: pd.DataFrame, feature_cols, target_col="Class",
                                    time_col="Time", train_frac=0.8):
    """Splits by transaction time rather than randomly: the first `train_frac`
    of transactions (by `time_col`) become the training set, the remainder
    the test set, with no shuffling and no stratification. This mimics real
    deployment (train on the past, score the future) and is a stricter check
    than a random split, since it cannot "leak" later transactions into
    training and exposes any temporal drift in fraud patterns.
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * train_frac)

    train_df = df_sorted.iloc[:split_idx]
    test_df = df_sorted.iloc[split_idx:]

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    return X_train, X_test, y_train, y_test


def build_amount_scaling_preprocessor() -> ColumnTransformer:
    """Returns an unfitted ColumnTransformer that standardizes `Amount` and
    passes every other feature column through unchanged.

    Returning it unfitted (and building a fresh one per call) is deliberate:
    it is meant to be placed inside an sklearn Pipeline (or rebuilt once per
    cross-validation fold), so `StandardScaler` is only ever fit on whatever
    training data it is handed -- never on the full dataset before a split,
    which would leak test-set statistics into the scaling of training data.
    """
    preprocessor = ColumnTransformer(
        transformers=[("scale_amount", StandardScaler(), ["Amount"])],
        remainder="passthrough",
    )
    return preprocessor.set_output(transform="pandas")
