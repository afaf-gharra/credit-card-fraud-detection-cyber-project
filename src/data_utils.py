"""Data loading and inspection helpers for the credit card fraud dataset."""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


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
    from sklearn.model_selection import train_test_split

    X = df[feature_cols]
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def normal_only_train_test_split(df: pd.DataFrame, feature_cols, target_col="Class",
                                  test_size=0.2, seed=RANDOM_SEED):
    """Reproduces the original blog's split: the training set retains only
    normal (Class == 0) transactions so the autoencoder never sees fraud
    during training; the test set keeps the natural class mix.
    """
    from sklearn.model_selection import train_test_split

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
