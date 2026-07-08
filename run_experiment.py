"""Runs the core fraud-detection experiment outside Jupyter.

Reproduces the notebook's central comparison (Logistic Regression vs. Random
Forest, each under a random split and a chronological split) using the same
shared, leakage-free pipelines from src/models.py and src/data_utils.py, so
this script and the notebook can never silently drift apart on
hyperparameters or preprocessing.

Usage:
    python run_experiment.py
    python run_experiment.py --data-path data/creditcard.csv --with-autoencoder
    python run_experiment.py --output outputs/metrics.csv
"""

import argparse
import os
import sys
from pathlib import Path

# Must be set before TensorFlow is ever imported (only happens if
# --with-autoencoder is passed) for reproducible autoencoder training.
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd  # noqa: E402

from data_utils import (  # noqa: E402
    RANDOM_SEED,
    add_time_features,
    chronological_train_test_split,
    load_dataset,
    normal_only_train_test_split,
    train_test_split_stratified,
)
from eval_utils import compute_metrics, metrics_table  # noqa: E402
from models import build_logistic_regression_pipeline, build_random_forest_pipeline  # noqa: E402


def load_and_prepare(data_path: Path) -> pd.DataFrame:
    """Loads the raw CSV, de-duplicates, and adds temporal features -- the
    same steps applied at the start of the notebook.
    """
    df = load_dataset(data_path)
    df = df.drop_duplicates().reset_index(drop=True)
    df = add_time_features(df)
    return df


def run_random_split(df: pd.DataFrame, feature_cols: list) -> dict:
    X_train, X_test, y_train, y_test = train_test_split_stratified(
        df, feature_cols, target_col="Class"
    )

    lr = build_logistic_regression_pipeline()
    lr.fit(X_train, y_train)
    lr_metrics = compute_metrics(
        y_test, lr.predict(X_test), y_score=lr.predict_proba(X_test)[:, 1]
    )

    rf = build_random_forest_pipeline()
    rf.fit(X_train, y_train)
    rf_metrics = compute_metrics(
        y_test, rf.predict(X_test), y_score=rf.predict_proba(X_test)[:, 1]
    )

    return {
        "Logistic Regression (random split)": lr_metrics,
        "Random Forest (random split)": rf_metrics,
    }


def run_chronological_split(df: pd.DataFrame, feature_cols: list) -> dict:
    X_train, X_test, y_train, y_test = chronological_train_test_split(
        df, feature_cols, target_col="Class"
    )

    lr = build_logistic_regression_pipeline()
    lr.fit(X_train, y_train)
    lr_metrics = compute_metrics(
        y_test, lr.predict(X_test), y_score=lr.predict_proba(X_test)[:, 1]
    )

    rf = build_random_forest_pipeline()
    rf.fit(X_train, y_train)
    rf_metrics = compute_metrics(
        y_test, rf.predict(X_test), y_score=rf.predict_proba(X_test)[:, 1]
    )

    return {
        "Logistic Regression (chronological split)": lr_metrics,
        "Random Forest (chronological split)": rf_metrics,
    }


def run_autoencoder(df: pd.DataFrame, feature_cols: list, epochs: int) -> dict:
    """Trains the reproduced autoencoder (normal-only training, leakage-free
    Amount scaling) and returns its metrics at a validation-selected
    threshold. Imported lazily so TensorFlow is only required when this path
    is actually used.
    """
    import numpy as np
    import tensorflow as tf
    from sklearn.metrics import precision_recall_curve
    from sklearn.model_selection import train_test_split as sk_split
    from sklearn.preprocessing import StandardScaler
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    tf.config.experimental.enable_op_determinism()

    X_train_ae, X_test_ae, y_test_ae = normal_only_train_test_split(
        df, feature_cols, target_col="Class"
    )
    scaler = StandardScaler()
    X_train_ae = X_train_ae.copy()
    X_test_ae = X_test_ae.copy()
    X_train_ae["Amount"] = scaler.fit_transform(X_train_ae[["Amount"]])
    X_test_ae["Amount"] = scaler.transform(X_test_ae[["Amount"]])

    input_dim = X_train_ae.shape[1]
    autoencoder = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(14, activation="tanh", activity_regularizer=regularizers.l1(1e-4)),
        layers.Dense(7, activation="relu"),
        layers.Dense(7, activation="tanh"),
        layers.Dense(input_dim, activation="relu"),
    ])
    autoencoder.compile(optimizer="adam", loss="mse")
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    autoencoder.fit(
        X_train_ae.values, X_train_ae.values,
        epochs=epochs, batch_size=32, validation_split=0.1,
        shuffle=True, callbacks=[early_stop], verbose=0,
    )

    reconstructions = autoencoder.predict(X_test_ae.values, verbose=0)
    reconstruction_error = np.mean(np.square(X_test_ae.values - reconstructions), axis=1)

    val_idx, holdout_idx = sk_split(
        range(len(y_test_ae)), test_size=0.5, stratify=y_test_ae, random_state=RANDOM_SEED
    )
    val_error, val_class = reconstruction_error[val_idx], y_test_ae.iloc[val_idx]
    holdout_error, holdout_class = reconstruction_error[holdout_idx], y_test_ae.iloc[holdout_idx]

    precisions, recalls, thresholds = precision_recall_curve(val_class, val_error)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    threshold = thresholds[f1_scores[:-1].argmax()]

    pred = (holdout_error > threshold).astype(int)
    return {"Autoencoder (clean threshold)": compute_metrics(holdout_class, pred, y_score=holdout_error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path", type=Path,
        default=Path(__file__).parent / "data" / "creditcard.csv",
        help="Path to creditcard.csv (run data/download_data.py first if missing).",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "outputs" / "experiment_results.csv",
        help="Where to save the resulting metrics table as CSV.",
    )
    parser.add_argument(
        "--with-autoencoder", action="store_true",
        help="Also train the autoencoder reproduction (requires TensorFlow, slower).",
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
        help="Autoencoder training epochs, only used with --with-autoencoder.",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        raise FileNotFoundError(
            f"{args.data_path} not found. Run `python data/download_data.py` first."
        )

    df = load_and_prepare(args.data_path)
    print(f"Loaded {len(df)} de-duplicated transactions.")

    supervised_features = [c for c in df.columns if c.startswith("V")] + [
        "Amount", "hour_sin", "hour_cos",
    ]

    results = {}
    results.update(run_random_split(df, supervised_features))
    results.update(run_chronological_split(df, supervised_features))

    if args.with_autoencoder:
        ae_features = [c for c in df.columns if c.startswith("V")] + ["Amount"]
        results.update(run_autoencoder(df, ae_features, args.epochs))

    table = metrics_table(results)
    print("\nResults:\n")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output)
    print(f"\nSaved metrics to {args.output}")


if __name__ == "__main__":
    main()
