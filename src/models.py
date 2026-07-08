"""Shared model-pipeline factories, used by both the notebook and
run_experiment.py so the two never define hyperparameters independently.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from data_utils import RANDOM_SEED, build_amount_scaling_preprocessor


def build_logistic_regression_pipeline(seed: int = RANDOM_SEED) -> Pipeline:
    """Logistic Regression with leakage-free Amount scaling: the scaler is
    part of the pipeline, so calling `.fit()` on a training split only ever
    fits it on that split.
    """
    return Pipeline([
        ("preprocess", build_amount_scaling_preprocessor()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
    ])


def build_random_forest_pipeline(seed: int = RANDOM_SEED) -> Pipeline:
    """Random Forest wrapped in the same preprocessing pipeline as Logistic
    Regression for consistency, even though tree splits are scale-invariant.
    """
    return Pipeline([
        ("preprocess", build_amount_scaling_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        random_state=seed, n_jobs=-1)),
    ])
