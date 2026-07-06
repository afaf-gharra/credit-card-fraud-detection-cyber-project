"""Evaluation metric and plotting helpers shared across all trained models.

Centralizing metric computation here avoids recomputing/duplicating the same
precision/recall/F1/MCC/ROC-AUC/PR-AUC logic for each of the three models
trained in the notebook.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_score=None, beta=0.5) -> dict:
    """Computes classification metrics appropriate for a severely imbalanced
    binary fraud-detection task.

    beta < 1 (default 0.5) weights precision more heavily than recall, on the
    assumption that false positives (blocking legitimate transactions) carry
    real customer-friction cost. Pass beta > 1 to weight recall instead if the
    business priority is catching more fraud at the expense of more blocked
    legitimate transactions.
    """
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        f"f{beta}": fbeta_score(y_true, y_pred, beta=beta, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
    return metrics


def plot_confusion_matrix(y_true, y_pred, title, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["Normal", "Fraud"])
    ax.set_yticks([0, 1], ["Normal", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return ax


def metrics_table(results: dict) -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame(results).T
