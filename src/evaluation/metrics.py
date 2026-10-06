"""Classification metrics computed with scikit-learn from real predictions.

Multi-class (3 classes here) → macro averages are the headline numbers because
they weight the minority class (Ischemic) equally; weighted averages are
reported for reference. Per-class sensitivity (recall) and specificity are
computed one-vs-rest.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.evaluation.predictions import probability_matrix


def _per_class_specificity(cm: np.ndarray) -> np.ndarray:
    total = cm.sum()
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = total - tp - fp - fn
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(tn + fp > 0, tn / (tn + fp), np.nan)


def _roc_auc(y_true: np.ndarray, probs: np.ndarray, class_names: tuple[str, ...]) -> dict[str, Any]:
    """One-vs-rest ROC-AUC per class + macro; undefined when a class has no positives or no negatives."""
    per_class: dict[str, float | None] = {}
    for index, name in enumerate(class_names):
        positives = y_true == index
        per_class[name] = (
            float(roc_auc_score(positives, probs[:, index])) if 0 < positives.sum() < len(y_true) else None
        )
    defined = [v for v in per_class.values() if v is not None]
    macro = float(np.mean(defined)) if len(defined) == len(class_names) else None
    return {"per_class": per_class, "macro_ovr": macro,
            "note": None if macro is not None else "undefined: a class is absent from y_true"}


def compute_metrics(predictions: pd.DataFrame, class_names: tuple[str, ...]) -> dict[str, Any]:
    """All headline, per-class and probabilistic metrics for one set of predictions."""
    labels = list(range(len(class_names)))
    y_true = predictions["actual_index"].to_numpy()
    y_pred = predictions["predicted_index"].to_numpy()
    probs = probability_matrix(predictions, class_names)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    specificity = _per_class_specificity(cm)
    averages = {}
    for average in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average=average, zero_division=0)
        averages[average] = {"precision": float(p), "recall": float(r), "f1": float(f)}

    return {
        "n_images": int(len(y_true)),
        "n_patients": int(predictions["patient_id"].nunique()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": averages["macro"]["precision"],
        "macro_recall": averages["macro"]["recall"],
        "macro_f1": averages["macro"]["f1"],
        "weighted_precision": averages["weighted"]["precision"],
        "weighted_recall": averages["weighted"]["recall"],
        "weighted_f1": averages["weighted"]["f1"],
        "loss": float(log_loss(y_true, probs, labels=labels)),
        "per_class": {
            name: {
                "support": int(support[i]),
                "precision": float(precision[i]),
                "recall_sensitivity": float(recall[i]),
                "specificity": None if np.isnan(specificity[i]) else float(specificity[i]),
                "f1": float(f1[i]),
            }
            for i, name in enumerate(class_names)
        },
        "roc_auc": _roc_auc(y_true, probs, class_names),
        "confusion_matrix": {"labels": list(class_names), "rows_actual_cols_predicted": cm.tolist()},
    }


def report_text(predictions: pd.DataFrame, class_names: tuple[str, ...]) -> str:
    """scikit-learn classification report (per class + macro/weighted averages)."""
    return classification_report(
        predictions["actual_index"],
        predictions["predicted_index"],
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )


def breakdown(predictions: pd.DataFrame, by: str) -> pd.DataFrame:
    """Accuracy and prediction counts per group (e.g. sequence, patient).

    Groups are small, so only counts and accuracy are reported — macro metrics
    on a handful of images would be meaningless.
    """
    rows = []
    for key, group in predictions.groupby([by, "actual"]):
        counts = group["predicted"].value_counts().to_dict()
        rows.append({
            by: key[0],
            "actual": key[1],
            "n": int(len(group)),
            "correct": int(group["correct"].sum()),
            "accuracy": float(group["correct"].mean()),
            "mean_confidence": float(group["confidence"].mean()),
            "predicted_counts": "; ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        })
    return pd.DataFrame(rows)
