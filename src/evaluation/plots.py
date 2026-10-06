"""Evaluation figures drawn from real training histories and test predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_curve

from src.data.visualization import GRID, INK_PRIMARY, INK_SECONDARY, SERIES_COLOURS, SURFACE, _style_axes
from src.evaluation.predictions import probability_matrix
from src.utils.config import PROJECT_ROOT

TRAIN_COLOUR, VAL_COLOUR = SERIES_COLOURS[0], SERIES_COLOURS[1]


def _finish(fig: plt.Figure, ax: plt.Axes, path: Path, title: str) -> None:
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metric_curves(history: pd.DataFrame, metric: str, path: Path, title: str, ylabel: str) -> None:
    """Training vs validation curve for one metric; dashed line marks the start of fine-tuning."""
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    _style_axes(ax)
    ax.plot(history["epoch"], history[metric], color=TRAIN_COLOUR, linewidth=2, label=f"Training {ylabel.lower()}")
    ax.plot(history["epoch"], history[f"val_{metric}"], color=VAL_COLOUR, linewidth=2,
            label=f"Validation {ylabel.lower()}")
    if history["stage"].nunique() > 1:
        boundary = history.loc[history["stage"] == 2, "epoch"].min() - 0.5
        ax.axvline(boundary, color=INK_SECONDARY, linestyle="--", linewidth=1)
        ax.text(boundary, ax.get_ylim()[1], " fine-tuning →", va="top", fontsize=8, color=INK_SECONDARY)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_PRIMARY)
    _finish(fig, ax, path, title)


def plot_confusion_matrix(cm: np.ndarray, class_names: tuple[str, ...], path: Path, title: str) -> None:
    """Counts with row-normalised share in each cell (rows = actual, columns = predicted)."""
    row_totals = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals > 0)
    fig, ax = plt.subplots(figsize=(5.8, 4.8), facecolor=SURFACE)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    ticks = range(len(class_names))
    ax.set_xticks(ticks, class_names)
    ax.set_yticks(ticks, class_names)
    ax.set_xlabel("Predicted", color=INK_SECONDARY)
    ax.set_ylabel("Actual", color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY)
    for (i, j), count in np.ndenumerate(cm):
        ax.text(j, i, f"{count}\n({share[i, j]:.0%})", ha="center", va="center", fontsize=10,
                color="#ffffff" if share[i, j] > 0.55 else INK_PRIMARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, loc="left", color=INK_PRIMARY)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curves(predictions: pd.DataFrame, class_names: tuple[str, ...], auc: dict, path: Path) -> None:
    """One-vs-rest ROC curve per class (only classes with both positives and negatives)."""
    probs = probability_matrix(predictions, class_names)
    y_true = predictions["actual_index"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.8, 5), facecolor=SURFACE)
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=GRID, linestyle="--", linewidth=1)
    for index, name in enumerate(class_names):
        if auc["per_class"].get(name) is None:
            continue
        fpr, tpr, _ = roc_curve(y_true == index, probs[:, index])
        n_pos = int((y_true == index).sum())
        ax.plot(fpr, tpr, color=SERIES_COLOURS[index % len(SERIES_COLOURS)], linewidth=2,
                label=f"{name} (AUC {auc['per_class'][name]:.3f}, n+={n_pos})")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK_PRIMARY)
    _finish(fig, ax, path, "ROC curves, one-vs-rest (test set)")


def plot_misclassified(errors: pd.DataFrame, path: Path, max_images: int = 12) -> None:
    """Grid of misclassified test images, most confident errors first."""
    if errors.empty:
        return
    shown = errors.sort_values("confidence", ascending=False).head(max_images)
    columns = min(4, len(shown))
    rows = int(np.ceil(len(shown) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.6 * columns, 3.4 * rows), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.45})
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, row) in zip(axes, shown.iterrows()):
        with Image.open(PROJECT_ROOT / row["image_path"]) as img:
            ax.imshow(np.asarray(img.convert("L")), cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"{row['patient_id']} · {row['sequence']}\nactual {row['actual']}\npred {row['predicted']} ({row['confidence']:.2f})",
            fontsize=7.5, color=INK_PRIMARY,
        )
    fig.suptitle("Misclassified test images (highest-confidence errors first)", fontsize=11,
                 color=INK_PRIMARY, x=0.02, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
