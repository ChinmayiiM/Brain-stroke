"""Persist training history and analyse the curves for overfitting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def history_frame(history: dict[str, list[float]], stage: int, first_epoch: int) -> pd.DataFrame:
    """Keras ``History.history`` → tidy frame with 1-based global epoch numbers."""
    frame = pd.DataFrame(history)
    frame.insert(0, "epoch", range(first_epoch + 1, first_epoch + 1 + len(frame)))
    frame.insert(0, "stage", stage)
    return frame


def save_history(frame: pd.DataFrame, directory: Path) -> None:
    """Write the combined history as CSV and JSON (survives the Python process)."""
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / "training_history.csv", index=False)
    (directory / "training_history.json").write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2), encoding="utf-8"
    )


def analyse_stage(frame: pd.DataFrame, stage: int, monitor: str = "val_loss") -> dict[str, Any]:
    """Quantify overfitting signals for one stage from its real curves.

    Signals reported (not thresholds that change training):
      * train/validation accuracy gap at the best epoch and at the last epoch
      * validation loss increase from its minimum to the last epoch
      * epochs trained after the best validation epoch
      * validation-accuracy volatility (std of epoch-to-epoch change)
    """
    data = frame[frame["stage"] == stage].reset_index(drop=True)
    if data.empty:
        return {"stage": stage, "epochs": 0}
    best_idx = int(data[monitor].idxmin())
    best, last = data.iloc[best_idx], data.iloc[-1]
    return {
        "stage": stage,
        "epochs": int(len(data)),
        "best_epoch": int(best["epoch"]),
        "epochs_after_best": int(len(data) - 1 - best_idx),
        "best": {k: float(best[k]) for k in ("loss", "val_loss", "accuracy", "val_accuracy")},
        "last": {k: float(last[k]) for k in ("loss", "val_loss", "accuracy", "val_accuracy")},
        "accuracy_gap_at_best": float(best["accuracy"] - best["val_accuracy"]),
        "accuracy_gap_at_last": float(last["accuracy"] - last["val_accuracy"]),
        "val_loss_rise_after_min": float(last["val_loss"] - best["val_loss"]),
        "train_loss_change_after_best": float(last["loss"] - best["loss"]),
        "val_accuracy_volatility": float(data["val_accuracy"].diff().abs().std()) if len(data) > 2 else None,
    }
