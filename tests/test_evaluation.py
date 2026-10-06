"""Unit tests for evaluation logic on small hand-built prediction tables."""

import json

import numpy as np
import pandas as pd
import pytest

from src.evaluation.leakage import check_leakage
from src.evaluation.metrics import breakdown, compute_metrics, report_text
from src.evaluation.selection import select_stage
from src.evaluation.training_analysis import analyse_stage, history_frame

CLASSES = ("A", "B", "C")


def _predictions(actual, predicted, probs):
    names = np.asarray(CLASSES)
    frame = pd.DataFrame({
        "image_path": [f"img{i}.png" for i in range(len(actual))],
        "patient_id": [f"P{i % 2}" for i in range(len(actual))],
        "sequence": ["DWI"] * len(actual),
        "actual_index": actual,
        "actual": names[actual],
        "predicted_index": predicted,
        "predicted": names[predicted],
        "confidence": np.max(probs, axis=1),
        "correct": np.asarray(actual) == np.asarray(predicted),
    })
    for i, name in enumerate(CLASSES):
        frame[f"prob_{name}"] = np.asarray(probs)[:, i]
    return frame


@pytest.fixture()
def predictions():
    actual = np.array([0, 0, 1, 1, 2, 2])
    predicted = np.array([0, 1, 1, 1, 2, 0])
    probs = np.array([
        [0.8, 0.1, 0.1], [0.3, 0.6, 0.1], [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1], [0.1, 0.1, 0.8], [0.5, 0.1, 0.4],
    ])
    return _predictions(actual, predicted, probs)


def test_metrics_match_hand_computation(predictions):
    m = compute_metrics(predictions, CLASSES)
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["per_class"]["A"]["recall_sensitivity"] == pytest.approx(0.5)
    assert m["per_class"]["B"]["precision"] == pytest.approx(2 / 3)
    # specificity of A: negatives are 4 (B,B,C,C); one C predicted as A → 3/4
    assert m["per_class"]["A"]["specificity"] == pytest.approx(0.75)
    assert m["macro_recall"] == pytest.approx(m["balanced_accuracy"])
    assert m["confusion_matrix"]["rows_actual_cols_predicted"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert m["roc_auc"]["macro_ovr"] is not None


def test_roc_auc_undefined_when_class_absent():
    frame = _predictions(np.array([0, 1]), np.array([0, 1]), np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]]))
    m = compute_metrics(frame, CLASSES)
    assert m["roc_auc"]["per_class"]["C"] is None and m["roc_auc"]["macro_ovr"] is None


def test_report_and_breakdown(predictions):
    assert "macro avg" in report_text(predictions, CLASSES)
    table = breakdown(predictions, "sequence")
    assert table["n"].sum() == len(predictions)


def test_selection_prefers_metric_then_loss():
    rule = {"metric": "macro_recall", "tie_breaker": "loss", "tie_tolerance": 1e-6}
    base = {"accuracy": 0.5, "macro_f1": 0.5}
    better_metric = select_stage({1: {**base, "macro_recall": 0.6, "loss": 0.5},
                                  2: {**base, "macro_recall": 0.7, "loss": 0.9}}, rule)
    assert better_metric["selected_stage"] == 2 and better_metric["test_set_used"] is False
    tie = select_stage({1: {**base, "macro_recall": 0.6, "loss": 0.4},
                        2: {**base, "macro_recall": 0.6, "loss": 0.5}}, rule)
    assert tie["selected_stage"] == 1


def test_overfitting_analysis():
    hist = {"loss": [1.0, 0.6, 0.3, 0.1], "val_loss": [1.1, 0.8, 0.9, 1.2],
            "accuracy": [0.5, 0.7, 0.9, 1.0], "val_accuracy": [0.4, 0.6, 0.6, 0.55]}
    report = analyse_stage(history_frame(hist, 1, 0), 1)
    assert report["best_epoch"] == 2 and report["epochs_after_best"] == 2
    assert report["val_loss_rise_after_min"] == pytest.approx(0.4)
    assert report["accuracy_gap_at_last"] == pytest.approx(0.45)


def _metadata(test_patient="T1", test_hash="h3"):
    return pd.DataFrame({
        "image_path": ["dataset/train/A/a.png", "dataset/validation/A/b.png", "dataset/test/A/c.png"],
        "split": ["train", "validation", "test"],
        "patient_id": ["P1", "P2", test_patient],
        "file_sha256": ["h1", "h2", test_hash],
    })


def test_leakage_checks(tmp_path):
    selection = tmp_path / "selection.json"
    assert not check_leakage(_metadata(), selection, None)["passed"]  # no selection recorded yet
    selection.write_text(json.dumps({"test_set_used": False}))
    assert check_leakage(_metadata(), selection, None)["passed"]
    assert not check_leakage(_metadata(test_patient="P1"), selection, None)["passed"]
    assert not check_leakage(_metadata(test_hash="h1"), selection, None)["passed"]
