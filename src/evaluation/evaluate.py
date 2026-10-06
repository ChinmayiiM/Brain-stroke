"""Held-out test-set evaluation — run ONCE, after training and validation-based selection.

Usage (project root, venv active):
    python -m src.evaluation.evaluate

Steps:
  1. Leakage checks (abort if any blocking check fails).
  2. Evaluate the SELECTED model (decided on validation in train.py) on the test split.
  3. For transparency, also evaluate the non-selected stage on the test split —
     reported as a post-selection comparison only; it does not change the selection.
  4. Write metrics, report, predictions, misclassifications, breakdowns and figures.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from src.evaluation.leakage import check_leakage
from src.evaluation.metrics import breakdown, compute_metrics, report_text
from src.evaluation.plots import plot_confusion_matrix, plot_misclassified, plot_roc_curves
from src.evaluation.predictions import load_trained_model, predict_split
from src.training.callbacks import checkpoint_path
from src.training.data_loader import get_split, load_metadata
from src.training.settings import TrainingSettings
from src.utils.config import load_config, resolve_path
from src.utils.logger import get_logger

logger = get_logger("evaluate")

COMPARISON_KEYS = ("loss", "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
                   "weighted_f1")


def _comparison_rows(stage: int, split: str, metrics: dict, class_names: tuple[str, ...]) -> dict:
    row = {"stage": stage, "split": split, **{k: metrics[k] for k in COMPARISON_KEYS}}
    for name in class_names:
        row[f"recall_{name}"] = metrics["per_class"][name]["recall_sensitivity"]
        row[f"precision_{name}"] = metrics["per_class"][name]["precision"]
    row["roc_auc_macro_ovr"] = metrics["roc_auc"]["macro_ovr"]
    return row


def evaluate(config: dict) -> dict:
    settings = TrainingSettings.from_config(config)
    results = settings.results_dir
    plots_dir, metrics_dir, analysis_dir = results / "plots", results / "metrics", results / "analysis"
    selection_file = metrics_dir / "model_selection.json"

    metadata = load_metadata(settings)
    leakage = check_leakage(
        metadata, selection_file, resolve_path(config["paths"]["reports"]) / "data_analysis" / "near_duplicates.csv"
    )
    (analysis_dir / "leakage_checks.json").write_text(json.dumps(leakage, indent=2, default=str), encoding="utf-8")
    if not leakage["passed"]:
        raise RuntimeError(f"Leakage checks failed — refusing to report test metrics: {leakage}")
    logger.info("Leakage checks passed")

    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    selected = int(selection["selected_stage"])
    test_spec = get_split(metadata, "test", settings)

    stage_results = {}
    for stage in (1, 2):
        model = load_trained_model(checkpoint_path(settings, stage))
        predictions = predict_split(model, test_spec, settings)
        stage_results[stage] = (predictions, compute_metrics(predictions, settings.class_names))

    # ---- Selected model: full artifact set -------------------------------------------------
    predictions, metrics = stage_results[selected]
    names = settings.class_names
    metrics_payload = {
        "description": "Performance on the held-out test split of the selected research dataset. "
                       "Not a measure of clinical performance.",
        "selected_stage": selected,
        "selection_criterion": selection["criterion"],
        **metrics,
    }
    (metrics_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    (metrics_dir / "classification_report.txt").write_text(
        f"Selected model: stage {selected} — test split ({metrics['n_images']} images, "
        f"{metrics['n_patients']} patients)\n\n" + report_text(predictions, names),
        encoding="utf-8",
    )
    predictions.to_csv(metrics_dir / "test_predictions.csv", index=False)

    # Confound-controlled view: restrict to MRI sequences that occur in EVERY class of the
    # training split, so the sequence itself cannot reveal the label (see DATA_PIPELINE §6).
    train_rows = metadata[metadata["split"] == "train"]
    shared = sorted(
        seq for seq, group in train_rows.groupby("sequence") if set(group["class_name"]) == set(names)
    )
    subset = predictions[predictions["sequence"].isin(shared)].reset_index(drop=True)
    if not subset.empty and subset["actual_index"].nunique() == len(names):
        subset_metrics = compute_metrics(subset, names)
        metrics_payload["shared_sequence_subset"] = {
            "sequences": shared,
            "rationale": "sequences present in every class of the training split",
            **{k: subset_metrics[k] for k in ("n_images", "accuracy", "balanced_accuracy", "macro_precision",
                                              "macro_recall", "macro_f1", "per_class", "confusion_matrix")},
        }
        (metrics_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        (metrics_dir / "classification_report_shared_sequences.txt").write_text(
            f"Selected model: stage {selected} — test images of sequences {shared} only "
            f"({subset_metrics['n_images']} images)\n\n" + report_text(subset, names),
            encoding="utf-8",
        )

    errors = predictions[~predictions["correct"]]
    errors.to_csv(analysis_dir / "misclassified_images.csv", index=False)
    breakdown(predictions, "sequence").to_csv(analysis_dir / "test_breakdown_by_sequence.csv", index=False)
    breakdown(predictions, "patient_id").to_csv(analysis_dir / "test_breakdown_by_patient.csv", index=False)
    pd.crosstab([errors["actual"], errors["sequence"]], errors["predicted"]).to_csv(
        analysis_dir / "error_patterns.csv"
    )

    cm = np.asarray(metrics["confusion_matrix"]["rows_actual_cols_predicted"])
    plot_confusion_matrix(cm, names, plots_dir / "confusion_matrix.png",
                          f"Confusion matrix — test split, stage-{selected} model")
    if metrics["roc_auc"]["macro_ovr"] is not None or any(metrics["roc_auc"]["per_class"].values()):
        plot_roc_curves(predictions, names, metrics["roc_auc"], plots_dir / "roc_curves.png")
    plot_misclassified(errors, plots_dir / "misclassified_examples.png")

    # ---- Stage comparison (validation = selection basis; test = post-selection report) ----
    rows = []
    for stage in (1, 2):
        validation_metrics = json.loads(
            (metrics_dir / f"validation_metrics_stage{stage}.json").read_text(encoding="utf-8")
        )
        rows.append(_comparison_rows(stage, "validation", validation_metrics, names))
        rows.append(_comparison_rows(stage, "test", stage_results[stage][1], names))
        stage_results[stage][0].to_csv(metrics_dir / f"test_predictions_stage{stage}.csv", index=False)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(metrics_dir / "stage_comparison.csv", index=False)

    # Record evaluation in the best model's metadata.
    meta_path = settings.best_dir / "model_metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["test_evaluation"] = {k: metrics[k] for k in COMPARISON_KEYS} | {"n_images": metrics["n_images"]}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info("Test metrics (stage %d): %s", selected, {k: round(metrics[k], 4) for k in COMPARISON_KEYS})
    print(report_text(predictions, names))
    print(comparison.round(4).to_string(index=False))
    return metrics_payload


def main() -> int:
    evaluate(load_config())
    return 0


if __name__ == "__main__":
    sys.exit(main())
