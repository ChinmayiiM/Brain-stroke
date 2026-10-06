"""Training entry point — two-stage EfficientNetB0 transfer learning.

Usage (project root, venv active):
    python -m src.training.train --sanity-check   # verify pipeline + model, no training
    python -m src.training.train                  # full two-stage training + validation-based selection
    python -m src.evaluation.evaluate             # afterwards: held-out test evaluation

This module reads ONLY the train and validation splits. Stage selection is
decided on validation metrics and recorded before any test evaluation exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import yaml

from src.evaluation.metrics import compute_metrics
from src.evaluation.plots import plot_metric_curves
from src.evaluation.predictions import load_trained_model, predict_split
from src.evaluation.selection import select_stage
from src.evaluation.training_analysis import analyse_stage, history_frame, save_history
from src.training.callbacks import build_callbacks, checkpoint_path
from src.training.class_balance import compute_class_weights
from src.training.data_loader import build_dataset, get_split, load_metadata
from src.training.inspection import parameter_counts, save_summary, verify_stage1, verify_stage2
from src.training.model import build_model, compile_model, get_backbone, unfreeze_top_layers
from src.training.settings import TrainingSettings
from src.utils.config import load_config
from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger("train")

DISCLAIMER = "AI-assisted MRI image classification prototype — not a medical diagnosis."
FIT_SPLITS = ("train", "validation")


# ---------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_model_metadata(settings: TrainingSettings, directory: Path, extra: dict) -> None:
    """Everything inference needs to use the model correctly, stored next to it."""
    metadata = {
        "model_name": settings.model_name,
        "backbone": "EfficientNetB0",
        "class_names": list(settings.class_names),
        "input_shape": list(settings.input_shape),
        "output": "sigmoid (1 unit)" if settings.is_binary else f"softmax ({settings.num_classes} units)",
        "preprocessing": {
            "pipeline": "src.preprocessing.image_ops.preprocess_image",
            "color": "RGB (grayscale replicated)",
            "resize": list(settings.preprocessing.image_size),
            "resize_interpolation": settings.preprocessing.interpolation,
            "pixel_range": "[0, 255] float32 — EfficientNet normalises internally",
        },
        "sequences": list(settings.sequences) if settings.sequences else "all",
        "disclaimer": DISCLAIMER,
        **extra,
    }
    _write_json(directory / "model_metadata.json", metadata)


def describe_dataset(metadata: pd.DataFrame, settings: TrainingSettings, train_ds: tf.data.Dataset) -> dict:
    """Log the actual dataset facts before training (no images outside train are loaded)."""
    counts = pd.crosstab(metadata["split"], metadata["class_name"]).reindex(columns=list(settings.class_names))
    x, y = next(iter(train_ds))
    facts = {
        "num_classes": settings.num_classes,
        "label_encoding": {name: i for i, name in enumerate(settings.class_names)},
        "images_per_split_and_class": counts.to_dict(orient="index"),
        "images_per_split": metadata["split"].value_counts().to_dict(),
        "batch_image_shape": list(x.shape),
        "batch_label_shape": list(y.shape),
        "model_input_shape": list(settings.input_shape),
        "source_image_sizes": metadata.groupby(["image_width", "image_height"]).size().rename("n")
        .reset_index().to_dict(orient="records"),
    }
    logger.info("Dataset verification:\n%s", json.dumps(facts, indent=2, default=str))
    return facts


def layer_report(model: keras.Model) -> dict:
    """Which backbone layers are trainable vs frozen, with parameter counts."""
    backbone = get_backbone(model)
    trainable = [layer.name for layer in backbone.layers if layer.trainable_weights]
    frozen_with_weights = [layer.name for layer in backbone.layers if layer.weights and not layer.trainable_weights]
    return {
        "trainable_backbone_layers": trainable,
        "frozen_backbone_layers_with_weights": len(frozen_with_weights),
        "frozen_batchnorm_layers": sum(
            isinstance(layer, keras.layers.BatchNormalization) and not layer.trainable_weights
            for layer in backbone.layers
        ),
        "parameters": parameter_counts(model),
    }


def _plot_curves(history: pd.DataFrame, plots_dir: Path, suffix: str) -> None:
    plot_metric_curves(history, "accuracy", plots_dir / f"accuracy{suffix}.png",
                       "Training vs validation accuracy", "Accuracy")
    plot_metric_curves(history, "loss", plots_dir / f"loss{suffix}.png",
                       "Training vs validation loss", "Loss")
    if "f1_macro" in history:
        plot_metric_curves(history, "f1_macro", plots_dir / f"f1_macro{suffix}.png",
                           "Training vs validation macro-F1", "Macro-F1")


# ---------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------

def train(config: dict) -> dict:
    settings = TrainingSettings.from_config(config)
    set_global_seed(settings.seed)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = settings.training_reports_dir / run_id
    results = settings.results_dir
    plots_dir, metrics_dir, analysis_dir = results / "plots", results / "metrics", results / "analysis"
    for directory in (run_dir, plots_dir, metrics_dir, analysis_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    metadata = load_metadata(settings)
    train_spec = get_split(metadata, FIT_SPLITS[0], settings)
    val_spec = get_split(metadata, FIT_SPLITS[1], settings)
    train_ds = build_dataset(train_spec, settings, training=True)
    val_ds = build_dataset(val_spec, settings, training=False)
    dataset_facts = describe_dataset(metadata, settings, train_ds)
    class_weight = compute_class_weights(train_spec.labels, settings.class_names, settings.class_weighting)

    # ---- Stage 1: feature extraction (frozen backbone) ---------------------------------
    model = build_model(settings)
    stage1_check = verify_stage1(model)
    logger.info("Stage 1 freeze check: %s", stage1_check)
    compile_model(model, settings, settings.stage1.learning_rate)
    save_summary(model, run_dir / "model_stage1.txt")
    history1 = model.fit(
        train_ds, validation_data=val_ds, epochs=settings.stage1.epochs,
        class_weight=class_weight, callbacks=build_callbacks(settings, 1, run_dir), verbose=2,
    )
    frame1 = history_frame(history1.history, stage=1, first_epoch=0)
    save_history(frame1, metrics_dir)
    _plot_curves(frame1, plots_dir, "_stage1")
    analysis1 = analyse_stage(frame1, 1, settings.checkpoint.monitor)
    logger.info("Stage 1 curve analysis: %s", analysis1)
    epochs_done = len(frame1)

    # Continue from the best stage-1 weights (explicit, independent of EarlyStopping behaviour).
    model.set_weights(load_trained_model(checkpoint_path(settings, 1)).get_weights())

    # ---- Stage 2: fine-tuning (top layers, BatchNorm frozen) ---------------------------
    unfreeze_top_layers(model, settings.stage2.unfreeze_top_n_layers)
    stage2_check = verify_stage2(model, settings.stage2.unfreeze_top_n_layers)
    fine_tune_layers = layer_report(model)
    logger.info("Stage 2 trainable layers: %s", fine_tune_layers)
    _write_json(analysis_dir / "fine_tuning_layers.json", fine_tune_layers)
    compile_model(model, settings, settings.stage2.learning_rate)  # recompile after changing trainability
    save_summary(model, run_dir / "model_stage2.txt")
    history2 = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs_done + settings.stage2.epochs, initial_epoch=epochs_done,
        class_weight=class_weight, callbacks=build_callbacks(settings, 2, run_dir), verbose=2,
    )
    frame2 = history_frame(history2.history, stage=2, first_epoch=epochs_done)
    history = pd.concat([frame1, frame2], ignore_index=True)
    save_history(history, metrics_dir)
    _plot_curves(history, plots_dir, "")
    analysis2 = analyse_stage(history, 2, settings.checkpoint.monitor)
    _write_json(analysis_dir / "overfitting_analysis.json", {"stage1": analysis1, "stage2": analysis2})

    # ---- Final model (end-of-training state) -------------------------------------------
    settings.final_dir.mkdir(parents=True, exist_ok=True)
    final_path = settings.final_dir / f"{settings.model_name}.keras"
    model.save(final_path)

    # ---- Stage selection on VALIDATION data only ---------------------------------------
    validation_metrics = {}
    for stage in (1, 2):
        stage_model = load_trained_model(checkpoint_path(settings, stage))
        predictions = predict_split(stage_model, val_spec, settings)
        validation_metrics[stage] = compute_metrics(predictions, settings.class_names)
        predictions.to_csv(metrics_dir / f"validation_predictions_stage{stage}.csv", index=False)
        _write_json(metrics_dir / f"validation_metrics_stage{stage}.json", validation_metrics[stage])
    selection = select_stage(validation_metrics, settings.model_selection)
    selected_stage = selection["selected_stage"]
    selection["selected_checkpoint"] = checkpoint_path(settings, selected_stage).name
    _write_json(metrics_dir / "model_selection.json", selection)
    logger.info("Model selection (validation only): %s", selection)

    settings.best_dir.mkdir(parents=True, exist_ok=True)
    best_path = settings.best_dir / f"{settings.model_name}.keras"
    shutil.copy2(checkpoint_path(settings, selected_stage), best_path)

    # ---- Reproducibility record ------------------------------------------------------------
    summary = {
        "run_id": run_id,
        "selected_stage": selected_stage,
        "stage1_epochs": epochs_done,
        "stage2_epochs": len(frame2),
        "hyperparameters": {
            "image_size": list(settings.preprocessing.image_size),
            "batch_size": settings.batch_size,
            "optimizer": "Adam",
            "loss": "binary_crossentropy" if settings.is_binary else "categorical_crossentropy",
            "stage1_learning_rate": settings.stage1.learning_rate,
            "stage1_max_epochs": settings.stage1.epochs,
            "stage2_learning_rate": settings.stage2.learning_rate,
            "stage2_max_epochs": settings.stage2.epochs,
            "stage2_unfreeze_top_n_layers": settings.stage2.unfreeze_top_n_layers,
            "dropout": settings.dropout,
            "backbone_weights": settings.backbone_weights,
            "early_stopping": settings.early_stopping,
            "reduce_lr_on_plateau": settings.reduce_lr,
            "checkpoint_monitor": settings.checkpoint.monitor,
            "augmentation": settings.augmentation,
            "seed": settings.seed,
        },
        "class_weight": {settings.class_names[k]: v for k, v in class_weight.items()} if class_weight else None,
        "dataset": {
            **dataset_facts,
            "metadata_file": settings.metadata_file.name,
            "metadata_sha256": _sha256(settings.metadata_file),
            "sequences": list(settings.sequences) if settings.sequences else "all",
        },
        "stage1_freeze_check": stage1_check,
        "stage2_freeze_check": {k: v for k, v in stage2_check.items() if k != "backbone_trainable_layers"},
        "environment": {
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "keras": keras.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "artifacts": {
            "stage1_checkpoint": str(checkpoint_path(settings, 1).relative_to(settings.results_dir.parent)),
            "stage2_checkpoint": str(checkpoint_path(settings, 2).relative_to(settings.results_dir.parent)),
            "best_model": str(best_path.relative_to(settings.results_dir.parent)),
            "final_model": str(final_path.relative_to(settings.results_dir.parent)),
        },
    }
    _write_json(metrics_dir / "training_summary.json", summary)
    _write_json(run_dir / "training_summary.json", summary)
    write_model_metadata(settings, settings.best_dir, {"selected_stage": selected_stage, "training_run": run_id,
                                                       "selection": selection})
    write_model_metadata(settings, settings.final_dir, {"state": "end of stage 2", "training_run": run_id})
    logger.info("Best model: %s | final model: %s", best_path, final_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sanity-check", action="store_true", help="verify data pipeline and model; no training")
    args = parser.parse_args()
    config = load_config()

    if args.sanity_check:
        from src.training.sanity import run_sanity_check

        report = run_sanity_check(config)
        return 0 if report["passed"] else 1

    train(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
