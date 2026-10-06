"""Pre-training sanity check on real data — verifies the pipeline end to end without training.

Checks: data loading, batch/tensor shapes, pixel ranges, label correctness,
train-only augmentation, class weights, model construction, freeze state for
both stages (including that one optimisation step changes ONLY the intended
weights), forward pass, output shape and loss compatibility.

The test split is counted from metadata only; its images are never loaded.
Nothing is saved to models/.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from src.preprocessing.image_ops import preprocess_image
from src.training.class_balance import compute_class_weights
from src.training.data_loader import build_dataset, get_split, load_metadata
from src.training.inspection import model_summary_text, save_summary, verify_stage1, verify_stage2
from src.training.model import build_model, compile_model, get_backbone, unfreeze_top_layers
from src.training.settings import TrainingSettings
from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger("sanity_check")


class SanityCheckError(AssertionError):
    pass


def _check(condition: bool, message: str, results: list[dict]) -> None:
    results.append({"check": message, "passed": bool(condition)})
    logger.info("[%s] %s", "PASS" if condition else "FAIL", message)
    if not condition:
        raise SanityCheckError(message)


def _snapshot(weights) -> list[np.ndarray]:
    return [np.array(w) for w in weights]


def _changed(before: list[np.ndarray], after_weights) -> list[bool]:
    return [not np.array_equal(b, np.array(a)) for b, a in zip(before, after_weights)]


def run_sanity_check(config: dict) -> dict[str, Any]:
    settings = TrainingSettings.from_config(config)
    set_global_seed(settings.seed)
    results: list[dict] = []
    report: dict[str, Any] = {"checks": results}
    out_dir = settings.model_reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ---- Data ---------------------------------------------------------------------
        metadata = load_metadata(settings)
        train_spec = get_split(metadata, "train", settings)
        val_spec = get_split(metadata, "validation", settings)
        test_count = int((metadata["split"] == "test").sum())  # counted, never loaded
        report["split_sizes"] = {"train": train_spec.size, "validation": val_spec.size, "test (not loaded)": test_count}
        report["class_names"] = list(settings.class_names)
        _check(train_spec.size > 0 and val_spec.size > 0 and test_count > 0, "all three splits are non-empty", results)

        train_ds = build_dataset(train_spec, settings, training=True)
        val_ds = build_dataset(val_spec, settings, training=False)
        x_train, y_train = (np.asarray(t) for t in next(iter(train_ds)))
        x_val, y_val = (np.asarray(t) for t in next(iter(val_ds)))
        expected_batch = min(settings.batch_size, val_spec.size)
        report["train_batch"] = {"x_shape": list(x_train.shape), "x_dtype": str(x_train.dtype),
                                 "y_shape": list(y_train.shape),
                                 "x_min": float(x_train.min()), "x_max": float(x_train.max()),
                                 "x_mean": round(float(x_train.mean()), 2)}
        report["validation_batch"] = {"x_shape": list(x_val.shape), "y_shape": list(y_val.shape),
                                      "x_min": float(x_val.min()), "x_max": float(x_val.max()),
                                      "x_mean": round(float(x_val.mean()), 2)}

        _check(x_train.shape[1:] == settings.input_shape, f"image tensor shape is (B, {settings.input_shape})", results)
        _check(x_val.shape[0] == expected_batch, f"validation batch size is {expected_batch}", results)
        _check(x_train.dtype == np.float32, "images are float32", results)
        _check(0.0 <= x_train.min() and x_train.max() <= 255.0 and x_train.max() > 1.0,
               "pixel range is [0, 255] (not pre-scaled to [0, 1])", results)
        label_width = 1 if settings.is_binary else settings.num_classes
        _check(y_train.shape[1:] == (label_width,), f"label shape is (B, {label_width})", results)
        if not settings.is_binary:
            _check(np.allclose(y_train.sum(axis=1), 1.0) and np.allclose(y_val.sum(axis=1), 1.0),
                   "labels are valid one-hot vectors", results)
            val_labels = y_val.argmax(axis=1)
        else:
            val_labels = y_val[:, 0].astype(int)
        _check(np.array_equal(val_labels, val_spec.labels[: len(val_labels)]),
               "validation labels match metadata order (no shuffling)", results)
        reference = np.stack([preprocess_image(p, settings.preprocessing) for p in val_spec.paths[: len(x_val)]])
        _check(np.array_equal(reference, x_val),
               "validation images equal shared preprocess_image output (no augmentation)", results)
        val_again = np.asarray(next(iter(val_ds))[0])
        _check(np.array_equal(val_again, x_val), "validation pipeline is deterministic across iterations", results)

        class_weight = compute_class_weights(train_spec.labels, settings.class_names, settings.class_weighting)
        report["class_weight"] = (
            {settings.class_names[k]: round(v, 4) for k, v in class_weight.items()} if class_weight else None
        )
        report["train_class_counts"] = {
            name: int((train_spec.labels == i).sum()) for i, name in enumerate(settings.class_names)
        }

        # ---- Stage 1 ------------------------------------------------------------------
        model = build_model(settings)
        stage1 = verify_stage1(model)
        report["stage1"] = stage1
        _check(True, "stage 1: backbone frozen, only head trainable", results)
        compile_model(model, settings, settings.stage1.learning_rate)
        _check(model.optimizer is not None and model.loss is not None, "stage 1 compilation succeeded", results)
        save_summary(model, out_dir / "model_summary_stage1.txt")
        (out_dir / "backbone_summary.txt").write_text(model_summary_text(get_backbone(model)), encoding="utf-8")

        probs = np.asarray(model(x_val, training=False))
        report["forward_pass"] = {"output_shape": list(probs.shape),
                                  "output_min": float(probs.min()), "output_max": float(probs.max())}
        _check(probs.shape == (len(x_val), label_width), f"output shape is (B, {label_width})", results)
        if not settings.is_binary:
            _check(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "softmax outputs sum to 1", results)

        loss_and_metrics = model.test_on_batch(x_val, y_val, return_dict=True)
        report["untrained_validation_batch_loss"] = float(loss_and_metrics["loss"])
        _check(np.isfinite(loss_and_metrics["loss"]), "loss is finite and compatible with labels", results)

        backbone_before = _snapshot(get_backbone(model).weights)
        head_before = _snapshot(model.get_layer("classifier").weights)
        model.train_on_batch(x_train, y_train, class_weight=class_weight)
        _check(not any(_changed(backbone_before, get_backbone(model).weights)),
               "stage 1 step: no backbone weight (incl. BatchNorm stats) changed", results)
        _check(all(_changed(head_before, model.get_layer("classifier").weights)),
               "stage 1 step: classifier weights updated", results)

        # ---- Stage 2 ------------------------------------------------------------------
        n_layers = settings.stage2.unfreeze_top_n_layers
        unfreeze_top_layers(model, n_layers)
        stage2 = verify_stage2(model, n_layers)
        report["stage2"] = stage2
        _check(True, f"stage 2: exactly the top {n_layers} backbone layers (minus BatchNorm) trainable", results)
        compile_model(model, settings, settings.stage2.learning_rate)
        save_summary(model, out_dir / "model_summary_stage2.txt")

        backbone = get_backbone(model)
        trainable_ids = {id(w) for w in backbone.trainable_weights}
        frozen = [w for w in backbone.weights if id(w) not in trainable_ids]
        frozen_before = _snapshot(frozen)
        trainable_before = _snapshot(backbone.trainable_weights)
        model.train_on_batch(x_train, y_train, class_weight=class_weight)
        _check(not any(_changed(frozen_before, frozen)),
               "stage 2 step: frozen backbone weights and all BatchNorm statistics unchanged", results)
        _check(any(_changed(trainable_before, backbone.trainable_weights)),
               "stage 2 step: unfrozen backbone weights updated", results)

        report["passed"] = True
    except SanityCheckError:
        report["passed"] = False
    finally:
        (out_dir / "sanity_check.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    logger.info("Sanity check %s — report: %s", "PASSED" if report["passed"] else "FAILED", out_dir / "sanity_check.json")
    return report
