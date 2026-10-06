"""Programmatic inspection of model parameters and freezing state."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np

from src.training.model import HEAD_LAYER_NAMES, expected_unfrozen_layers, get_backbone


class FreezeVerificationError(AssertionError):
    """Raised when the trainable state differs from the intended transfer-learning stage."""


def _count(weights) -> int:
    return int(sum(np.prod(w.shape) for w in weights))


def parameter_counts(model: keras.Model) -> dict[str, int]:
    backbone = get_backbone(model)
    return {
        "total": _count(model.weights),
        "trainable": _count(model.trainable_weights),
        "non_trainable": _count(model.non_trainable_weights),
        "backbone_total": _count(backbone.weights),
        "backbone_trainable": _count(backbone.trainable_weights),
        "head_trainable": sum(_count(model.get_layer(n).trainable_weights) for n in HEAD_LAYER_NAMES),
    }


def model_summary_text(model: keras.Model, expand_nested: bool = False) -> str:
    """Full-width Keras summary as text (print_fn avoids terminal-width truncation)."""
    lines: list[str] = []
    model.summary(
        expand_nested=expand_nested,
        show_trainable=True,
        line_length=120,
        print_fn=lambda line, *args, **kwargs: lines.append(line),
    )
    return "\n".join(lines)


def save_summary(model: keras.Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = model_summary_text(model) + "\n\nParameter counts:\n"
    text += "\n".join(f"  {k:<20} {v:>12,}" for k, v in parameter_counts(model).items())
    path.write_text(text, encoding="utf-8")


def trainable_backbone_layers(model: keras.Model) -> set[str]:
    return {layer.name for layer in get_backbone(model).layers if layer.trainable_weights}


def verify_stage1(model: keras.Model) -> dict:
    """Backbone fully frozen; only the head (classifier) is trainable."""
    counts = parameter_counts(model)
    unexpected = trainable_backbone_layers(model)
    if unexpected or counts["backbone_trainable"] != 0:
        raise FreezeVerificationError(f"Stage 1: backbone layers trainable: {sorted(unexpected)[:10]}")
    if counts["trainable"] != counts["head_trainable"] or counts["trainable"] == 0:
        raise FreezeVerificationError(f"Stage 1: trainable params {counts['trainable']} != head {counts['head_trainable']}")
    return {"stage": 1, "backbone_trainable_layers": 0, **counts}


def verify_stage2(model: keras.Model, n_layers: int) -> dict:
    """Exactly the intended top layers (minus BatchNorm) are trainable."""
    expected = expected_unfrozen_layers(model, n_layers)
    actual = trainable_backbone_layers(model)
    if actual != expected:
        raise FreezeVerificationError(
            f"Stage 2: unexpected trainable layers {sorted(actual - expected)}, missing {sorted(expected - actual)}"
        )
    bn_trainable = [
        layer.name for layer in get_backbone(model).layers
        if isinstance(layer, keras.layers.BatchNormalization) and layer.trainable_weights
    ]
    if bn_trainable:
        raise FreezeVerificationError(f"Stage 2: BatchNorm layers must stay frozen: {bn_trainable}")
    counts = parameter_counts(model)
    if counts["trainable"] != counts["head_trainable"] + counts["backbone_trainable"]:
        raise FreezeVerificationError("Stage 2: trainable parameter accounting mismatch")
    return {
        "stage": 2,
        "unfreeze_top_n_layers": n_layers,
        "backbone_trainable_layers": sorted(actual),
        "backbone_layers_total": len(get_backbone(model).layers),
        **counts,
    }
