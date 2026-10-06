"""Load the trained model once and verify it matches the inference configuration.

The model's ``model_metadata.json`` (written by Phase 4 training) is the
contract: class order, input shape and preprocessing. Loading fails loudly if
the model, its metadata and ``config.yaml`` disagree.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import keras
import tensorflow as tf

from src.inference.errors import (
    ClassMappingError,
    ModelLoadError,
    ModelNotFoundError,
    PreprocessingMismatchError,
)
from src.preprocessing.image_ops import PreprocessingConfig
from src.utils.config import resolve_path
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXPECTED_PREPROCESSING_PIPELINE = "src.preprocessing.image_ops.preprocess_image"


@dataclass(frozen=True)
class InferenceSettings:
    """Everything inference reads from ``config.yaml`` (no magic numbers elsewhere)."""

    model_path: Path
    metadata_file: str
    class_names: tuple[str, ...]
    preprocessing: PreprocessingConfig
    allowed_extensions: tuple[str, ...]
    batch_output_dir: Path
    probability_tolerance: float
    disclaimer: str

    @classmethod
    def from_config(cls, config: dict, model_path: str | Path | None = None) -> "InferenceSettings":
        inference = config["inference"]
        return cls(
            model_path=resolve_path(model_path or inference["model_path"]),
            metadata_file=str(inference.get("metadata_file", "model_metadata.json")),
            class_names=tuple(config["data"]["class_names"]),
            preprocessing=PreprocessingConfig.from_config(config),
            allowed_extensions=tuple(e.lower() for e in config["data"]["allowed_extensions"]),
            batch_output_dir=resolve_path(inference.get("batch_output_dir", "results/inference")),
            probability_tolerance=float(inference.get("probability_tolerance", 1e-4)),
            disclaimer=str(inference["disclaimer"]),
        )


@dataclass(frozen=True)
class LoadedModel:
    model: keras.Model
    path: Path
    metadata: dict[str, Any]
    class_names: tuple[str, ...]
    output_kind: str  # "softmax" or "sigmoid"
    infer: Callable  # compiled forward pass (training=False), shared by every prediction

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


_CACHE: dict[tuple[str, float], tuple[keras.Model, dict[str, Any], Callable]] = {}


def _compile_forward_pass(model: keras.Model) -> Callable:
    """Wrap ``model(x, training=False)`` in a ``tf.function`` with a fixed input signature.

    Graph execution gives the same outputs as eager Keras calls (measured difference ≤ 1.2e-7)
    but avoids per-layer Python overhead: ~150 ms instead of ~1.3 s per image on this CPU.
    It is traced once on first use (see ``Predictor.warm_up``) and reused afterwards.
    """
    spec = tf.TensorSpec(shape=(None, *model.input_shape[1:]), dtype=tf.float32)

    @tf.function(input_signature=[spec], reduce_retracing=True)
    def forward(images):
        return model(images, training=False)

    return forward


def _all_layers(model: keras.Model) -> list[keras.layers.Layer]:
    layers: list[keras.layers.Layer] = []
    for layer in model.layers:
        layers.append(layer)
        if isinstance(layer, keras.Model):
            layers.extend(_all_layers(layer))
    return layers


def _read_metadata(model_path: Path, metadata_file: str) -> dict[str, Any]:
    path = model_path.parent / metadata_file
    if not path.is_file():
        raise ModelNotFoundError(
            f"Model metadata not found: {path}. It is written by `python -m src.training.train` "
            "next to the model and is required for the class mapping."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelLoadError(f"Model metadata is not valid JSON: {path} ({exc})") from exc


def _verify(model: keras.Model, metadata: dict, settings: InferenceSettings) -> tuple[tuple[str, ...], str]:
    """Cross-check model ↔ metadata ↔ config. Returns (class_names, output_kind)."""
    # Class mapping
    meta_classes = tuple(metadata.get("class_names") or ())
    if not meta_classes:
        raise ClassMappingError("Model metadata has no 'class_names'.")
    if meta_classes != settings.class_names:
        raise ClassMappingError(
            f"Class order mismatch: model was trained with {list(meta_classes)}, "
            f"config data.class_names is {list(settings.class_names)}."
        )

    # Output layer
    units = int(model.output_shape[-1])
    activation = getattr(getattr(model.layers[-1], "activation", None), "__name__", None)
    if units == len(meta_classes) and activation == "softmax":
        output_kind = "softmax"
    elif units == 1 and len(meta_classes) == 2 and activation == "sigmoid":
        output_kind = "sigmoid"
    else:
        raise ClassMappingError(
            f"Model output ({units} units, activation={activation}) is incompatible with "
            f"{len(meta_classes)} classes {list(meta_classes)}."
        )

    # Input shape / preprocessing contract
    expected_shape = (*settings.preprocessing.image_size, settings.preprocessing.channels)
    model_shape = tuple(int(d) for d in model.input_shape[1:])
    if model_shape != expected_shape or tuple(metadata.get("input_shape", ())) != expected_shape:
        raise PreprocessingMismatchError(
            f"Input shape mismatch: model {model_shape}, metadata {metadata.get('input_shape')}, "
            f"config preprocessing {expected_shape}."
        )
    prep = metadata.get("preprocessing", {})
    if prep.get("pipeline") != EXPECTED_PREPROCESSING_PIPELINE:
        raise PreprocessingMismatchError(f"Model was trained with preprocessing {prep.get('pipeline')!r}.")
    if prep.get("resize_interpolation") != settings.preprocessing.interpolation:
        raise PreprocessingMismatchError(
            f"Resize interpolation mismatch: trained with {prep.get('resize_interpolation')!r}, "
            f"config uses {settings.preprocessing.interpolation!r}."
        )

    # Training-only augmentation must not be part of the inference graph.
    random_layers = [layer.name for layer in _all_layers(model) if type(layer).__name__.startswith("Random")]
    if random_layers:
        raise PreprocessingMismatchError(f"Model contains random augmentation layers: {random_layers}")
    return meta_classes, output_kind


def load_inference_model(settings: InferenceSettings) -> LoadedModel:
    """Load (or reuse) the model at ``settings.model_path`` and verify it.

    The cache key includes the file's modification time, so a retrained model
    is picked up automatically while repeated predictions reuse one instance.
    """
    path = settings.model_path
    if not path.is_file():
        raise ModelNotFoundError(
            f"Trained model not found: {path}. Train it with `python -m src.training.train` "
            "or pass --model with the path to a .keras file."
        )
    if path.suffix != ".keras":
        raise ModelLoadError(f"Expected a native Keras '.keras' file, got: {path.name}")

    key = (str(path.resolve()), path.stat().st_mtime)
    if key not in _CACHE:
        metadata = _read_metadata(path, settings.metadata_file)
        start = time.perf_counter()
        try:
            model = keras.models.load_model(path, compile=False)
        except Exception as exc:  # noqa: BLE001 — any deserialisation failure is reported uniformly
            raise ModelLoadError(f"Could not load model {path}: {type(exc).__name__}: {exc}") from exc
        _CACHE[key] = (model, metadata, _compile_forward_pass(model))
        logger.info("Loaded model %s in %.2f s", path, time.perf_counter() - start)
    model, metadata, infer = _CACHE[key]

    # Verification runs on every call (cheap), so a cached model is never used with
    # settings it has not been checked against.
    class_names, output_kind = _verify(model, metadata, settings)
    return LoadedModel(model, path, metadata, class_names, output_kind, infer)


def clear_model_cache() -> None:
    _CACHE.clear()
