"""tf.data input pipelines built from ``dataset/metadata/dataset_metadata.csv``.

Per split:

    metadata rows ─► (path, label) ─► decode/validate/resize (shared image_ops)
        ─► [cache] ─► [shuffle: train only] ─► batch
        ─► [augment: train only] ─► prefetch

Image decoding reuses ``src.preprocessing.image_ops.preprocess_image`` through
``tf.numpy_function`` so training and inference share one implementation
(no duplicated preprocessing logic in TensorFlow ops).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf

from src.preprocessing.augmentation import build_augmentation_pipeline
from src.preprocessing.image_ops import preprocess_image
from src.training.settings import TrainingSettings
from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class SplitSpec:
    """The rows of one split after filtering, plus derived facts."""

    name: str
    paths: tuple[str, ...]
    labels: np.ndarray  # integer class indices
    frame: pd.DataFrame

    @property
    def size(self) -> int:
        return len(self.paths)


def load_metadata(settings: TrainingSettings) -> pd.DataFrame:
    """Read and validate the Phase 2 metadata against the configured class names."""
    if not settings.metadata_file.is_file():
        raise FileNotFoundError(
            f"Metadata not found: {settings.metadata_file}. Run `python scripts/prepare_dataset.py` first."
        )
    metadata = pd.read_csv(settings.metadata_file)

    unknown = sorted(set(metadata["class_name"]) - set(settings.class_names))
    if unknown:
        raise ValueError(f"Metadata contains classes not in config: {unknown}")
    expected_index = metadata["class_name"].map({name: i for i, name in enumerate(settings.class_names)})
    if not (expected_index == metadata["label"]).all():
        raise ValueError("Metadata label indices do not match the order of config data.class_names")

    if settings.sequences:
        before = len(metadata)
        metadata = metadata[metadata["sequence"].isin(settings.sequences)]
        logger.info("Sequence filter %s: %d -> %d images", list(settings.sequences), before, len(metadata))
    return metadata.reset_index(drop=True)


def get_split(metadata: pd.DataFrame, split: str, settings: TrainingSettings) -> SplitSpec:
    """Select one split; fail loudly if a class is missing from it."""
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    frame = metadata[metadata["split"] == split].sort_values("image_path").reset_index(drop=True)
    missing = sorted(set(settings.class_names) - set(frame["class_name"]))
    if missing:
        raise ValueError(f"Split {split!r} has no images for classes {missing}")
    paths = tuple(str(PROJECT_ROOT / p) for p in frame["image_path"])
    return SplitSpec(split, paths, frame["label"].to_numpy(dtype=np.int32), frame)


def _encode_labels(labels: tf.Tensor, settings: TrainingSettings) -> tf.Tensor:
    """Binary → float scalar (sigmoid head); multi-class → one-hot (softmax head)."""
    if settings.is_binary:
        return tf.reshape(tf.cast(labels, tf.float32), [1])  # matches Dense(1) output
    return tf.one_hot(labels, settings.num_classes, dtype=tf.float32)


def _estimated_cache_mb(spec: SplitSpec, settings: TrainingSettings) -> float:
    height, width, channels = settings.input_shape
    return spec.size * height * width * channels * 4 / 1e6  # float32


def build_dataset(spec: SplitSpec, settings: TrainingSettings, training: bool) -> tf.data.Dataset:
    """Create the tf.data pipeline for one split.

    Args:
        spec: rows of the split (from :func:`get_split`).
        settings: training settings.
        training: True only for the training split — enables shuffling and augmentation.
    """
    prep = settings.preprocessing
    input_shape = settings.input_shape

    def _load(path: bytes) -> np.ndarray:
        return preprocess_image(path.decode("utf-8"), prep)

    def _load_tf(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.numpy_function(_load, [path], tf.float32, stateful=False)
        image = tf.ensure_shape(image, input_shape)
        return image, _encode_labels(label, settings)

    ds = tf.data.Dataset.from_tensor_slices((list(spec.paths), spec.labels))
    ds = ds.map(_load_tf, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    cache_mb = _estimated_cache_mb(spec, settings)
    if settings.cache and cache_mb <= settings.cache_max_mb:
        ds = ds.cache()  # decoded once; later epochs read from memory
        logger.info("%s: caching %d images in memory (~%.0f MB)", spec.name, spec.size, cache_mb)
    elif settings.cache:
        logger.info("%s: not cached (~%.0f MB > cache_max_mb=%.0f)", spec.name, cache_mb, settings.cache_max_mb)

    if training:
        buffer = settings.shuffle_buffer or spec.size
        ds = ds.shuffle(buffer, seed=settings.seed, reshuffle_each_iteration=True)

    ds = ds.batch(settings.batch_size, drop_remainder=False)

    if training:
        augmenter = build_augmentation_pipeline({"augmentation": settings.augmentation}, seed=settings.seed)
        if augmenter is not None:
            ds = ds.map(lambda x, y: (augmenter(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)
