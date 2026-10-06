"""Run a trained model over one split and tabulate its real predictions."""

from __future__ import annotations

from pathlib import Path

import keras
import numpy as np
import pandas as pd

from src.training.data_loader import SplitSpec, build_dataset
from src.training.settings import TrainingSettings


def load_trained_model(path: Path) -> keras.Model:
    """Load a saved ``.keras`` model for inference (no optimizer state needed)."""
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return keras.models.load_model(path, compile=False)


def predict_split(model: keras.Model, spec: SplitSpec, settings: TrainingSettings) -> pd.DataFrame:
    """Predict every image of ``spec`` (ordered, unaugmented pipeline).

    Returns one row per image with the true label, predicted label, the
    predicted-class probability (``confidence``) and every class probability.
    """
    dataset = build_dataset(spec, settings, training=False)
    probabilities = np.asarray(model.predict(dataset, verbose=0), dtype=np.float64)
    if settings.is_binary:  # sigmoid → two-column probability matrix
        probabilities = np.column_stack([1.0 - probabilities[:, 0], probabilities[:, 0]])
    if probabilities.shape != (spec.size, settings.num_classes):
        raise ValueError(f"Unexpected prediction shape {probabilities.shape}")
    # float32 softmax rows can deviate from 1 by ~1e-7; renormalise in float64 so
    # probabilistic metrics (log-loss) receive valid distributions. Argmax is unchanged.
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

    names = np.asarray(settings.class_names)
    predicted = probabilities.argmax(axis=1)
    frame = spec.frame
    result = pd.DataFrame(
        {
            "image_path": frame["image_path"],
            "patient_id": frame["patient_id"],
            "sequence": frame["sequence"],
            "actual_index": spec.labels,
            "actual": names[spec.labels],
            "predicted_index": predicted,
            "predicted": names[predicted],
            "confidence": probabilities.max(axis=1),
            "correct": predicted == spec.labels,
        }
    )
    for index, name in enumerate(settings.class_names):
        result[f"prob_{name}"] = probabilities[:, index]
    return result


def probability_matrix(predictions: pd.DataFrame, class_names: tuple[str, ...]) -> np.ndarray:
    return predictions[[f"prob_{name}" for name in class_names]].to_numpy()
