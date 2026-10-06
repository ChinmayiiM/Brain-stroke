"""Class-imbalance handling.

Strategy: **loss re-weighting** computed from the training split only.

Why not oversampling: the minority class (Ischemic) has few images from only
a handful of patients. Duplicating them (or resampling them more often)
makes the network see the same few slices repeatedly, which encourages
memorising those patients rather than learning the pathology. Re-weighting
keeps every image seen once per epoch while making minority errors cost more.
"""

from __future__ import annotations

import numpy as np
from sklearn.utils.class_weight import compute_class_weight

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_class_weights(train_labels: np.ndarray, class_names: tuple[str, ...], method: str | None) -> dict[int, float] | None:
    """Return ``{class_index: weight}`` for ``model.fit(class_weight=...)``, or None.

    ``balanced``: w_c = n_samples / (n_classes * n_c), so every class contributes
    equally to the total loss.
    """
    if method is None:
        return None
    if method != "balanced":
        raise ValueError(f"Unsupported class_weighting {method!r}; use 'balanced' or null")

    classes = np.arange(len(class_names))
    present = np.unique(train_labels)
    if not np.array_equal(present, classes):
        raise ValueError(f"Training split is missing classes: expected {classes.tolist()}, found {present.tolist()}")

    weights = compute_class_weight("balanced", classes=classes, y=train_labels)
    result = {int(c): float(w) for c, w in zip(classes, weights)}
    for c, w in result.items():
        logger.info("class weight %-14s n=%4d  w=%.3f", class_names[c], int((train_labels == c).sum()), w)
    return result
