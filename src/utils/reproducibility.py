"""Seeding helpers for reproducible experiments."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) TensorFlow random generators.

    Note: full GPU determinism additionally requires
    ``tf.config.experimental.enable_op_determinism()``, which slows training;
    it is intentionally not enabled here.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.random.set_seed(seed)
