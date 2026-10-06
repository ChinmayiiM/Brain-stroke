"""Training-time data augmentation built from Keras preprocessing layers.

Keras random layers are only active when called with ``training=True``; in
``model.predict`` / ``model.evaluate`` (and when ``training=False``) they are
identity ops. The pipeline is therefore attached to the training graph and
can never leak into validation, test or inference.

Rationale for each transform (all mild — this is medical imaging):
  * RandomFlip("horizontal") — left/right hemispheres are mirror-plausible and
    stroke occurs on either side; vertical flips are anatomically impossible.
  * RandomRotation(±~11°)    — head tilt within the scanner.
  * RandomZoom(±10 %)        — head size / field-of-view differences.
  * RandomTranslation(±5 %)  — head position within the field of view.
  * RandomContrast / RandomBrightness (small) — scanner and windowing variation.
Not used: elastic/shear deformation, colour jitter, noise injection, cutout —
these can fabricate or erase lesion-like structures.
"""

from __future__ import annotations

import keras

# Pixel values entering the model are in [0, 255] (see image_ops.to_model_input).
VALUE_RANGE = (0.0, 255.0)


def build_augmentation_pipeline(config: dict, seed: int | None = None) -> keras.Sequential | None:
    """Build the augmentation ``keras.Sequential`` from ``config['augmentation']``.

    Returns ``None`` when augmentation is disabled.
    """
    aug = config["augmentation"]
    if not aug.get("enabled", False):
        return None

    layers: list[keras.layers.Layer] = []
    if aug.get("horizontal_flip"):
        layers.append(keras.layers.RandomFlip("horizontal", seed=seed))
    if aug.get("rotation"):
        layers.append(keras.layers.RandomRotation(aug["rotation"], fill_mode="constant", fill_value=0.0, seed=seed))
    if aug.get("zoom"):
        layers.append(keras.layers.RandomZoom(aug["zoom"], fill_mode="constant", fill_value=0.0, seed=seed))
    if aug.get("translation"):
        t = aug["translation"]
        layers.append(keras.layers.RandomTranslation(t, t, fill_mode="constant", fill_value=0.0, seed=seed))
    if aug.get("contrast"):
        layers.append(keras.layers.RandomContrast(aug["contrast"], value_range=VALUE_RANGE, seed=seed))
    if aug.get("brightness"):
        layers.append(keras.layers.RandomBrightness(aug["brightness"], value_range=VALUE_RANGE, seed=seed))

    # fill_value=0 matches the black MRI background instead of reflecting anatomy into the corners.
    return keras.Sequential(layers, name="training_augmentation")
