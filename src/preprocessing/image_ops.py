"""Single source of truth for turning an MRI image into an EfficientNetB0 input.

Used by both the training data pipeline and (later) the inference/app code,
so an image is processed identically everywhere.

Deliberately NOT done here (they could alter clinically relevant intensity
information or are handled elsewhere):
  * histogram equalisation / CLAHE, denoising, sharpening, skull stripping
  * division by 255 — Keras EfficientNet normalises its own input
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

_INTERPOLATION = {
    "area": cv2.INTER_AREA,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
}


@dataclass(frozen=True)
class PreprocessingConfig:
    image_size: tuple[int, int]  # (height, width)
    channels: int = 3
    interpolation: str = "area"
    min_dimension: int = 64

    @classmethod
    def from_config(cls, config: dict) -> "PreprocessingConfig":
        prep = config["preprocessing"]
        return cls(
            image_size=(int(prep["image_size"][0]), int(prep["image_size"][1])),
            channels=int(prep["channels"]),
            interpolation=str(prep.get("interpolation", "area")),
            min_dimension=int(config.get("cleaning", {}).get("min_dimension", 64)),
        )


def validate_image(image: np.ndarray, min_dimension: int) -> None:
    """Reject arrays that cannot be a usable MRI slice.

    Raises:
        ValueError: wrong rank/channels, too small, or constant (blank) image.
    """
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[-1] not in (1, 3, 4)):
        raise ValueError(f"Unsupported image shape {image.shape}")
    height, width = image.shape[:2]
    if min(height, width) < min_dimension:
        raise ValueError(f"Image too small: {width}x{height} (minimum {min_dimension}px)")
    if image.size == 0 or int(image.max()) == int(image.min()):
        raise ValueError("Image is blank (constant intensity)")


def load_image_rgb(source: str | Path | bytes) -> np.ndarray:
    """Decode an image file (path or raw bytes) to an RGB ``uint8`` array of shape (H, W, 3).

    Grayscale MRI is replicated across the three channels (required for
    ImageNet-pretrained weights). EXIF orientation is honoured; alpha is dropped.

    Raises:
        ValueError: if the data cannot be decoded as an image.
    """
    try:
        handle = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else Path(source)
        with Image.open(handle) as img:
            img = ImageOps.exif_transpose(img)
            return np.asarray(img.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError, SyntaxError) as exc:
        raise ValueError(f"Could not decode image: {exc}") from exc


def resize_image(image: np.ndarray, size: tuple[int, int], interpolation: str = "area") -> np.ndarray:
    """Resize to ``size`` = (height, width).

    A direct resize is used (no padding/cropping): every source image in the
    selected dataset is square, so aspect ratio is preserved.
    """
    height, width = size
    if image.shape[:2] == (height, width):
        return image
    return cv2.resize(image, (width, height), interpolation=_INTERPOLATION[interpolation])


def to_model_input(image: np.ndarray) -> np.ndarray:
    """Convert a uint8 RGB image to the float32 array EfficientNetB0 expects.

    Values stay in the [0, 255] range: ``keras.applications.EfficientNetB0``
    contains its own Rescaling + Normalization layers, and
    ``keras.applications.efficientnet.preprocess_input`` is an identity
    pass-through for this reason.
    """
    return image.astype(np.float32)


def preprocess_image(source: str | Path | bytes | np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Full pipeline: load → validate → RGB → resize → float32. Returns shape (H, W, 3)."""
    image = source if isinstance(source, np.ndarray) else load_image_rgb(source)
    validate_image(image, config.min_dimension)
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    resized = resize_image(image, config.image_size, config.interpolation)
    return to_model_input(resized)


def preprocess_batch(sources: list[str | Path | bytes], config: PreprocessingConfig) -> np.ndarray:
    """Preprocess several images into one (N, H, W, 3) float32 batch."""
    return np.stack([preprocess_image(source, config) for source in sources], axis=0)
