"""Image-file validation for inference — Pillow only, no TensorFlow import.

Kept separate from :mod:`src.inference.predictor` so that user interfaces can
validate and preview an upload without paying the TensorFlow import cost.
``predictor`` re-exports these names, so existing imports keep working.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.inference.errors import ImageValidationError

# PIL modes that convert to RGB without losing the intensity scale used in training (8-bit).
SUPPORTED_MODES = {"1", "L", "LA", "P", "RGB", "RGBA", "CMYK", "YCbCr"}


@dataclass(frozen=True)
class ImageInfo:
    """Facts about a validated image (used e.g. for UI previews)."""

    format: str | None
    mode: str
    width: int
    height: int


def _check_extension(name: str, allowed_extensions: tuple[str, ...]) -> None:
    suffix = Path(name).suffix.lower()
    if suffix not in allowed_extensions:
        raise ImageValidationError(
            f"Unsupported file type '{suffix or '(none)'}'. Supported: {', '.join(allowed_extensions)}"
        )


def _inspect_image(source: Path | bytes, label: str) -> ImageInfo:
    """Decode-level checks shared by file and in-memory inputs."""
    def _open():
        return Image.open(io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source)

    try:
        with _open() as probe:
            probe.verify()  # detects truncation / corruption
        with _open() as img:
            info = ImageInfo(img.format, img.mode, *img.size)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError(f"Cannot read image (corrupted or not an image): {label} ({exc})") from exc
    if info.mode not in SUPPORTED_MODES:
        raise ImageValidationError(
            f"Unsupported pixel format '{info.mode}' (e.g. 16-bit or float). The model was trained on "
            "8-bit images; export the MRI slice as 8-bit grayscale/RGB PNG or JPEG."
        )
    if min(info.width, info.height) <= 0:
        raise ImageValidationError(f"Invalid image dimensions {info.width}x{info.height}: {label}")
    return info


def validate_image_file(path: str | Path, allowed_extensions: tuple[str, ...]) -> Path:
    """File-level checks before decoding. Raises :class:`ImageValidationError`."""
    image_path = Path(path)
    if not image_path.exists():
        raise ImageValidationError(f"Image not found: {image_path}")
    if not image_path.is_file():
        raise ImageValidationError(f"Not a file: {image_path}")
    _check_extension(image_path.name, allowed_extensions)
    if image_path.stat().st_size == 0:
        raise ImageValidationError(f"Image file is empty: {image_path}")
    _inspect_image(image_path, str(image_path))
    return image_path


def validate_image_bytes(data: bytes, filename: str, allowed_extensions: tuple[str, ...]) -> ImageInfo:
    """Same checks as :func:`validate_image_file` for in-memory uploads (e.g. a web form)."""
    _check_extension(filename, allowed_extensions)
    if not data:
        raise ImageValidationError(f"Uploaded file is empty: {filename}")
    return _inspect_image(data, filename)
