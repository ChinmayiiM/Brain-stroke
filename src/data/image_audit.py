"""Per-file integrity and property checks (readability, dimensions, format, hashes)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from src.utils.logger import get_logger

logger = get_logger(__name__)

AUDIT_COLUMNS = [
    "file_exists",
    "file_bytes",
    "readable",
    "read_error",
    "image_format",
    "image_mode",
    "image_width",
    "image_height",
    "is_grayscale_content",
    "file_sha256",
    "pixel_sha256",
    "phash",
]


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of the raw file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(gray: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> int:
    """DCT perceptual hash (pHash) of a grayscale image as a 64-bit integer.

    Robust to re-encoding, resizing and small intensity changes, so it finds
    near-duplicates that exact hashing misses.
    """
    size = hash_size * highfreq_factor
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)[:hash_size, :hash_size]
    bits = (dct > np.median(dct)).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def audit_image(path: Path) -> dict:
    """Inspect a single image file without modifying it."""
    result: dict = {column: None for column in AUDIT_COLUMNS}
    result["file_exists"] = path.is_file()
    result["readable"] = False
    if not result["file_exists"]:
        result["read_error"] = "missing file"
        return result

    result["file_bytes"] = path.stat().st_size
    result["file_sha256"] = file_sha256(path)
    try:
        # verify() detects truncation/corruption but invalidates the handle, so reopen to decode.
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as img:
            result["image_format"] = img.format
            result["image_mode"] = img.mode
            result["image_width"], result["image_height"] = img.size
            img.load()
            rgb = np.asarray(img.convert("RGB"))
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        result["read_error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["readable"] = True
    result["is_grayscale_content"] = bool(
        np.array_equal(rgb[..., 0], rgb[..., 1]) and np.array_equal(rgb[..., 1], rgb[..., 2])
    )
    result["pixel_sha256"] = hashlib.sha256(rgb.tobytes() + str(rgb.shape).encode()).hexdigest()
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    result["phash"] = perceptual_hash(gray)
    return result


def audit_images(records: pd.DataFrame, path_column: str = "source_path") -> pd.DataFrame:
    """Run :func:`audit_image` for every record and append the audit columns."""
    audits = [audit_image(Path(p)) for p in records[path_column]]
    audit_df = pd.DataFrame(audits, columns=AUDIT_COLUMNS, index=records.index)
    unreadable = int((~audit_df["readable"]).sum())
    logger.info("Audited %d images (%d unreadable)", len(audit_df), unreadable)
    return pd.concat([records, audit_df], axis=1)
