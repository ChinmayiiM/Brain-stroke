"""Occlusion-sensitivity view: which MRI area the trained classifier's STROKE score depends on.

This is MODEL INTERPRETABILITY, not lesion localisation. The dataset has no masks,
boxes or lesion coordinates, and the model outputs only class probabilities, so
nothing here can say where a lesion is. It answers a narrower, checkable question:

    "If this area of the image is hidden, how much does the model's P(STROKE) fall?"

Method (the existing model and preprocessing, unchanged; no gradients, no retraining):

    preprocessed image (same function as training/prediction)
      ─► slide a black square patch over it (``patch_size`` / ``stride`` from config)
      ─► run the existing model on every occluded copy (batched)
      ─► drop = P(STROKE | original) − P(STROKE | occluded), averaged per pixel
      ─► smooth the drop map, resize it to the uploaded image, keep the head area
      ─► region = connected area with drop ≥ ``threshold`` × max drop
         (shown only if hiding one patch lowers P(STROKE) by ≥ ``min_drop``)

P(STROKE) is the same sum of positive-class probabilities as the binary view.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy import ndimage

from src.preprocessing.image_ops import load_image_rgb
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.inference.binary_view import BinaryViewConfig
    from src.inference.predictor import Predictor

logger = get_logger(__name__)

REGION_RGB = (205, 45, 45)  # outline / fill colour of the highlighted area


@dataclass(frozen=True)
class RegionViewConfig:
    patch_size: int
    stride: int
    batch_size: int
    threshold: float  # fraction of the strongest drop that still counts as part of the region
    min_drop: float  # below this, no area mattered enough to highlight
    fill_alpha: float
    head_intensity: int  # grey level separating the head from the black background

    @classmethod
    def from_config(cls, config: dict) -> "RegionViewConfig":
        view = config["inference"]["region_view"]
        return cls(
            patch_size=int(view["patch_size"]),
            stride=int(view["stride"]),
            batch_size=int(view["batch_size"]),
            threshold=float(view["threshold"]),
            min_drop=float(view["min_drop"]),
            fill_alpha=float(view["fill_alpha"]),
            head_intensity=int(view["head_intensity"]),
        )


@dataclass(frozen=True)
class RegionResult:
    overlay_png: bytes | None  # uploaded MRI with the area outlined; None if no area mattered enough
    stroke_probability: float  # P(STROKE) for the unmodified image
    occluded_probability: float  # P(STROKE) with the single most influential patch hidden
    seconds: float

    @property
    def max_drop(self) -> float:
        return self.stroke_probability - self.occluded_probability


def occlusion_map(predictor: "Predictor", image: np.ndarray, positive: list[int],
                  cfg: RegionViewConfig) -> tuple[float, np.ndarray, float]:
    """Return (P(STROKE), per-pixel mean drop map in model-input space, largest single-patch drop)."""
    height, width = image.shape[:2]
    base = float(predictor._run_model(image[None, ...])[0, positive].sum())
    coords = [(y, x) for y in range(0, height - cfg.patch_size + 1, cfg.stride)
              for x in range(0, width - cfg.patch_size + 1, cfg.stride)]
    drops: list[float] = []
    for start in range(0, len(coords), cfg.batch_size):
        chunk = coords[start:start + cfg.batch_size]
        batch = np.repeat(image[None, ...], len(chunk), axis=0)
        for k, (y, x) in enumerate(chunk):
            batch[k, y:y + cfg.patch_size, x:x + cfg.patch_size] = 0.0  # black = MRI background
        drops.extend(base - predictor._run_model(batch)[:, positive].sum(axis=1))
    total = np.zeros((height, width)); count = np.zeros((height, width))
    for (y, x), drop in zip(coords, drops):
        total[y:y + cfg.patch_size, x:x + cfg.patch_size] += drop
        count[y:y + cfg.patch_size, x:x + cfg.patch_size] += 1
    return base, total / np.maximum(count, 1), float(max(drops))


def head_mask(rgb: np.ndarray, cfg: RegionViewConfig) -> np.ndarray:
    """Head area (incl. dark interior such as haemorrhage on SWI) vs the black scan background."""
    grey = cv2.GaussianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    return ndimage.binary_fill_holes(grey > cfg.head_intensity)


def region_mask(drop_map: np.ndarray, head: np.ndarray, cfg: RegionViewConfig) -> np.ndarray | None:
    """Connected area with the strongest drop (inside the head), or None if there is none."""
    # Smooth the grid-shaped map (σ = half the stride) so the outline follows the measured trend, not
    # the patch grid; this changes the drawing only, not which area produced the largest drop.
    smooth = cv2.GaussianBlur(drop_map.astype(np.float32), (0, 0), sigmaX=cfg.stride / 2)
    resized = cv2.resize(smooth, (head.shape[1], head.shape[0]), interpolation=cv2.INTER_LINEAR)
    resized[~head] = 0.0
    peak = float(resized.max())
    if peak <= 0:
        return None
    labels, count = ndimage.label(resized >= cfg.threshold * peak)
    if count == 0:
        return None
    strongest = 1 + int(np.argmax(ndimage.sum(resized, labels, index=range(1, count + 1))))
    return labels == strongest


def render_overlay(rgb: np.ndarray, mask: np.ndarray, cfg: RegionViewConfig) -> bytes:
    """Translucent fill + outline of ``mask`` on the uploaded MRI (original resolution), as PNG."""
    out = rgb.astype(np.float32)
    colour = np.array(REGION_RGB, dtype=np.float32)
    out[mask] = (1 - cfg.fill_alpha) * out[mask] + cfg.fill_alpha * colour
    out = out.astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, REGION_RGB, thickness=max(1, round(min(rgb.shape[:2]) / 160)))
    ok, png = cv2.imencode(".png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("Could not encode the region overlay as PNG")
    return png.tobytes()


def stroke_region(predictor: "Predictor", data: bytes, filename: str, view: "BinaryViewConfig",
                  cfg: RegionViewConfig) -> RegionResult:
    """Occlusion-sensitivity region for an uploaded MRI, using the cached predictor's existing model."""
    started = time.perf_counter()
    image = predictor.preprocess_bytes(data, filename)  # same preprocessing as the prediction
    positive = [predictor.loaded.class_names.index(name) for name in view.positive_classes]
    base, drop_map, max_drop = occlusion_map(predictor, image, positive, cfg)
    overlay = None
    if max_drop >= cfg.min_drop:
        rgb = load_image_rgb(data)
        mask = region_mask(drop_map, head_mask(rgb, cfg), cfg)
        overlay = render_overlay(rgb, mask, cfg) if mask is not None else None
    seconds = time.perf_counter() - started
    logger.info("Region view %s: P(stroke) %.3f, max single-patch drop %.3f, region %s, %.2f s",
                filename, base, max_drop, "shown" if overlay else "not shown", seconds)
    return RegionResult(overlay, base, base - max_drop, seconds)
