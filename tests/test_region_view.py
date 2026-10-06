"""Occlusion-sensitivity region view, tested with a stand-in model whose behaviour is known.

The stand-in's P(STROKE) depends only on one square of the image, so the method must
outline that square — and must outline nothing when no area matters.
"""

import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.inference.binary_view import BinaryViewConfig
from src.inference.region_view import RegionViewConfig, region_mask, stroke_region
from src.utils.config import load_config

VIEW = BinaryViewConfig("STROKE", "NO STROKE", ("Haemorrhagic", "Ischemic"), ("Normal",))
CFG = RegionViewConfig.from_config(load_config())
SIZE = 224
SPOT = (slice(140, 180), slice(40, 80))  # rows, cols of the area the stand-in model "looks at"


def _png(size=SIZE) -> bytes:
    """Bright disc ("head") on black, like an axial MRI slice."""
    yy, xx = np.mgrid[:size, :size]
    grey = np.where((yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (size * 0.45) ** 2, 160, 0).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(grey).save(buffer, format="PNG")
    return buffer.getvalue()


class FakePredictor:
    """P(STROKE) = 0.9 while SPOT is visible, falling as it is hidden (or constant if insensitive)."""

    def __init__(self, sensitive=True):
        self.sensitive = sensitive
        self.loaded = SimpleNamespace(class_names=("Haemorrhagic", "Ischemic", "Normal"))

    def preprocess_bytes(self, data, filename):
        rgb = np.asarray(Image.open(io.BytesIO(data)).convert("RGB").resize((SIZE, SIZE)), dtype=np.float32)
        return rgb

    def _run_model(self, batch):
        visible = (batch[:, SPOT[0], SPOT[1], 0] > 0).mean(axis=(1, 2)) if self.sensitive else np.ones(len(batch))
        stroke = 0.2 + 0.7 * visible
        return np.column_stack([stroke / 2, stroke / 2, 1 - stroke])


def _mask_from_png(png: bytes) -> np.ndarray:
    rgb = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.int16)
    return (rgb[..., 0] - rgb[..., 1]) > 40  # red fill/outline only


def test_region_outlines_the_area_the_model_depends_on():
    result = stroke_region(FakePredictor(), _png(), "slice.png", VIEW, CFG)
    assert result.stroke_probability == pytest.approx(0.9)
    assert result.max_drop > CFG.min_drop
    red = _mask_from_png(result.overlay_png)
    rows, cols = np.nonzero(red)
    centre = (rows.mean(), cols.mean())
    assert abs(centre[0] - 160) < 25 and abs(centre[1] - 60) < 25   # around SPOT, not elsewhere
    assert red.mean() < 0.15                                         # one area, not the whole head


def test_nothing_is_highlighted_when_no_area_matters():
    result = stroke_region(FakePredictor(sensitive=False), _png(), "slice.png", VIEW, CFG)
    assert result.overlay_png is None
    assert result.max_drop == pytest.approx(0.0)


def test_overlay_keeps_uploaded_resolution():
    result = stroke_region(FakePredictor(), _png(size=300), "slice.png", VIEW, CFG)
    assert Image.open(io.BytesIO(result.overlay_png)).size == (300, 300)


def test_region_mask_is_none_without_a_positive_drop():
    head = np.ones((64, 64), dtype=bool)
    assert region_mask(np.zeros((SIZE, SIZE)), head, CFG) is None
    assert region_mask(-np.ones((SIZE, SIZE)), head, CFG) is None


def test_region_stays_inside_the_head():
    drop = np.zeros((SIZE, SIZE)); drop[:, :60] = 1.0               # strongest drop over the background edge
    head = np.zeros((SIZE, SIZE), dtype=bool); head[:, 30:] = True
    mask = region_mask(drop, head, CFG)
    assert mask is not None and not mask[:, :30].any()
