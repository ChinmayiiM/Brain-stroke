"""Unit tests for the shared preprocessing and augmentation components."""

import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing.augmentation import build_augmentation_pipeline
from src.preprocessing.image_ops import PreprocessingConfig, load_image_rgb, preprocess_batch, preprocess_image
from src.utils.config import load_config

CONFIG = load_config()
PREP = PreprocessingConfig.from_config(CONFIG)


def _png_bytes(array: np.ndarray, mode: str) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(array).convert(mode).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def gray_array():
    return (np.arange(300 * 300) % 256).reshape(300, 300).astype(np.uint8)


def test_grayscale_is_replicated_to_rgb(gray_array):
    rgb = load_image_rgb(_png_bytes(gray_array, "L"))
    assert rgb.shape == (300, 300, 3) and rgb.dtype == np.uint8
    assert np.array_equal(rgb[..., 0], rgb[..., 2])


def test_rgba_alpha_is_dropped(gray_array):
    assert load_image_rgb(_png_bytes(gray_array, "RGBA")).shape[-1] == 3


def test_preprocess_output_contract(gray_array):
    x = preprocess_image(_png_bytes(gray_array, "L"), PREP)
    assert x.shape == (*PREP.image_size, 3)
    assert x.dtype == np.float32
    # EfficientNetB0 normalises internally — values must stay in [0, 255], not [0, 1].
    assert x.max() > 1.0 and x.min() >= 0.0 and x.max() <= 255.0


def test_path_bytes_and_array_inputs_agree(tmp_path, gray_array):
    path = tmp_path / "img.png"
    data = _png_bytes(gray_array, "L")
    path.write_bytes(data)
    from_path = preprocess_image(path, PREP)
    assert np.array_equal(from_path, preprocess_image(data, PREP))
    assert np.array_equal(from_path, preprocess_image(gray_array, PREP))


def test_batch_shape(gray_array):
    data = _png_bytes(gray_array, "L")
    assert preprocess_batch([data, data], PREP).shape == (2, *PREP.image_size, 3)


def test_invalid_bytes_raise_value_error():
    with pytest.raises(ValueError):
        load_image_rgb(b"definitely not an image")


def test_augmentation_only_active_in_training(gray_array):
    pipeline = build_augmentation_pipeline(CONFIG, seed=0)
    batch = preprocess_image(gray_array, PREP)[None]
    assert np.allclose(np.asarray(pipeline(batch, training=False)), batch)
    augmented = np.asarray(pipeline(batch, training=True))
    assert augmented.shape == batch.shape
    assert not np.allclose(augmented, batch)
    assert augmented.min() >= 0.0 and augmented.max() <= 255.0


def test_augmentation_disabled_returns_none():
    config = {"augmentation": {"enabled": False}}
    assert build_augmentation_pipeline(config) is None
