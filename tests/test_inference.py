"""Tests for the Phase 5 inference pipeline.

Most tests use a small, randomly initialised model saved to tmp_path (fast,
offline) — they exercise code paths only; its predictions mean nothing.
One integration test uses the real trained model when it exists.
"""

import copy
import json

import numpy as np
import pytest
from PIL import Image

from src.inference import cli
from src.inference.errors import (
    ClassMappingError,
    ImageValidationError,
    ModelNotFoundError,
    ModelOutputError,
    PreprocessingMismatchError,
)
from src.inference.model_loader import clear_model_cache
from src.inference.predictor import Predictor, batch_to_frame
from src.training.model import build_model
from src.training.settings import TrainingSettings
from src.training.train import write_model_metadata
from src.utils.config import PROJECT_ROOT, load_config, resolve_path


def _config(class_names=None) -> dict:
    config = copy.deepcopy(load_config())
    config["model"]["weights"] = None
    config["preprocessing"]["image_size"] = [64, 64]
    if class_names:
        config["data"]["class_names"] = class_names
    return config


def _save_model(directory, config):
    settings = TrainingSettings.from_config(config)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.keras"
    build_model(settings).save(path)
    write_model_metadata(settings, directory, {})
    return path


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    return _save_model(tmp_path_factory.mktemp("model"), _config())


@pytest.fixture()
def predictor(model_path):
    return Predictor(_config(), model_path=model_path)


def _mri_like(size=96, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((size, size)) * 255).astype(np.uint8)


def _save(path, array, mode):
    Image.fromarray(array).convert(mode).save(path)
    return path


# ---- valid inputs -----------------------------------------------------------------------

@pytest.mark.parametrize("mode,suffix", [("L", ".png"), ("RGB", ".jpg"), ("RGBA", ".png"), ("P", ".png")])
def test_valid_images_produce_valid_probabilities(predictor, tmp_path, mode, suffix):
    path = _save(tmp_path / f"img{suffix}", _mri_like(), mode)
    result = predictor.predict(path)
    probs = np.array(list(result.probabilities.values()))
    assert list(result.probabilities) == list(load_config()["data"]["class_names"])
    assert np.all((probs >= 0) & (probs <= 1)) and probs.sum() == pytest.approx(1.0)
    assert result.predicted_class == max(result.probabilities, key=result.probabilities.get)
    assert result.confidence == pytest.approx(probs.max())
    assert result.disclaimer


def test_prediction_is_deterministic(predictor, tmp_path):
    path = _save(tmp_path / "img.png", _mri_like(), "L")
    assert predictor.predict(path).probabilities == predictor.predict(path).probabilities


def test_preprocessing_matches_training_function(predictor, tmp_path):
    from src.preprocessing.image_ops import preprocess_image

    path = _save(tmp_path / "img.png", _mri_like(), "L")
    assert np.array_equal(predictor.preprocess(path), preprocess_image(path, predictor.settings.preprocessing))


def test_predict_bytes_matches_predict_path(predictor, tmp_path):
    path = _save(tmp_path / "img.png", _mri_like(), "L")
    from_bytes = predictor.predict_bytes(path.read_bytes(), "upload.png")
    from_path = predictor.predict(path)
    assert from_bytes.probabilities == from_path.probabilities
    assert from_bytes.image_path == "upload.png"


def test_predict_bytes_rejects_bad_uploads(predictor):
    with pytest.raises(ImageValidationError, match="Unsupported file type"):
        predictor.predict_bytes(b"abc", "scan.dcm")
    with pytest.raises(ImageValidationError, match="empty"):
        predictor.predict_bytes(b"", "scan.png")
    with pytest.raises(ImageValidationError, match="Cannot read image"):
        predictor.predict_bytes(b"\x89PNG\r\n\x1a\nxx", "scan.png")


# ---- invalid inputs ---------------------------------------------------------------------

def test_missing_image(predictor, tmp_path):
    with pytest.raises(ImageValidationError, match="not found"):
        predictor.predict(tmp_path / "nope.png")


def test_unsupported_extension(predictor, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")
    with pytest.raises(ImageValidationError, match="Unsupported file type"):
        predictor.predict(path)


def test_corrupted_image(predictor, tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage" * 50)
    with pytest.raises(ImageValidationError, match="Cannot read image"):
        predictor.predict(path)


def test_empty_file(predictor, tmp_path):
    path = tmp_path / "empty.png"
    path.write_bytes(b"")
    with pytest.raises(ImageValidationError, match="empty"):
        predictor.predict(path)


def test_sixteen_bit_image_rejected(predictor, tmp_path):
    path = tmp_path / "deep.png"
    Image.fromarray((np.arange(96 * 96).reshape(96, 96) * 7).astype(np.uint16)).save(path)
    with pytest.raises(ImageValidationError, match="pixel format"):
        predictor.predict(path)


def test_too_small_image_rejected(predictor, tmp_path):
    path = _save(tmp_path / "tiny.png", _mri_like(size=16), "L")
    with pytest.raises(ImageValidationError, match="too small"):
        predictor.predict(path)


# ---- model / mapping errors ---------------------------------------------------------------

def test_missing_model(tmp_path):
    with pytest.raises(ModelNotFoundError):
        Predictor(_config(), model_path=tmp_path / "missing.keras")


def test_class_mapping_mismatch(model_path):
    clear_model_cache()
    with pytest.raises(ClassMappingError):
        Predictor(_config(class_names=["Normal", "Ischemic", "Haemorrhagic"]), model_path=model_path)


def test_preprocessing_mismatch(model_path):
    clear_model_cache()
    config = _config()
    config["preprocessing"]["interpolation"] = "cubic"
    with pytest.raises(PreprocessingMismatchError):
        Predictor(config, model_path=model_path)


def test_invalid_output_shape(predictor):
    with pytest.raises(ModelOutputError):
        predictor._probabilities(np.ones((1, 5)))
    with pytest.raises(ModelOutputError):
        predictor._probabilities(np.array([[0.5, 0.5, 0.5]]))  # does not sum to 1


def test_binary_sigmoid_model(tmp_path):
    config = _config(class_names=["no_stroke", "stroke"])
    path = _save_model(tmp_path / "binary", config)
    binary = Predictor(config, model_path=path)
    assert binary.loaded.output_kind == "sigmoid"
    result = binary.predict(_save(tmp_path / "img.png", _mri_like(), "L"))
    assert set(result.probabilities) == {"no_stroke", "stroke"}
    assert sum(result.probabilities.values()) == pytest.approx(1.0)


def test_model_is_loaded_once(model_path):
    first = Predictor(_config(), model_path=model_path)
    second = Predictor(_config(), model_path=model_path)
    assert first.loaded.model is second.loaded.model


# ---- batch & CLI --------------------------------------------------------------------------

def test_batch_reports_errors_per_item(predictor, tmp_path):
    good = _save(tmp_path / "good.png", _mri_like(), "L")
    items = predictor.predict_batch([good, tmp_path / "missing.png"])
    assert items[0].ok and not items[1].ok and "not found" in items[1].error
    frame = batch_to_frame(items, predictor.loaded.class_names)
    assert list(frame["status"]) == ["ok", "error"]
    single = predictor.predict(good)
    assert frame.loc[0, "predicted_class"] == single.predicted_class
    assert frame.loc[0, "confidence"] == pytest.approx(single.confidence, abs=1e-5)


def test_cli_exit_codes(model_path, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "Predictor", lambda model_path=None: Predictor(_config(), model_path=model_path))
    good = _save(tmp_path / "good.png", _mri_like(), "L")
    assert cli.main(["--image", str(good), "--model", str(model_path), "--json"]) == cli.EXIT_OK
    assert json.loads(capsys.readouterr().out)["predicted_class"]
    assert cli.main(["--image", str(tmp_path / "missing.png"), "--model", str(model_path)]) == cli.EXIT_INPUT
    assert cli.main(["--image", str(good), "--model", str(tmp_path / "none.keras")]) == cli.EXIT_MODEL


# ---- real trained model (integration) -----------------------------------------------------

def test_real_model_on_real_test_image():
    config = load_config()
    model_file = resolve_path(config["inference"]["model_path"])
    image = next((PROJECT_ROOT / "dataset" / "test").rglob("*.png"), None)
    if not model_file.is_file() or image is None:
        pytest.skip("trained model or prepared test split not available")
    result = Predictor(config).predict(image)
    assert result.predicted_class in config["data"]["class_names"]
    assert sum(result.probabilities.values()) == pytest.approx(1.0)
