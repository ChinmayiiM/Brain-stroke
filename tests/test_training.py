"""Unit tests for the training package (settings, data loader, model, freezing, callbacks).

Models are built with ``weights=None`` and a small input so tests run offline and fast;
the real ImageNet configuration is exercised by ``python -m src.training.train --sanity-check``.
"""

import copy

import keras
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.training import data_loader
from src.training.callbacks import build_callbacks, checkpoint_path
from src.training.class_balance import compute_class_weights
from src.training.inspection import FreezeVerificationError, parameter_counts, verify_stage1, verify_stage2
from src.training.model import build_model, compile_model, get_backbone, unfreeze_top_layers
from src.training.settings import TrainingSettings
from src.utils.config import load_config


def _config(**overrides) -> dict:
    config = copy.deepcopy(load_config())
    config["model"]["weights"] = None
    config["preprocessing"]["image_size"] = [64, 64]
    config["training"]["batch_size"] = 4
    for dotted, value in overrides.items():
        node = config
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return config


@pytest.fixture(scope="module")
def settings() -> TrainingSettings:
    return TrainingSettings.from_config(_config())


# ---- settings ---------------------------------------------------------------------------

def test_settings_derive_classes_and_shape(settings):
    assert settings.num_classes == len(load_config()["data"]["class_names"])
    assert settings.input_shape == (64, 64, 3)


def test_settings_reject_fine_tune_lr_not_smaller():
    with pytest.raises(ValueError):
        TrainingSettings.from_config(_config(**{"training.stage2.learning_rate": 1.0}))


# ---- class weights ----------------------------------------------------------------------

def test_balanced_class_weights():
    labels = np.array([0] * 60 + [1] * 10 + [2] * 30)
    weights = compute_class_weights(labels, ("a", "b", "c"), "balanced")
    assert weights[1] > weights[2] > weights[0]
    # balanced → every class contributes equally: n_c * w_c == n / k
    for c, n_c in zip(range(3), (60, 10, 30)):
        assert n_c * weights[c] == pytest.approx(100 / 3)


def test_class_weights_disabled_and_missing_class():
    assert compute_class_weights(np.array([0, 1]), ("a", "b"), None) is None
    with pytest.raises(ValueError):
        compute_class_weights(np.array([0, 0, 2]), ("a", "b", "c"), "balanced")


# ---- model & freezing -------------------------------------------------------------------

def test_multiclass_head_is_softmax(settings):
    model = build_model(settings)
    assert model.output_shape == (None, settings.num_classes)
    assert model.get_layer("classifier").activation.__name__ == "softmax"


def test_binary_head_is_single_sigmoid():
    binary = TrainingSettings.from_config(_config(**{"data.class_names": ["no", "yes"]}))
    model = build_model(binary)
    assert model.output_shape == (None, 1)
    assert model.get_layer("classifier").activation.__name__ == "sigmoid"


def test_stage1_freezes_entire_backbone(settings):
    model = build_model(settings)
    report = verify_stage1(model)
    assert report["backbone_trainable"] == 0
    assert report["trainable"] == report["head_trainable"] > 0


def test_stage2_unfreezes_only_top_layers_and_keeps_batchnorm_frozen(settings):
    model = build_model(settings)
    unfreeze_top_layers(model, 20)
    report = verify_stage2(model, 20)
    backbone = get_backbone(model)
    lower = backbone.layers[: len(backbone.layers) - 20]
    assert not any(layer.trainable_weights for layer in lower)
    assert not any(
        layer.trainable_weights for layer in backbone.layers if isinstance(layer, keras.layers.BatchNormalization)
    )
    assert 0 < report["backbone_trainable"] < report["backbone_total"]


def test_stage1_verification_detects_unfrozen_backbone(settings):
    model = build_model(settings)
    unfreeze_top_layers(model, 5)
    with pytest.raises(FreezeVerificationError):
        verify_stage1(model)


def test_unfreeze_rejects_invalid_layer_count(settings):
    model = build_model(settings)
    with pytest.raises(ValueError):
        unfreeze_top_layers(model, 0)


def test_compile_and_forward_pass(settings):
    model = build_model(settings)
    compile_model(model, settings, settings.stage1.learning_rate)
    x = np.random.default_rng(0).uniform(0, 255, (2, *settings.input_shape)).astype("float32")
    y = keras.utils.to_categorical([0, 2], settings.num_classes)
    probs = np.asarray(model(x, training=False))
    assert probs.shape == (2, settings.num_classes)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(model.test_on_batch(x, y, return_dict=True)["loss"])
    assert parameter_counts(model)["total"] == model.count_params()


# ---- callbacks --------------------------------------------------------------------------

def test_callbacks(settings, tmp_path):
    callbacks = build_callbacks(settings, 1, tmp_path / "run")
    kinds = {type(cb).__name__ for cb in callbacks}
    assert {"EarlyStopping", "ModelCheckpoint", "ReduceLROnPlateau", "CSVLogger"} <= kinds
    checkpoint = next(cb for cb in callbacks if isinstance(cb, keras.callbacks.ModelCheckpoint))
    assert checkpoint.save_best_only
    assert checkpoint_path(settings, 1).suffix == ".keras"
    assert checkpoint_path(settings, 1) != checkpoint_path(settings, 2)


# ---- data loader ------------------------------------------------------------------------

@pytest.fixture()
def tiny_metadata(tmp_path, monkeypatch):
    """A 3-class, 3-split metadata file over small synthetic images in tmp_path."""
    monkeypatch.setattr(data_loader, "PROJECT_ROOT", tmp_path)
    rng = np.random.default_rng(0)
    rows = []
    class_names = load_config()["data"]["class_names"]
    for split in data_loader.SPLITS:
        for label, name in enumerate(class_names):
            for i in range(3):
                rel = f"{split}/{name}/img_{i}.png"
                (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((rng.random((80, 80)) * 255).astype("uint8")).save(tmp_path / rel)
                rows.append({"image_path": rel, "label": label, "class_name": name, "split": split, "sequence": "DWI"})
    path = tmp_path / "metadata.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    config = _config(**{"pipeline.metadata_file": str(path)})
    return TrainingSettings.from_config(config)


def test_validation_pipeline_is_ordered_and_unaugmented(tiny_metadata):
    metadata = data_loader.load_metadata(tiny_metadata)
    spec = data_loader.get_split(metadata, "validation", tiny_metadata)
    ds = data_loader.build_dataset(spec, tiny_metadata, training=False)
    x, y = (np.asarray(t) for t in next(iter(ds)))
    assert x.shape == (4, 64, 64, 3) and x.dtype == np.float32
    assert np.array_equal(y.argmax(axis=1), spec.labels[:4])
    x2 = np.asarray(next(iter(ds))[0])
    assert np.array_equal(x, x2)


def test_training_pipeline_covers_all_samples(tiny_metadata):
    metadata = data_loader.load_metadata(tiny_metadata)
    spec = data_loader.get_split(metadata, "train", tiny_metadata)
    ds = data_loader.build_dataset(spec, tiny_metadata, training=True)
    labels = np.concatenate([np.asarray(y).argmax(axis=1) for _, y in ds])
    assert sorted(labels.tolist()) == sorted(spec.labels.tolist())


def test_metadata_label_order_mismatch_is_rejected(tiny_metadata):
    frame = pd.read_csv(tiny_metadata.metadata_file)
    frame["label"] = (frame["label"] + 1) % 3
    frame.to_csv(tiny_metadata.metadata_file, index=False)
    with pytest.raises(ValueError):
        data_loader.load_metadata(tiny_metadata)


def test_sequence_filter(tiny_metadata):
    filtered = TrainingSettings.from_config(
        _config(**{"pipeline.metadata_file": str(tiny_metadata.metadata_file), "pipeline.sequences": ["SWI"]})
    )
    assert data_loader.load_metadata(filtered).empty
