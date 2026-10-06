"""STROKE / NO STROKE presentation of the 3-class model output."""

import pytest

from src.inference.binary_view import BinaryViewConfig, to_binary
from src.inference.errors import ClassMappingError
from src.inference.predictor import PredictionResult
from src.utils.config import load_config

VIEW = BinaryViewConfig.from_config(load_config())


def _result(h: float, i: float, n: float) -> PredictionResult:
    probs = {"Haemorrhagic": h, "Ischemic": i, "Normal": n}
    top = max(probs, key=probs.get)
    return PredictionResult("x.png", top, list(probs).index(top), probs[top], probs, "m.keras", "d")


def test_config_groups_cover_model_classes():
    VIEW.validate(tuple(load_config()["data"]["class_names"]))
    assert VIEW.positive_classes == ("Haemorrhagic", "Ischemic") and VIEW.negative_classes == ("Normal",)


def test_example_no_stroke():  # Haemorrhagic 20.49 % + Ischemic 5.11 % = 25.60 % ; Normal 74.40 %
    binary = to_binary(_result(0.2049, 0.0511, 0.7440), VIEW)
    assert binary.label == "NO STROKE"
    assert binary.confidence == pytest.approx(0.7440)
    assert binary.positive_probability == pytest.approx(0.2560)
    assert f"{binary.confidence:.2%}" == "74.40%"


def test_example_stroke():  # 55 % + 25 % = 80 % stroke
    binary = to_binary(_result(0.55, 0.25, 0.20), VIEW)
    assert binary.label == "STROKE"
    assert binary.confidence == pytest.approx(0.80)
    assert f"{binary.confidence:.2%}" == "80.00%"


def test_binary_is_not_the_three_class_argmax():
    # 3-class argmax is Normal (0.45), but stroke classes together are 0.55
    binary = to_binary(_result(0.30, 0.25, 0.45), VIEW)
    assert binary.label == "STROKE" and binary.confidence == pytest.approx(0.55)


def test_tie_goes_to_stroke():
    assert to_binary(_result(0.25, 0.25, 0.50), VIEW).label == "STROKE"


def test_confidence_is_at_least_half_and_probabilities_complement():
    for h, i, n in [(0.1, 0.1, 0.8), (0.6, 0.1, 0.3), (0.33, 0.33, 0.34)]:
        b = to_binary(_result(h, i, n), VIEW)
        assert b.confidence >= 0.5
        assert b.positive_probability + b.negative_probability == pytest.approx(1.0)


def test_mapping_must_partition_classes():
    bad = BinaryViewConfig("S", "N", ("Haemorrhagic",), ("Normal",))  # Ischemic missing
    with pytest.raises(ClassMappingError):
        to_binary(_result(0.2, 0.3, 0.5), bad)
