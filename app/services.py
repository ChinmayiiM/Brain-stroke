"""Non-UI helpers for the Streamlit app (no Streamlit imports → unit-testable).

This module deliberately does NOT import TensorFlow at import time: the page must
render immediately, while the model is loaded in a background thread. Everything
shown about the model comes from ``config.yaml`` and the model's
``model_metadata.json`` written by Phase 4 — nothing is hard-coded here.
"""

from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.utils.config import PROJECT_ROOT, load_config, resolve_path
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.inference.predictor import Predictor

logger = get_logger("streamlit_app")

# Optional override, e.g. to point the app at another trained model file.
MODEL_PATH_ENV = "STROKE_APP_MODEL_PATH"


def model_path_override() -> str | None:
    return os.environ.get(MODEL_PATH_ENV) or None


def model_file(config: dict, override: str | None = None) -> Path:
    return resolve_path(override or config["inference"]["model_path"])


def model_files_problem(config: dict, override: str | None = None) -> str | None:
    """Cheap pre-check (no TensorFlow): describe a missing model/metadata file, or None if present."""
    path = model_file(config, override)
    if not path.is_file():
        return (f"Trained model not found: {path}. Train it with `python -m src.training.train` "
                f"or set {MODEL_PATH_ENV} to a .keras file.")
    metadata = path.parent / config["inference"].get("metadata_file", "model_metadata.json")
    if not metadata.is_file():
        return f"Model metadata not found: {metadata}."
    return None


def read_model_metadata(config: dict, override: str | None = None) -> dict[str, Any]:
    path = model_file(config, override)
    metadata = path.parent / config["inference"].get("metadata_file", "model_metadata.json")
    return json.loads(metadata.read_text(encoding="utf-8"))


def create_predictor(model_path: str | None = None) -> "Predictor":
    """Build the Phase 5 predictor (the app's only route to the model) and warm it up once.

    TensorFlow is imported here, on first use, not when the app starts.
    """
    started = time.perf_counter()
    from src.inference.predictor import Predictor  # heavy: imports TensorFlow/Keras

    logger.info("TensorFlow/Keras import took %.2f s", time.perf_counter() - started)
    predictor = Predictor(load_config(), model_path=model_path)
    predictor.warm_up()  # build the TensorFlow graph once, so predictions are not slowed by it
    logger.info("Model ready (import + load + warm-up) in %.2f s", time.perf_counter() - started)
    return predictor


def start_model_loading(model_path: str | None = None) -> Future:
    """Load the model in a background thread and return a Future for the Predictor.

    The page keeps rendering and stays interactive while TensorFlow initialises.
    """
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-loader")
    future = executor.submit(create_predictor, model_path)
    executor.shutdown(wait=False)  # the worker thread finishes the load, then exits
    logger.info("Model loading started in the background")
    return future


@dataclass(frozen=True)
class ModelCard:
    backbone: str
    pretrained_weights: str | None
    input_size: str
    class_names: tuple[str, ...]
    output: str
    dataset_name: str
    dataset_url: str
    selected_stage: int | None
    test_accuracy: float | None
    test_balanced_accuracy: float | None
    test_macro_f1: float | None
    test_images: int | None
    test_patients: int | None
    model_file: str


def build_model_card(config: dict, override: str | None = None) -> ModelCard:
    """Model facts from model_metadata.json + config (no model loading needed)."""
    meta = read_model_metadata(config, override)
    test = meta.get("test_evaluation") or {}
    height, width, channels = meta["input_shape"]
    path = model_file(config, override)
    return ModelCard(
        backbone=meta.get("backbone", "unknown"),
        pretrained_weights=config["model"].get("weights"),
        input_size=f"{height} × {width} × {channels}",
        class_names=tuple(meta["class_names"]),
        output=meta.get("output", "unknown"),
        dataset_name=config["dataset"]["name"],
        dataset_url=config["dataset"]["source_url"],
        selected_stage=meta.get("selected_stage"),
        test_accuracy=test.get("accuracy"),
        test_balanced_accuracy=test.get("balanced_accuracy"),
        test_macro_f1=test.get("macro_f1"),
        test_images=test.get("n_images"),
        test_patients=_count_test_patients(config),
        model_file=path.relative_to(PROJECT_ROOT).as_posix() if path.is_relative_to(PROJECT_ROOT) else str(path),
    )


def _count_test_patients(config: dict) -> int | None:
    metadata_file = resolve_path(config["pipeline"]["metadata_file"])
    if not metadata_file.is_file():
        return None
    # stdlib csv, not pandas: this runs at page start-up and only needs a unique count
    with metadata_file.open(newline="", encoding="utf-8") as handle:
        patients = {row["patient_id"] for row in csv.DictReader(handle)
                    if row["split"] == "test" and row["patient_id"]}
    return len(patients)
