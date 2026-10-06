"""Single-image and batch prediction with the trained EfficientNetB0 model.

Pipeline::

    image path / uploaded bytes ─► file checks (exists, extension, non-empty, decodable, supported mode)
               ─► shared preprocessing (src.preprocessing.image_ops.preprocess_image:
                  validate → RGB → resize → float32 [0, 255])
               ─► add batch dimension ─► model(x, training=False)
               ─► probabilities (softmax rows / sigmoid → [1-p, p]) ─► argmax class + confidence

No augmentation is applied: the saved model contains no random layers (verified
at load time) and the model is called with ``training=False`` (Dropout inactive).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import keras
import numpy as np

from src.inference.errors import ImageValidationError, InferenceError, ModelOutputError
from src.inference.model_loader import InferenceSettings, LoadedModel, load_inference_model
from src.inference.validation import (  # re-exported: validation lives in a TensorFlow-free module
    SUPPORTED_MODES,
    ImageInfo,
    validate_image_bytes,
    validate_image_file,
)
from src.preprocessing.image_ops import load_image_rgb, preprocess_image
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.logger import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

@dataclass(frozen=True)
class PredictionResult:
    image_path: str
    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]
    model_path: str
    disclaimer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BatchItem:
    image_path: str
    result: PredictionResult | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.result is not None


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


class Predictor:
    """Reusable predictor: loads the model once, then predicts any number of images."""

    def __init__(self, config: dict | None = None, model_path: str | Path | None = None) -> None:
        self.settings = InferenceSettings.from_config(config or load_config(), model_path)
        self.loaded: LoadedModel = load_inference_model(self.settings)

    # ---- preprocessing --------------------------------------------------------------
    def preprocess(self, path: str | Path) -> np.ndarray:
        """Validate and preprocess one image into an (H, W, 3) float32 array."""
        image_path = validate_image_file(path, self.settings.allowed_extensions)
        try:
            rgb = load_image_rgb(image_path)
            return preprocess_image(rgb, self.settings.preprocessing)  # same function as training
        except ValueError as exc:
            raise ImageValidationError(f"Image rejected: {exc} ({image_path})") from exc

    def preprocess_bytes(self, data: bytes, filename: str) -> np.ndarray:
        """Same as :meth:`preprocess` for an in-memory upload (``filename`` gives the extension)."""
        validate_image_bytes(data, filename, self.settings.allowed_extensions)
        try:
            rgb = load_image_rgb(data)
            return preprocess_image(rgb, self.settings.preprocessing)  # same function as training
        except ValueError as exc:
            raise ImageValidationError(f"Image rejected: {exc} ({filename})") from exc

    # ---- model output -----------------------------------------------------------------
    def _probabilities(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float64)
        expected_units = 1 if self.loaded.output_kind == "sigmoid" else self.loaded.num_classes
        if raw.ndim != 2 or raw.shape[1] != expected_units:
            raise ModelOutputError(f"Unexpected model output shape {raw.shape}; expected (N, {expected_units})")
        if not np.all(np.isfinite(raw)):
            raise ModelOutputError("Model produced non-finite outputs (NaN/Inf)")
        tol = self.settings.probability_tolerance
        if raw.min() < -tol or raw.max() > 1 + tol:
            raise ModelOutputError(f"Model outputs outside [0, 1]: min={raw.min():.4g}, max={raw.max():.4g}")
        probs = np.column_stack([1.0 - raw[:, 0], raw[:, 0]]) if self.loaded.output_kind == "sigmoid" else raw
        sums = probs.sum(axis=1)
        if np.any(np.abs(sums - 1.0) > tol):
            raise ModelOutputError(f"Probabilities do not sum to 1 (sums={np.round(sums, 6).tolist()})")
        return probs / sums[:, None]  # remove float32 rounding (same as Phase 4 evaluation)

    def _result(self, image_label: str, probs: np.ndarray) -> PredictionResult:
        index = int(np.argmax(probs))
        names = self.loaded.class_names
        return PredictionResult(
            image_path=image_label,
            predicted_class=names[index],
            predicted_index=index,
            confidence=float(probs[index]),
            probabilities={name: float(p) for name, p in zip(names, probs)},
            model_path=_display_path(self.loaded.path),
            disclaimer=self.settings.disclaimer,
        )

    def _run_model(self, batch: np.ndarray) -> np.ndarray:
        try:
            raw = self.loaded.infer(batch)  # compiled model(x, training=False): no dropout, no augmentation
        except Exception as exc:  # noqa: BLE001
            raise ModelOutputError(f"Prediction failed: {type(exc).__name__}: {exc}") from exc
        return self._probabilities(keras.ops.convert_to_numpy(raw))

    # ---- public API -------------------------------------------------------------------
    def warm_up(self) -> None:
        """Run the model once on a blank input so TensorFlow builds its graph now, not on the first real
        prediction. The output is discarded; it does not affect any prediction."""
        height, width = self.settings.preprocessing.image_size
        blank = np.zeros((1, height, width, self.settings.preprocessing.channels), dtype=np.float32)
        start = time.perf_counter()
        self.loaded.infer(blank)  # traces the compiled forward pass once per process
        logger.info("Warm-up (graph tracing) took %.2f s", time.perf_counter() - start)

    def predict(self, path: str | Path) -> PredictionResult:
        """Predict a single image."""
        image = self.preprocess(path)
        probs = self._run_model(image[None, ...])[0]  # add batch dimension
        return self._result(_display_path(Path(path)), probs)

    def predict_bytes(self, data: bytes, filename: str) -> PredictionResult:
        """Predict an in-memory image (e.g. a web upload). ``image_path`` in the result is ``filename``."""
        start = time.perf_counter()
        image = self.preprocess_bytes(data, filename)
        prepared = time.perf_counter()
        probs = self._run_model(image[None, ...])[0]
        done = time.perf_counter()
        logger.info("Timing %s: validate+preprocess %.1f ms | model %.1f ms | total %.1f ms", Path(filename).name,
                    (prepared - start) * 1000, (done - prepared) * 1000, (done - start) * 1000)
        return self._result(Path(filename).name, probs)

    def predict_batch(self, paths: list[str | Path], batch_size: int = 16) -> list[BatchItem]:
        """Predict many images; invalid images are reported per item instead of aborting the batch."""
        items = [BatchItem(image_path=str(p)) for p in paths]
        valid: list[tuple[int, np.ndarray]] = []
        for i, path in enumerate(paths):
            try:
                valid.append((i, self.preprocess(path)))
            except InferenceError as exc:
                items[i].error = str(exc)
        for start in range(0, len(valid), batch_size):
            chunk = valid[start:start + batch_size]
            probs = self._run_model(np.stack([image for _, image in chunk]))
            for (i, _), row in zip(chunk, probs):
                items[i].result = self._result(_display_path(Path(paths[i])), row)
        return items


def batch_to_frame(items: list[BatchItem], class_names: tuple[str, ...]) -> pd.DataFrame:
    """Tabulate batch results (one row per input; errors kept with an 'error' column)."""
    import pandas as pd  # lazy: only the CLI batch output needs it, not the Streamlit app

    rows = []
    for item in items:
        row: dict[str, Any] = {"image_path": item.image_path, "status": "ok" if item.ok else "error"}
        if item.ok:
            row.update(
                image_path=item.result.image_path,
                predicted_class=item.result.predicted_class,
                confidence=item.result.confidence,
                **{f"prob_{name}": item.result.probabilities[name] for name in class_names},
            )
        row["error"] = item.error
        rows.append(row)
    columns = ["image_path", "status", "predicted_class", "confidence",
               *[f"prob_{name}" for name in class_names], "error"]
    return pd.DataFrame(rows).reindex(columns=columns)


__all__ = [
    "Predictor", "PredictionResult", "BatchItem", "ImageInfo",
    "batch_to_frame", "validate_image_file", "validate_image_bytes",
]
