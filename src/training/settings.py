"""Typed, validated view of ``config.yaml`` for the training pipeline.

Every tunable value used by data loading, model building, callbacks and the
training loop is read here once, so no module hard-codes hyperparameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.preprocessing.image_ops import PreprocessingConfig
from src.utils.config import resolve_path


@dataclass(frozen=True)
class StageSettings:
    epochs: int
    learning_rate: float
    unfreeze_top_n_layers: int = 0


@dataclass(frozen=True)
class MonitorSettings:
    monitor: str
    mode: str


@dataclass(frozen=True)
class TrainingSettings:
    # data
    class_names: tuple[str, ...]
    metadata_file: Path
    sequences: tuple[str, ...] | None
    cache: bool
    cache_max_mb: float
    shuffle_buffer: int | None
    preprocessing: PreprocessingConfig
    augmentation: dict
    # model
    model_name: str
    backbone_weights: str | None
    dropout: float
    # training
    seed: int
    batch_size: int
    class_weighting: str | None
    stage1: StageSettings
    stage2: StageSettings
    early_stopping: dict
    checkpoint: MonitorSettings
    reduce_lr: dict
    model_selection: dict
    # artifacts
    checkpoints_dir: Path
    best_dir: Path
    final_dir: Path
    training_reports_dir: Path
    model_reports_dir: Path
    results_dir: Path

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def is_binary(self) -> bool:
        return self.num_classes == 2

    @property
    def input_shape(self) -> tuple[int, int, int]:
        height, width = self.preprocessing.image_size
        return (height, width, self.preprocessing.channels)

    @classmethod
    def from_config(cls, config: dict) -> "TrainingSettings":
        data, pipeline, model, training, artifacts = (
            config["data"], config["pipeline"], config["model"], config["training"], config["artifacts"]
        )
        class_names = tuple(data["class_names"])
        if len(class_names) < 2:
            raise ValueError(f"Need at least 2 classes, got {class_names}")
        if model["backbone"] != "EfficientNetB0":
            raise ValueError(f"Only EfficientNetB0 is supported, config says {model['backbone']}")
        if not 0.0 <= float(model["dropout"]) < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")

        stage2 = training["stage2"]
        settings = cls(
            class_names=class_names,
            metadata_file=resolve_path(pipeline["metadata_file"]),
            sequences=tuple(pipeline["sequences"]) if pipeline.get("sequences") else None,
            cache=bool(pipeline.get("cache", False)),
            cache_max_mb=float(pipeline.get("cache_max_mb", 0)),
            shuffle_buffer=pipeline.get("shuffle_buffer"),
            preprocessing=PreprocessingConfig.from_config(config),
            augmentation=dict(config["augmentation"]),
            model_name=str(model["name"]),
            backbone_weights=model.get("weights"),
            dropout=float(model["dropout"]),
            seed=int(config["project"]["seed"]),
            batch_size=int(training["batch_size"]),
            class_weighting=training.get("class_weighting"),
            stage1=StageSettings(int(training["stage1"]["epochs"]), float(training["stage1"]["learning_rate"])),
            stage2=StageSettings(
                int(stage2["epochs"]), float(stage2["learning_rate"]), int(stage2["unfreeze_top_n_layers"])
            ),
            early_stopping=dict(training["early_stopping"]),
            checkpoint=MonitorSettings(training["checkpoint"]["monitor"], training["checkpoint"]["mode"]),
            reduce_lr=dict(training["reduce_lr_on_plateau"]),
            model_selection=dict(
                training.get("model_selection", {"metric": "macro_recall", "tie_breaker": "loss", "tie_tolerance": 1e-6})
            ),
            checkpoints_dir=resolve_path(artifacts["checkpoints_dir"]),
            best_dir=resolve_path(artifacts["best_dir"]),
            final_dir=resolve_path(artifacts["final_dir"]),
            training_reports_dir=resolve_path(artifacts["training_reports_dir"]),
            model_reports_dir=resolve_path(artifacts["model_reports_dir"]),
            results_dir=resolve_path(artifacts.get("results_dir", "results")),
        )
        if settings.stage2.learning_rate >= settings.stage1.learning_rate:
            raise ValueError("Fine-tuning learning rate must be smaller than the stage-1 learning rate")
        return settings
