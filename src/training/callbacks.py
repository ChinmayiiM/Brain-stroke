"""Training callbacks for both transfer-learning stages."""

from __future__ import annotations

from pathlib import Path

import keras

from src.training.settings import TrainingSettings


def checkpoint_path(settings: TrainingSettings, stage: int) -> Path:
    """Best-so-far model of one stage (native ``.keras`` format).

    Stages keep separate checkpoints so the frozen (stage 1) and fine-tuned
    (stage 2) models can be compared on validation data before selection.
    """
    return settings.checkpoints_dir / f"{settings.model_name}_stage{stage}_best.keras"


def build_callbacks(
    settings: TrainingSettings,
    stage: int,
    run_dir: Path,
) -> list[keras.callbacks.Callback]:
    """Callbacks for one stage.

    * EarlyStopping — stops when the monitored validation metric stalls and
      restores the best weights of the stage.
    * ModelCheckpoint — writes the stage's best model only on improvement
      (``save_best_only``), never every epoch.
    * ReduceLROnPlateau — lowers the LR on a plateau before early stopping triggers.
    * CSVLogger — per-epoch history for later plots; TerminateOnNaN — fail fast.
    """
    settings.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    es, lr = settings.early_stopping, settings.reduce_lr

    return [
        keras.callbacks.EarlyStopping(
            monitor=es["monitor"],
            mode=es["mode"],
            patience=int(es["patience"]),
            min_delta=float(es.get("min_delta", 0.0)),
            restore_best_weights=bool(es.get("restore_best_weights", True)),
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path(settings, stage)),
            monitor=settings.checkpoint.monitor,
            mode=settings.checkpoint.mode,
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=lr["monitor"],
            mode=lr["mode"],
            factor=float(lr["factor"]),
            patience=int(lr["patience"]),
            min_lr=float(lr["min_lr"]),
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(run_dir / f"history_stage{stage}.csv")),
        keras.callbacks.TerminateOnNaN(),
    ]
