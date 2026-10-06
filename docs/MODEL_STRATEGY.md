# Model Strategy — EfficientNetB0 Transfer Learning

## Why EfficientNetB0

- **Transfer learning over training from scratch.** Medical imaging datasets are typically small (hundreds to a few thousand images). A large CNN trained from scratch would overfit badly. ImageNet-pretrained features (edges, textures, shapes) transfer well as a starting point.
- **Parameter efficiency.** ~4M backbone parameters — trains on a CPU-only Windows machine in reasonable time, and inference in the Streamlit app is fast.
- **Native 224×224 input** matches common MRI slice export resolutions after resizing.
- Available directly in `keras.applications` — no extra dependency.

## Planned Architecture

```
Input (224, 224, 3)  — raw pixels in [0, 255]
  ↓
[tf.data, training split only] augmentation: flip, rotation, zoom, translation, contrast, brightness
  ↓
EfficientNetB0 (include_top=False, weights="imagenet")
  ↓  (includes its own Rescaling + Normalization layers)
GlobalAveragePooling2D
  ↓
Dropout (0.3)
  ↓
Dense(3, softmax)   — derived from data.class_names (2 classes → Dense(1, sigmoid))
```

Notes:
- **No manual `/255` scaling.** Keras' EfficientNet has built-in input normalisation; dividing by 255 is a common silent bug that degrades accuracy.
- **Grayscale MRI → 3 channels** by replication, required for ImageNet weights.
- Classes (Phase 2): Haemorrhagic / Ischemic / Normal → multi-class softmax + categorical cross-entropy on one-hot labels.
- Augmentation runs in the tf.data training pipeline, not inside the model, so saved models contain no random layers.

## Two-Stage Training

| | Stage 1 — Feature extraction | Stage 2 — Fine-tuning |
|---|---|---|
| Backbone | Fully frozen | Top `unfreeze_top_n_layers` (20) backbone layers unfrozen |
| BatchNorm layers | Frozen | **Kept frozen** (`training=False`) — unfreezing BN on small datasets destroys pretrained statistics |
| Trainable | Classification head | Head + top backbone blocks |
| Learning rate | 1e-3 (Adam) | 1e-5 (Adam) — small, to avoid wrecking pretrained weights |
| Epochs (max) | 30 | 20 |
| Verified (sanity check) | 0 backbone params trainable; 3,843 head params | 7 backbone layers with weights trainable (block6d_project_conv, block7a_*, top_conv); 1,346,611 trainable params |

The backbone is called with `training=False` so BN layers run in inference mode even after unfreezing.

## Callbacks

| Callback | Purpose |
|---|---|
| `EarlyStopping(monitor="val_loss", patience=6, min_delta=1e-4, restore_best_weights=True)` | Stop when validation stops improving |
| `ModelCheckpoint(save_best_only=True)` → `models/checkpoints/*_stage{1,2}_best.keras` | Keep each stage's best epoch (not the last); the stage is then chosen on validation macro recall |
| `ReduceLROnPlateau(factor=0.2, patience=3)` | Lower LR when progress stalls |
| `CSVLogger` → `reports/training/<run_id>/` | Persist per-epoch history for plots |
| `TerminateOnNaN` | Fail fast on numerical divergence |

## Overfitting Monitoring

- Plot train vs validation **loss and accuracy** per epoch for both stages.
- Warning signs: validation loss rising while training loss falls; large train/val accuracy gap.
- Mitigations available: augmentation, dropout, fewer unfrozen layers, lower LR, class weights, early stopping.
- **The test set is used exactly once**, after all model and hyperparameter decisions are frozen.

## Class Imbalance

If the Phase 2 dataset is imbalanced, `class_weight` is computed from the **training split only** (`sklearn.utils.class_weight.compute_class_weight`). Accuracy alone will not be trusted in that case (see `EVALUATION_PLAN.md`).

## Reproducibility

- Global seed set via `src/utils/reproducibility.py` (Python, NumPy, TensorFlow).
- All hyperparameters live in `config/config.yaml`; the config used for a run is saved alongside its results.

## Out of Scope (this project)

Segmentation, 3D volumetric models, lesion localisation, and ensembles. Grad-CAM visual explanations may be considered as an optional later enhancement.
