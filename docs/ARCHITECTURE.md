# System Architecture

The system has **two independent pipelines** that share only two things:

1. the **preprocessing module** (`src/preprocessing`) — so an image is transformed identically at training and inference time (prevents train/serve skew), and
2. the **saved model artefact** (`models/*.keras`) plus its class-name mapping.

The Streamlit app never imports training code, and training never imports app code.

---

## 1. Inference Pipeline (runtime)

```
User
  ↓
Streamlit Application            app/
  ↓
MRI Image Upload                 app/            (st.file_uploader)
  ↓
Image Validation                 src/preprocessing  (format, decodable, size, channels)
  ↓
Preprocessing                    src/preprocessing  (resize 224×224, 3-channel, float32)
  ↓
EfficientNetB0                   src/inference      (load saved .keras model once, cached)
  ↓
Classification                   src/inference      (softmax over dataset classes)
  ↓
Prediction                       src/inference      (top class + per-class probabilities)
  ↓
Result Display                   app/            (label, confidence, disclaimer)
```

| Stage | Responsibility | Failure handling |
|---|---|---|
| Upload | Accept a single file of an allowed extension | Reject unsupported types before decoding |
| Validation | Decode safely, check non-empty, minimum resolution, convert mode | Show a clear user-facing error; never pass a bad tensor to the model |
| Preprocessing | Exactly the same function used in training | — |
| Model | Loaded once via `st.cache_resource` | Fail fast with message if model file is missing |
| Prediction | Return class + probability vector | — |
| Display | Show result **with** the mandatory disclaimer | — |

**Mandatory wording** (stored in `config.yaml → inference.disclaimer`):

> AI-assisted prediction — not a medical diagnosis.

The UI must never say "diagnosis", "you have…", or imply the output replaces a clinician.

---

## 2. Training Pipeline (offline)

```
Dataset                          dataset/raw         (untouched original)
  ↓
Data Cleaning                    src/data            (corrupt files, duplicates, label checks)
  ↓
Preprocessing                    src/preprocessing   (resize, channel normalisation)
  ↓
Train / Validation / Test        src/data            → dataset/{train,validation,test}/<class>/ (patient-level)
  ↓
EfficientNetB0 Transfer Learning src/training        (ImageNet weights, custom head)
  ↓
Training                         src/training        (stage 1 frozen → stage 2 fine-tune)
  ↓
Evaluation                       src/evaluation      → reports/
  ↓
Saved Model                      models/             (best checkpoint, .keras format)
```

---

## 3. Module Dependency Rules

```
            ┌──────────────┐
            │  src/utils   │  ← used by everything
            └──────┬───────┘
                   │
         ┌─────────┴──────────┐
         │ src/preprocessing  │  ← shared by both pipelines
         └───┬────────────┬───┘
             │            │
   TRAINING  │            │  INFERENCE
   src/data  │            │  src/inference
   src/training           │       ↑
   src/evaluation         │     app/
             │            │
             └──► models/ ◄┘   (artefact boundary)
```

- `app/` → may import `src.inference`, `src.preprocessing`, `src.utils` only.
- `src/inference` → must not import `src.training` / `src.data`.
- `src/training` → must not import `app/`.
- All tunable values come from `config/config.yaml`, not hard-coded constants.

---

## 4. Directory Responsibilities

| Directory | Purpose |
|---|---|
| `app/` | Streamlit UI only — upload, display, disclaimer. No ML logic. |
| `src/data/` | Dataset discovery, integrity checks, duplicate detection, stratified/grouped splitting. |
| `src/preprocessing/` | Image decoding, validation, resizing, augmentation layers. Single source of truth for image transforms. |
| `src/training/` | Model builder (EfficientNetB0 + head), callbacks, two-stage training loop. |
| `src/evaluation/` | Metrics, classification report, confusion matrix, training-curve plots. |
| `src/inference/` | Load trained model, predict on one preprocessed image. |
| `src/utils/` | Config loading, path resolution, seeding, logging. |
| `config/` | `config.yaml` — every hyperparameter and path. |
| `dataset/raw/` | Original dataset as downloaded. Treated as read-only. |
| `dataset/processed/` | Cleaned, de-duplicated, pseudonymised copy of every included image, by class. |
| `dataset/train/`, `validation/`, `test/` | Patient-level split folders (`<class>/`) consumed by training. Generated. |
| `dataset/metadata/` | `dataset_metadata.csv`, `cleaning_log.csv`, `patient_splits.csv`, `preparation_summary.json`. |
| `models/` | Saved models and checkpoints (git-ignored). |
| `reports/` | Metrics JSON/CSV, plots, training history (git-ignored). |
| `notebooks/` | Exploratory analysis only — production logic lives in `src/`. |
| `scripts/` | Command-line entry points: `verify_environment`, `download_dataset`, `analyze_dataset`, `prepare_dataset` (train/evaluate later). |
| `tests/` | Pytest unit tests. |
| `docs/` | Design documentation. |
