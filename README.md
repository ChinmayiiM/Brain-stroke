# Brain Stroke Detection using Brain MRI Images

AI-assisted classification of 2D brain MRI slices into **Haemorrhagic**, **Ischemic** or **Normal** with an EfficientNetB0 transfer-learning model, served through a Streamlit web application and a command-line tool.

> ⚠️ **Important disclaimer.** This project is an AI-assisted research/educational prototype and is not a medical diagnostic system. Predictions should not be used as a substitute for evaluation by a qualified healthcare professional. It is not a medical device, has not been clinically validated, and anyone with stroke symptoms must seek emergency care immediately.

---

## Quick Start (Windows, from the project root)

```powershell
.\.venv\Scripts\Activate.ps1                     # activate the environment (setup: §14)
streamlit run app/streamlit_app.py               # web app → http://localhost:8501
python -m src.inference.cli --image dataset/test/Ischemic/ISC-P05_DWI_s020.png    # CLI prediction
python scripts/run_integration_checks.py         # end-to-end checks → results/integration_test_report.md
pytest                                           # unit + application tests
```

The trained model must exist at `models/best/efficientnetb0_stroke_mri.keras`. Model binaries are not in git; §14 shows how to recreate them.

---

## 1. Project Overview

The project is an end-to-end, modular deep-learning pipeline:

1. **Data.** It downloads a public brain MRI dataset, audits it, cleans and pseudonymises it, and splits it at patient level.
2. **Training.** It trains EfficientNetB0 with transfer learning in two stages: a frozen backbone, then controlled fine-tuning.
3. **Evaluation.** It selects the model on validation data only and evaluates it once on an untouched test split.
4. **Inference.** It serves predictions through a validated inference pipeline, used by both a CLI and a Streamlit app.
5. **Testing.** It is covered by unit, application and end-to-end integration tests.

## 2. Problem Statement

Stroke is a time-critical emergency, and brain MRI is a key imaging modality for assessing it. Interpreting scans requires specialist expertise and time. The project investigates whether an image classifier, trained on a small public dataset, can label MRI slices by the dataset's categories. It also asks what such a model can and **cannot** be trusted to do.

## 3. Objective

Build a reproducible, honestly evaluated prototype that:

- classifies a single 2D brain MRI slice into the dataset's classes (Haemorrhagic / Ischemic / Normal);
- reports the model's class probabilities, not a diagnosis;
- uses identical preprocessing in training, evaluation, the CLI and the web app;
- documents its data, metrics and limitations transparently.

## 4. Important Disclaimer

This project is an AI-assisted research/educational prototype and is not a medical diagnostic system. Predictions should not be used as a substitute for evaluation by a qualified healthcare professional. The **model confidence** shown is the model's softmax probability, not a calibrated measure of clinical certainty. The model has made confident errors on test images.

## 5. Key Features

- **Data:** reproducible preparation with download and checksum, audit, duplicate detection, logged cleaning, and pseudonymisation of patient names.
- **Splitting:** patient-level, class-stratified train/validation/test split with automated leakage checks.
- **Preprocessing:** one shared function (RGB conversion, resize to 224×224, EfficientNet-compatible input) used everywhere.
- **Augmentation:** for the training split only (flip, small rotation, zoom, translation, contrast, brightness).
- **Model:** EfficientNetB0 with ImageNet weights, class-weighted loss, a frozen stage and then fine-tuning of the top layers.
- **Selection and evaluation:** stage chosen on validation metrics. Test metrics include per-class sensitivity, specificity, ROC-AUC, confusion matrix, and per-sequence and per-patient breakdowns.
- **Inference:** validation of every input; the model is loaded once and checked against its metadata; structured results.
- **Interfaces:** a CLI for single images and batches (CSV) that shows all three class probabilities, and a minimal Streamlit app (upload → preview → *Analyze MRI* → **STROKE / NO STROKE** with confidence).
- **Error handling:** graceful handling of missing, unsupported, corrupted, empty, 16-bit or tiny images, a missing model, and class-mapping, preprocessing or output mismatches.
- **Tests:** a pytest suite and an integration-check script that writes a test report.

## 6. Technology Stack

| Area | Technology (version used) |
|---|---|
| Language | Python 3.11.9 |
| Deep learning | TensorFlow 2.21.0, Keras 3.15.1, EfficientNetB0 (`keras.applications`) |
| Arrays / tables | NumPy 2.4.6, Pandas 3.0.6 |
| Images | Pillow 12.3.0, OpenCV (headless) 5.0.0 |
| Metrics | scikit-learn 1.9.1 |
| Plots | Matplotlib 3.11.2 |
| Web app | Streamlit 1.65.0 |
| Configuration | YAML (PyYAML 6.0.3) |
| Testing | pytest 9.1.1, `streamlit.testing` (AppTest) |
| Environment | Windows 11, CPU (no GPU), VS Code, `venv` |

## 7. System Architecture

```
TRAINING (offline)
 Kaggle archive ─► src/data: audit · clean · pseudonymise · patient-level split ─► dataset/{train,validation,test}
   ─► src/training: tf.data (+ train-only augmentation) ─► EfficientNetB0
        stage 1: frozen backbone, train head ─► stage 2: fine-tune top layers
   ─► src/evaluation: choose stage on VALIDATION ─► evaluate once on TEST ─► results/
   ─► models/best/efficientnetb0_stroke_mri.keras + model_metadata.json

INFERENCE (online)
 User ─► app/streamlit_app.py  or  python -m src.inference.cli
   ─► src/inference/Predictor: validate image ─► preprocess_image ─► EfficientNetB0 ─► softmax
   ─► PredictionResult (class, confidence, probabilities) ─► UI / CLI output

Shared by both: src/preprocessing/image_ops.preprocess_image and the saved model + metadata.
```

The Streamlit app contains **no** preprocessing or model code. It calls `Predictor.predict_bytes()` from the inference package. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 8. Dataset

| | |
|---|---|
| Name / source | **Brain Stroke MRI Images**, Kaggle [`mitangshu11/brain-stroke-mri-images`](https://www.kaggle.com/datasets/mitangshu11/brain-stroke-mri-images) (version 1, archive SHA-256 `21237d2a…c188`) |
| Licence | Listed as *Unknown* → academic, non-commercial use only; not redistributed |
| Classes | `Haemorrhagic`, `Ischemic`, `Normal` (label indices 0, 1, 2) |
| Images used | Author's cleaned `Stroke_classification/` folder (burned-in patient text removed): 615 images, 16 patients |
| After cleaning | **603** images (12 redundant exact duplicates removed; 0 corrupted) |
| Modality | 2D axial MRI slices: DWI, GRE, SWI, T2, FLAIR; PNG, 256×256 or 512×512 |

**Patient-level split** (seed 42, target 70/15/15; no patient or identical image in more than one split, verified):

| Split | Haemorrhagic | Ischemic | Normal | Total | Patients |
|---|---:|---:|---:|---:|---:|
| train | 128 | 21 | 285 | 434 | 9 |
| validation | 29 | 5 | 67 | 101 | 4 |
| test | 29 | 4 | 35 | 68 | 3 |

**Dataset structure:**
- `dataset/raw/`: download, git-ignored.
- `dataset/processed/<class>/`: cleaned copies.
- `dataset/{train,validation,test}/<class>/`: the splits.
- `dataset/metadata/`: `dataset_metadata.csv`, `cleaning_log.csv`, `patient_splits.csv` (all name-free).

**Known data issues:**
- **Severe imbalance:** Ischemic makes up only 5 % of images.
- **Very few patients:** 16 in total.
- **Sequence confounding:** T2 and FLAIR occur only in Ischemic, and SWI and GRE never do.
- **Unknown label provenance.**

Full audit: [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md); figures in `reports/data_analysis/`.

## 9. Model Architecture

```
Input MRI file
 └─ preprocessing (shared): decode → validate → RGB (grayscale replicated) → resize 224×224 (INTER_AREA)
                            → float32 in [0, 255]   (no /255: EfficientNet normalises internally)
     └─ EfficientNetB0 backbone (ImageNet, include_top=False, run with training=False)  → 7×7×1280
          └─ GlobalAveragePooling2D → 1280
               └─ Dropout(0.3)
                    └─ Dense(3, softmax)  → P(Haemorrhagic), P(Ischemic), P(Normal)
                         └─ prediction = argmax; model confidence = max probability
```

The model has 4,053,414 parameters, of which the backbone has 4,049,571. The output layer is derived from the configured classes: a 2-class configuration automatically uses a 1-unit sigmoid head.

## 10. Training Strategy

| | Stage 1: feature extraction | Stage 2: fine-tuning |
|---|---|---|
| Backbone | fully frozen | top 20 backbone layers unfrozen: 7 with weights (`block6d_project_conv`, `block7a_*`, `top_conv`); **all 49 BatchNorm layers stay frozen** |
| Trainable parameters | 3,843 (head) | 1,346,611 |
| Optimizer / learning rate | Adam 1e-3 | Adam 1e-5 (re-compiled after unfreezing) |
| Max epochs / actual | 30 / 28 (early stop; best epoch 22) | 20 / 8 (early stop; best epoch 30) |

- **Loss:** categorical cross-entropy with *balanced* class weights computed from the training split (Haemorrhagic 1.130, Ischemic 6.889, Normal 0.508).
- **Callbacks:**
  - EarlyStopping on val_loss (patience 6, restore best weights)
  - ModelCheckpoint, saving the best epoch per stage on val_loss
  - ReduceLROnPlateau (×0.2 after 3 epochs)
  - CSVLogger and TerminateOnNaN
- **Stage selection:** the stage with the higher **validation macro recall** wins; ties go to lower validation loss. Stage 2 was selected (0.791 vs 0.763). The decision was recorded before the test set was loaded.
- **Seed:** 42 for Python, NumPy, TensorFlow, shuffling and augmentation.

Full rationale: [docs/MODEL_STRATEGY.md](docs/MODEL_STRATEGY.md).

## 11. Evaluation (actual Phase 4 results)

> These numbers describe the **held-out test split of this research dataset**: 68 images from 3 patients, one per class. They are not clinical performance.

**Stage comparison:**

| Model | Split | Loss | Accuracy | Balanced accuracy | Macro F1 |
|---|---|---:|---:|---:|---:|
| Stage 1 (frozen) | validation | 0.608 | 0.723 | 0.763 | 0.639 |
| Stage 2 (fine-tuned, **selected**) | validation | 0.569 | 0.752 | 0.791 | 0.669 |
| Stage 2 (fine-tuned, **selected**) | **test** | 0.487 | **0.824** | **0.878** | **0.770** |

**Test results per class (selected model):**

| Class | Support | Precision | Recall / sensitivity | Specificity | F1 | ROC-AUC (one-vs-rest) |
|---|---:|---:|---:|---:|---:|---:|
| Haemorrhagic | 29 | 0.893 | 0.862 | 0.923 | 0.877 | 0.944 |
| Ischemic | 4 | 0.444 | 1.000 | 0.922 | 0.615 | 1.000 |
| Normal | 35 | 0.871 | 0.771 | 0.879 | 0.818 | 0.870 |
| Macro avg | 68 | 0.736 | 0.878 | | 0.770 | 0.938 |

**Confusion matrix** (rows = actual, columns = predicted: Haemorrhagic / Ischemic / Normal):

| Actual | Haemorrhagic | Ischemic | Normal |
|---|---:|---:|---:|
| Haemorrhagic | 25 | 0 | 4 |
| Ischemic | 0 | 4 | 0 |
| Normal | 3 | 5 | 27 |

**How to read these numbers:**

- **Sequence shortcut.** All 12 errors are on DWI and GRE. Every SWI image (Haemorrhagic-only in test) and every FLAIR image (Ischemic-only) is correct. On DWI alone, the only sequence present in every class, accuracy is 0.815 and balanced accuracy 0.836 (27 images). The headline score is partly inflated by sequence differences between classes.
- **Ischemic results are thin.** Recall of 1.000 rests on 4 slices from a single patient, and its precision is 0.444.
- **Overfitting.** The gap between training and validation accuracy was about 0.13–0.18.
- **No confidence intervals:** they cannot be estimated meaningfully with one test patient per class.

**Artifacts:**
- **`results/plots/`:** accuracy, loss and macro-F1 curves, confusion matrix, ROC curves, misclassified examples.
- **`results/metrics/`:** `metrics.json`, `classification_report.txt`, `test_predictions.csv`, `stage_comparison.csv`, `training_history.csv`/`.json`, `model_selection.json`, `training_summary.json`.
- **`results/analysis/`:** misclassifications, per-sequence and per-patient breakdowns, overfitting analysis, leakage checks.

## 12. Inference Pipeline

```
image (path or uploaded bytes)
 ─► validation: supported extension (.png/.jpg/.jpeg) · non-empty · decodable · 8-bit pixel mode · ≥ 64 px
 ─► preprocess_image (identical to training) ─► add batch dimension (1, 224, 224, 3)
 ─► model(x, training=False)   (no dropout, no augmentation)
 ─► probabilities validated (finite, in [0,1], Σ = 1) ─► argmax → class name via model_metadata.json
 ─► PredictionResult {image_path, predicted_class, predicted_index, confidence, probabilities, model_path, disclaimer}
```

- **Final inference model:** `models/best/efficientnetb0_stroke_mri.keras` (27.8 MB), the validation-selected Stage 2 model. The integration checks confirm it is byte-identical to its checkpoint.
- **Other model files:** `models/final/` (end-of-training state) and `models/checkpoints/` (per-stage bests) are kept for reference only.
- **Load-time checks:** the loader verifies class order, input shape, output layer, preprocessing contract and the absence of augmentation layers. Any mismatch is reported as a specific error.

**CLI** (from the project root):

```powershell
python -m src.inference.cli --image <path-to-mri-image>                       # readable output
python -m src.inference.cli --image <path> --json                              # structured JSON
python -m src.inference.cli --image <path> --save-figure results/inference/debug.png   # debug figure
python -m src.inference.cli --dir dataset/test/Ischemic --output results/inference/batch.csv   # batch → CSV
```

- **Input requirements:** one 2D MRI slice as PNG, JPG or JPEG; 8-bit grayscale or RGB; at least 64 px per side. DICOM and NIfTI are not supported.
- **Output:** predicted class, model confidence and every class probability.
- **Exit codes:** `0` OK · `2` invalid image · `3` model or configuration problem.

## 13. Streamlit Application

**Layout:**
- **Sidebar:** branding, then *About*, *How it works* and *Model information* (from `model_metadata.json`).
- **Main area:** header (*Brain Stroke Detection — Brain MRI Image Classification*) and two tabs, **Analyze MRI** and **Model & method**.
- **Toolbar:** `.streamlit/config.toml` sets `client.toolbarMode = "viewer"`, a supported Streamlit option. It hides Streamlit's developer controls such as *Deploy*; the ⋮ menu remains. The second tab covers the model, the pipeline and the real 3-class test metrics with caveats.

**Workflow (Analyze MRI tab):**
1. Under **Upload Brain MRI Image**, choose a PNG/JPG/JPEG file.
2. The uploaded MRI is previewed (left), with its filename, size and format.
3. Click **Analyze MRI** (right).
4. The **Prediction result** card shows **STROKE** or **NO STROKE** and its **Confidence**. No per-class probability table is shown.

**Binary result from the 3-class model.** The model is unchanged and still outputs P(Haemorrhagic), P(Ischemic) and P(Normal). The app only *presents* them in two groups (`config.yaml → inference.binary_view`, `src/inference/binary_view.py`):

```
P(STROKE)    = P(Haemorrhagic) + P(Ischemic)
P(NO STROKE) = P(Normal)
Result       = STROKE if P(STROKE) ≥ 0.5, else NO STROKE      Confidence = probability of the shown result
e.g. 20.49 % + 5.11 % = 25.60 % stroke vs 74.40 % normal → NO STROKE, confidence 74.40 %
```

This is not the 3-class argmax. For example, Haemorrhagic 30 % / Ischemic 25 % / Normal 45 % shows **STROKE 55 %**. The Phase 4 metrics in §11 are 3-class metrics; the binary view was not separately evaluated or tuned.

**Other behaviour:**
- **Error messages:** a missing model disables analysis and shows the reason. Invalid files show a short message, and users never see a stack trace.
- **Model loading:** the `.keras` model is loaded **once** per server process (`st.cache_resource`) and reused for every rerun, upload and click. Its forward pass is compiled once (`tf.function`) and warmed up at start-up. Results are cleared when a new image is uploaded.
- **Timing logs:** model load, warm-up, preprocessing, model and total time per click are written to `logs/streamlit_app.log`.

**Privacy:**
- Uploads are handled **in memory only**; the app writes no files and makes no network calls.
- Streamlit telemetry is disabled in `.streamlit/config.toml`.
- No patient information is requested.
- Uploaded bytes live in Streamlit's in-memory session storage until the session ends.

## 14. Installation

```powershell
# 1. Obtain the project (copy the folder or clone the repository), then from the project root:
# 2. Create a virtual environment (Python 3.11 recommended; TensorFlow does not support every newer version)
py -3.11 -m venv .venv                 # Linux/macOS: python3.11 -m venv .venv

# 3. Activate it
.\.venv\Scripts\Activate.ps1           # PowerShell   (cmd: .venv\Scripts\activate.bat)
                                       # Linux/macOS: source .venv/bin/activate

# 4. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt        # runtime
pip install -r requirements-dev.txt    # + pytest (optional)

# 5. Verify the environment
python scripts/verify_environment.py   # imports, versions, builds EfficientNetB0
```

- **Exact versions:** `requirements.txt` holds compatible version ranges. `requirements-lock.txt` pins the exact versions used to train and test the delivered model; install from it for the closest reproduction.
- **PowerShell activation blocked:** run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- **GPU:** TensorFlow ≥ 2.11 has no native Windows GPU support. Everything here ran on CPU; training took about 45 minutes.

**Recreate data and model** (not stored in git):

```powershell
python scripts/download_dataset.py       # download + SHA-256 check + extract → dataset/raw/
python scripts/analyze_dataset.py        # statistics + figures → reports/data_analysis/
python scripts/prepare_dataset.py        # clean + patient-level split + metadata + verification
python -m src.training.train --sanity-check   # verify data pipeline + model (no training)
python -m src.training.train             # two-stage training + validation-only selection
python -m src.evaluation.evaluate        # leakage checks + one-time test evaluation
python scripts/verify_inference.py       # inference ↔ Phase 4 consistency check
```

**Reproducibility:**
- Data preparation is deterministic. Its metadata CSVs are versioned, so the exact split can be audited.
- Training uses seed 42, but CPU/GPU kernels and library versions can still change results slightly. Bit-identical retraining across machines is not guaranteed.
- `results/metrics/training_summary.json` records every hyperparameter, the class weights, the metadata SHA-256 and the library versions of the delivered run.

## 15. Running the Application

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py
```

Open **http://localhost:8501**, upload an MRI image, click **Analyze MRI** and read the result. Optionally, set the environment variable `STROKE_APP_MODEL_PATH` to use a different `.keras` model. Its `model_metadata.json` must sit next to it.

**Deployment readiness:** the app runs from the project root with relative paths only and has no secrets or API keys.
- **Entry point:** `app/streamlit_app.py`.
- **Dependencies:** `requirements.txt`. `opencv-python-headless` suits servers.
- **Model artifact:** a host such as Streamlit Community Cloud needs the model file and its metadata in the repository. They are git-ignored by default; to include them, add these lines to `.gitignore` (or use Git LFS):
  ```
  !models/best/efficientnetb0_stroke_mri.keras
  !models/best/model_metadata.json
  ```
  The model was trained on a dataset with an *Unknown* licence; check usage rights before publishing it.
- **Dataset:** not needed by the app. The optional demo-sample selector simply hides itself when the images are absent. Do not upload the dataset.
- **Not done:** the app is **not deployed**, and no hosted version exists.

## 16. Project Structure

```
.
├── .streamlit/config.toml     app theme, 20 MB upload limit, telemetry off
├── app/                       Streamlit UI: streamlit_app.py (entry), services.py, ui.py
├── config/config.yaml         single source of paths, classes, preprocessing, training & inference settings
├── dataset/
│   ├── raw/                   downloaded archive + extracted original (git-ignored)
│   ├── processed/             cleaned, pseudonymised images by class (generated)
│   ├── train/ validation/ test/   patient-level split (generated)
│   └── metadata/              dataset_metadata.csv, cleaning_log.csv, patient_splits.csv
├── docs/                      ARCHITECTURE, DATA_PIPELINE, MODEL_STRATEGY, DATASET_REQUIREMENTS, EVALUATION_PLAN
├── models/
│   ├── best/                  efficientnetb0_stroke_mri.keras + model_metadata.json  ← inference model
│   ├── final/                 end-of-training model
│   └── checkpoints/           per-stage best checkpoints
├── reports/                   data_analysis/ (dataset figures), model/ (summaries, sanity check), training/ (run logs)
├── results/
│   ├── plots/                 training curves, confusion matrix, ROC, misclassified examples
│   ├── metrics/               metrics.json, classification report, predictions, history, selection, summary
│   ├── analysis/              error / sequence / patient breakdowns, overfitting, leakage checks
│   ├── inference/             CLI batch outputs, Phase 4 consistency check
│   └── integration_test_report.md / .json
├── scripts/                   verify_environment, download_dataset, analyze_dataset, prepare_dataset,
│                              verify_inference, run_integration_checks
├── src/
│   ├── data/                  discovery, audit, duplicates, cleaning, splitting, statistics, plots
│   ├── preprocessing/         image_ops (shared preprocessing), augmentation
│   ├── training/              settings, data_loader, model, callbacks, class_balance, train, sanity
│   ├── evaluation/            predictions, metrics, plots, selection, leakage, evaluate
│   ├── inference/             model_loader, predictor, cli, visualize, errors
│   └── utils/                 config, logger, reproducibility
├── tests/                     data, preprocessing, training, evaluation, inference, Streamlit app
├── requirements.txt / requirements-dev.txt / requirements-lock.txt
├── pytest.ini
└── .gitignore
```

## 17. Testing

- **`pytest`:** unit tests for data preparation, preprocessing, training components, evaluation logic and inference. Application tests drive the real Streamlit script with the real model through `streamlit.testing.AppTest`: no upload, upload and analyze, a second image, a corrupted file, a missing model, and the sample flow.
- **`python scripts/run_integration_checks.py`:** executes and records every end-to-end check in `results/integration_test_report.md`. It covers:
  - dataset integrity and leakage;
  - bit-identical preprocessing across training, CLI and Streamlit;
  - model integrity and class mapping across all components;
  - CLI predictions matching the Phase 4 evaluation;
  - 15 edge cases;
  - load and prediction latency;
  - live Streamlit server start;
  - privacy and wording scans.

## 18. Limitations

- **Not a diagnosis.** This is a research and educational prototype; a qualified healthcare professional must make every clinical decision.
- **Dataset:**
  - only 16 patients;
  - Ischemic is 5 % of images;
  - label provenance is undocumented;
  - the licence is unknown;
  - some images still carry faint overlay fragments.
- **Test evidence is thin.** One patient per class in the test split, so results are patient-specific and noisy.
- **Shortcut learning.** MRI sequence is confounded with class, and part of the measured performance reflects sequence differences rather than pathology.
- **Generalisation.** The model has not been validated on external scanners, protocols, hospitals or populations. Performance on other MRI images may be much worse.
- **Image quality and format.** Results depend on input quality. Only single 2D 8-bit slices are supported; there is no DICOM, 3D or multi-sequence context.
- **Confidence ≠ certainty.** Softmax confidence is uncalibrated, and the model has been confidently wrong (90.4 % "Normal" on a haemorrhagic slice).
- **External clinical validation** would be required before any clinical use. None has been done.

## 19. Future Enhancements (not implemented)

- Larger, multi-centre datasets with documented, radiologist-confirmed labels and healthy controls.
- Training on sequences present in every class (`pipeline.sequences: [DWI]`) or sequence-balanced data, to remove the shortcut.
- Patient-grouped cross-validation and confidence intervals.
- External validation on independent sites; probability calibration.
- Explainability (e.g. Grad-CAM), clearly labelled as model interpretability, not medical evidence.
- Lesion segmentation or localisation; 3D/multi-slice and DICOM support.
- Secure deployment (authentication, audit logging, data-protection review), model versioning and monitoring.

## Demonstration Guide

1. **Problem and scope:** §2–4, including the disclaimer.
2. **Architecture:** §7 diagram.
3. **Transfer learning and two-stage training:** §9–10.
4. **Results:** `results/plots/accuracy.png`, `loss.png`, `confusion_matrix.png`, and §11 including the sequence caveat.
5. **Live demo:** `streamlit run app/streamlit_app.py`. Upload a test image, click **Analyze MRI**, then try a second image and a corrupted file.
6. **Close with limitations:** §18. Make no clinical claims.

## Development Phases

| Phase | Focus | Status |
|---|---|---|
| 1 | Foundation: structure, environment, architecture | ✅ |
| 2 | Dataset selection, audit, cleaning, leakage-safe splitting | ✅ |
| 3 | Preprocessing, tf.data pipeline, EfficientNetB0 model | ✅ |
| 4 | Two-stage training, fine-tuning, test evaluation | ✅ |
| 5 | Inference pipeline and CLI | ✅ |
| 6 | Streamlit application | ✅ |
| 7 | End-to-end integration testing | ✅ |
| 8 | Finalisation, documentation, deployment readiness | ✅ |
