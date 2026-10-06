# Evaluation Plan

No results exist yet. This document defines **how** the trained model will be evaluated in a later phase.

## Principles

- Final metrics are computed **once**, on the held-out **test split**, after all training and tuning decisions are frozen.
- The validation split is used for early stopping, checkpoint selection and hyperparameter choices — never the test split.
- Every reported number must be reproducible from saved artefacts (`models/`, `reports/`, `config.yaml`).

## Metrics

| Metric | Definition | Why it is reported |
|---|---|---|
| **Accuracy** | Correct predictions / total | Overall performance; misleading alone if classes are imbalanced |
| **Precision** | TP / (TP + FP) per class | How trustworthy a positive prediction is (false-alarm rate) |
| **Recall / Sensitivity** | TP / (TP + FN) per class | **Most important clinically** — a missed stroke (false negative) is the costliest error |
| **F1-score** | Harmonic mean of precision and recall | Single balanced score per class |
| **Confusion Matrix** | Counts of true vs predicted class | Shows exactly which classes are confused |

- Per-class values **and** macro averages will be reported (macro so minority classes weigh equally); weighted averages included for reference.
- Specificity (TN / (TN + FP)) will be reported for the binary case.
- Optional: ROC curve / ROC-AUC if the final task is binary.

## Tooling

- `sklearn.metrics`: `classification_report`, `confusion_matrix`, `precision_recall_fscore_support`, `accuracy_score`
- `matplotlib`: confusion-matrix heatmap (raw counts + row-normalised), training/validation loss and accuracy curves

## Outputs (written to `reports/`)

- `metrics.json` — all scalar metrics
- `classification_report.txt`
- `confusion_matrix.png`
- `training_curves.png`
- Copy of the `config.yaml` used for the run

## Dataset-Specific Requirements (from Phase 2 findings)

- **Per-sequence breakdown** of all metrics (DWI / GRE / SWI / T2 / FLAIR), because MRI sequence is confounded with class.
- **Per-patient breakdown** — the test split contains one patient per class, so results are effectively three case studies.
- **Patient-level cross-validation** (grouped by `patient_id`, stratified by class) to obtain a less noisy performance estimate than the single held-out split.
- Report confidence intervals (e.g. bootstrap over patients) rather than point estimates alone; never report Ischemic recall without its denominator (4 test images).

## As Implemented (Phase 4)

- Stage selection: max validation macro recall, tie → lower validation loss (`src/evaluation/selection.py`), recorded in `results/metrics/model_selection.json` with `test_set_used: false` before evaluation.
- `src/evaluation/evaluate.py` refuses to report test metrics unless `src/evaluation/leakage.py` passes: no test path, hash or patient in train/validation; selection recorded; training entry point never requests the test split.
- Per-sequence and per-patient breakdowns, plus a confound-controlled subset: sequences present in every training class (DWI).
- Confidence intervals were **not** computed. With one test patient per class, patient-level resampling is impossible, and image-level bootstrapping ignores within-patient correlation, giving falsely narrow intervals. Patient-grouped cross-validation is the recommended next step.
- Results: see README "Phase 4: Training & Evaluation Results".

## Interpretation Guidelines

- High accuracy with low recall on the stroke class is a **failure**, not a success.
- Results describe performance on *this dataset's test split only* and do not imply clinical validity or generalisation to other scanners, protocols or populations.
