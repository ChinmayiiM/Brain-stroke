# Dataset Requirements

Criteria defined in Phase 1. The selected dataset (Phase 2: *Brain Stroke MRI Images*, Kaggle) is assessed against them at the end of this document; full details in [DATA_PIPELINE.md](DATA_PIPELINE.md).

## Mandatory Requirements

| Requirement | Why it matters | How it will be verified (Phase 2) |
|---|---|---|
| **Brain MRI images** (not CT unless the project scope is explicitly changed) | Model and documentation are MRI-specific; CT and MRI differ in contrast characteristics | Inspect source documentation and sample images |
| **Correct, trustworthy labels** | Label noise sets a ceiling on achievable performance | Prefer datasets with documented, clinician-provided labels; spot-check samples |
| **Clear licence permitting academic use** | Legal/ethical use | Read dataset licence |
| **Consistent image format** | Simplifies decoding; avoids silent failures | Scan extensions, decode every file, record mode (L/RGB) and resolution |
| **Sufficient samples per class** | Too few samples → unstable training and meaningless test metrics | Count images per class |
| **Reasonable class balance** | Severe imbalance makes accuracy misleading | Compute class ratios; plan class weights if needed |
| **No patient/image leakage between splits** | Leakage inflates test scores and invalidates results | Exact + near-duplicate detection (file hash / perceptual hash); group by patient ID if available |
| **Proper train / validation / test separation** | Unbiased model selection and final evaluation | Stratified split (70/15/15 default), grouped by patient when possible |

## Data Leakage — Specific Risks

- **Multiple slices from the same patient** placed in both train and test → model learns the patient, not the pathology. If patient IDs exist, split by patient (`GroupShuffleSplit` / `StratifiedGroupKFold`).
- **Duplicate or augmented copies** already present in public datasets (common on Kaggle). These must be detected and de-duplicated **before** splitting.
- **Pre-split datasets** must still be checked for cross-split duplicates.

## Expected Directory Layout After Phase 2

```
dataset/
├── raw/          # original download, never modified
├── processed/    # cleaned/standardised copies (if needed)
├── train/<class_name>/*.png
├── validation/<class_name>/*.png
├── test/<class_name>/*.png
└── metadata/dataset_metadata.csv
```

## Assessment of the Selected Dataset (Phase 2)

| Requirement | Status | Evidence |
|---|---|---|
| Brain MRI images | ✅ | DWI, GRE, SWI, T2, FLAIR axial slices |
| Correct, trustworthy labels | ⚠️ | Labels consistent between folder and filename (0 mismatches); clinical provenance undocumented |
| Licence permitting academic use | ⚠️ | Listed as "Unknown" on Kaggle — academic, non-commercial use only; not redistributed |
| Consistent image format | ✅ | 615/615 PNG, grayscale content, square 256 or 512 px |
| Sufficient samples per class | ❌ | Ischemic: 30 images from 5 patients |
| Reasonable class balance | ❌ | 30.2 % / 4.9 % / 64.9 % (Haemorrhagic / Ischemic / Normal) |
| No duplicate / patient leakage | ✅ | 12 redundant exact copies removed; patient-level split verified leak-free |
| Proper train/val/test separation | ✅ | Patient-level, class-stratified, seeded |
| *(added)* Acquisition not confounded with class | ❌ | Sequence distribution differs strongly by class — see DATA_PIPELINE.md §6 |
