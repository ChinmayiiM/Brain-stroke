# Data Pipeline (Phase 2)

All numbers below were produced by `scripts/analyze_dataset.py` and
`scripts/prepare_dataset.py` on the archive with SHA-256
`21237d2a…c188` (Kaggle version 1, downloaded 2026-10-04). Re-running the
scripts regenerates them; see `reports/data_analysis/` and `dataset/metadata/`.

---

## 1. Dataset selection

| Candidate | Modality / format | Labels | Licence | Patient IDs | Decision |
|---|---|---|---|---|---|
| **Brain Stroke MRI Images** (Kaggle, `mitangshu11`) | 2D MRI slices, PNG/JPG | Haemorrhagic / Ischemic / Normal | "Unknown" on Kaggle | Derivable from filenames | **Selected** |
| ISLES 2022 (Zenodo 7153326) | 3D NIfTI (DWI/ADC/FLAIR), 250 patients | Expert lesion masks (all patients have ischemic stroke) | CC-BY-4.0 | Yes | Not selected — no haemorrhagic or healthy class; would require 3D→2D slice labelling and an extra dependency (nibabel) |
| ATLAS v2.0, ICPSR 38464 | 3D MRI | Lesion masks | Data-use agreements | Yes | Not selected — access agreement, segmentation-oriented |

The Kaggle dataset was selected (project-owner decision) because it is the only
accessible candidate whose labels directly match the project's classification
objective — **stroke type vs. normal on 2D MRI slices** — with no label
re-derivation required. Its weaknesses are documented in §6 rather than hidden.

### What the archive actually contains (verified)

The archive holds 2,127 files in three top-level folders that are **three
copies of overlapping data**:

| Folder | Content | Used? |
|---|---|---|
| `Normal _ Stroke Patient Details/` | Raw exports per patient and sequence, **with burned-in DICOM text** (patient name, DOB, scan date, scanner) | No |
| `Dataset_MRI_Folder/` | Same scans grouped by class/patient/sequence, burned-in text still present; plus `New_Ischemic_Stroke_2/` (141 RGBA images, no patient identity, different source) | No |
| `Stroke_classification/` | Author's cleaned version: burned-in text removed; flat `<class>/` folders; filenames encode patient, sequence and slice | **Yes — sole source** |

Rationale: burned-in text is both personal data and a shortcut feature a CNN
can learn (e.g. scanner name ↔ class). `New_Ischemic_Stroke_2` is excluded
because without patient identity it cannot be split leakage-safely, and the
dataset author also left it out of `Stroke_classification`.

### Label mapping

Labels are taken verbatim from the dataset (`Haemorrhagic`, `Ischemic`,
`Normal`) → 3-class problem, label indices `0, 1, 2` in that order. Each label
was cross-checked against the label embedded in the filename (0 mismatches).
The labels were assigned by the dataset author; their clinical provenance is
not documented.

---

## 2. Analysis (raw `Stroke_classification`, before cleaning)

| | Haemorrhagic | Ischemic | Normal | Total |
|---|---:|---:|---:|---:|
| Images | 186 (30.2 %) | 30 (4.9 %) | 399 (64.9 %) | **615** |
| Patients | 5 | 5 | 6 | **16** |
| Images / patient | 29 – 50 | 3 – 11 | 33 – 103 | |

| Sequence | Haemorrhagic | Ischemic | Normal |
|---|---:|---:|---:|
| DWI | 32 | 19 | 108 |
| GRE | 27 | 0 | 93 |
| SWI | 127 | 0 | 198 |
| T2 | 0 | 6 | 0 |
| FLAIR | 0 | 5 | 0 |

* Format: 615 × PNG, mode RGB, all with grayscale content (R = G = B).
* Dimensions: 159 × 256×256, 456 × 512×512 (all square).
* Corrupted / unreadable: 0 · empty: 0 · unparseable filenames: 0 ·
  label mismatches: 0 · duplicate filenames: 0 · missing class folders: none.
* **Exact duplicates:** 18 files in 6 groups (12 redundant copies) — all from
  one Normal patient (SWI slices 34–39, each stored as `X`, `X(1)`, `X(2)`).
* **Near-duplicates** (pHash Hamming ≤ 4): 786 pairs, 777 of them adjacent
  slices of the same patient (expected). 9 pairs span different patients;
  these were inspected visually and are distinct scans with similar slice
  geometry (pHash false positives) → reported, not removed.

Figures: `reports/data_analysis/{class_distribution, images_per_patient,
sequence_by_class, image_dimensions, sample_images}.png`.

---

## 3. Cleaning

Implemented in `src/data/cleaning.py`. Source files are never modified or
deleted; decisions are recorded per file in `dataset/metadata/cleaning_log.csv`.

| Check | Action | Found |
|---|---|---:|
| Class folder not in `class_names` / missing class folder | exclude / abort | 0 |
| Unsupported extension | exclude | 0 |
| Missing, empty (< 1 KB) | exclude | 0 |
| Corrupted / undecodable (PIL `verify` + full decode) | exclude | 0 |
| Filename unparseable (no patient ID → cannot split safely) | exclude | 0 |
| Filename label ≠ folder label | exclude | 0 |
| Dimensions < 64 px or > 4096 px | exclude | 0 |
| Aspect ratio > 2, colour content | **flag only** | 0 |
| Exact pixel duplicate, same label | keep first (sorted), exclude rest | **12** |
| Exact pixel duplicate, conflicting labels | exclude all copies | 0 |
| Near-duplicate | **report only** | 786 pairs |

Result: **615 → 603 images**. Kept images are copied byte-for-byte (no
re-encoding, no pixel changes) to `dataset/processed/<class>/` under
pseudonymous names such as `HAE-P01_DWI_s004.png`.

**Pseudonymisation.** Source filenames contain what appear to be real patient
names. Patients are mapped to IDs (`HAE-P01`, `ISC-P03`, `NOR-P06`, …);
no name appears in any generated file, log, metadata or figure.

---

## 4. Train / validation / test split

* **Patient-level**: every patient's images go to exactly one split
  (`src/data/splitting.py`). An image-level split would place adjacent,
  near-identical slices of the same patient in train and test.
* **Class-stratified**: patients are partitioned per class; each split gets
  ≥ 1 patient of every class.
* **Ratio target 70 / 15 / 15** (by images). With only 5–6 patients per
  class, exact ratios are impossible; the assignment closest to the target
  is found by exhaustive search, ties broken by `seed = 42`. Deterministic.

| Split | Haemorrhagic | Ischemic | Normal | Total | Share |
|---|---:|---:|---:|---:|---:|
| train | 128 (3 pts) | 21 (3 pts) | 285 (3 pts) | 434 | 72.0 % |
| validation | 29 (1 pt) | 5 (1 pt) | 67 (2 pts) | 101 | 16.7 % |
| test | 29 (1 pt) | 4 (1 pt) | 35 (1 pt) | 68 | 11.3 % |

Patient → split map: `dataset/metadata/patient_splits.csv`.
Verified: 0 patients and 0 identical images in more than one split.

Why 70/15/15 rather than 80/10/10 (both were run): for Haemorrhagic and
Ischemic the patient assignment is identical (3 / 1 / 1). The only difference
is Normal: 80/10/10 moves one Normal patient from validation to train
(4 / 1 / 1 instead of 3 / 2 / 1). Normal is already the majority class, so
the extra training patient adds little, whereas a second validation patient
makes early stopping and model selection less dependent on a single person.

---

## 5. Preprocessing & augmentation

### Preprocessing (`src/preprocessing/image_ops.py`, shared by training and inference)

1. **Decode** with Pillow (path or raw bytes), honour EXIF orientation.
2. **RGB conversion** — grayscale replicated to 3 channels (required by
   ImageNet weights); alpha dropped.
3. **Resize** to 224 × 224 with `cv2.INTER_AREA` (best for downscaling from
   256/512 px). All sources are square, so no padding/cropping is needed.
4. **float32, range [0, 255]** — *no* `/255`: Keras' EfficientNetB0 contains
   its own Rescaling + Normalization layers, and
   `keras.applications.efficientnet.preprocess_input` is an identity (verified).

Intentionally **not** applied: histogram equalisation/CLAHE, denoising,
sharpening, skull stripping, intensity windowing — they alter the intensity
patterns that distinguish lesions and sequences.

Resizing happens at load time, not on disk, so `dataset/` keeps original
resolution and the input size remains a config change.

### Augmentation (`src/preprocessing/augmentation.py`, training split only)

| Transform | Setting | Why |
|---|---|---|
| Horizontal flip | on | Hemispheres are mirror-plausible; stroke occurs on either side |
| Vertical flip | **off** | Anatomically impossible orientation |
| Rotation | ±10.8° (`0.03`) | Head tilt in the scanner |
| Zoom | ±10 % | Head size / field-of-view variation |
| Translation | ±5 % | Head position within the field of view |
| Contrast | ±10 % | Scanner / windowing differences |
| Brightness | ±5 % | Scanner gain differences |

Excluded: shear/elastic deformation, noise, cutout, colour jitter — they can
create or erase lesion-like structures. Fill value 0 matches the black MRI
background. Keras random layers are identity ops unless `training=True`, so
augmentation cannot leak into validation, test or inference (verified by test).

---

## 6. Known limitations (must be considered in Phases 3–5)

1. **Very few patients (16).** Test = one patient per class. Test metrics will
   have very wide uncertainty and say little about generalisation. Patient-level
   (grouped) cross-validation is recommended for model assessment.
2. **Severe imbalance.** Ischemic is 4.9 % of images (21 train / 5 val / 4 test).
   Class weighting and macro-averaged metrics are mandatory; per-class recall on
   4 test images is not statistically meaningful.
3. **Sequence ↔ class confounding.** T2/FLAIR occur only in Ischemic; SWI/GRE
   never occur in Ischemic. In validation/test, SWI appears only in Haemorrhagic
   (the Normal patients holding SWI ended up in train). A model can score well by
   recognising the *sequence* rather than the pathology. Phase 5 must report
   metrics per sequence; restricting to sequences shared by all classes (DWI) is
   an option to evaluate.
4. **Label provenance undocumented**; no radiologist annotations or lesion masks.
5. **Licence "Unknown"** on Kaggle — use limited to non-commercial academic work;
   the dataset is not redistributed (git-ignored).
6. **Residual overlay fragments** (e.g. `[A]`, `[R]` orientation markers) remain
   on some cleaned images.
7. Patient pseudonym mapping relies on name spelling in filenames; two
   spellings of one person would be counted as two patients (none observed in
   `Stroke_classification`).
