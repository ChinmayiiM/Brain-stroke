"""Clean, split and materialise the dataset, then verify the result.

Usage (project root, venv active):
    python scripts/prepare_dataset.py

Pipeline:
    raw source ─► audit ─► cleaning rules ─► dataset/processed/<class>/
                                         └─► patient-level split ─► dataset/{train,validation,test}/<class>/
Outputs in dataset/metadata/:
    dataset_metadata.csv   — one row per prepared image (paths are project-relative)
    cleaning_log.csv       — every source file with status, exclusion reason and flags
    patient_splits.csv     — which pseudonymised patient went to which split
    preparation_summary.json
Also writes reports/data_analysis/split_distribution.png.

Re-running regenerates processed/ and the split folders from scratch (deterministic).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cleaning import CleaningRules, apply_cleaning_rules, check_class_folders  # noqa: E402
from src.data.pipeline import scan_source  # noqa: E402
from src.data.preparation import OutputDirs, materialise, reset_generated_dirs, verify_prepared_dataset  # noqa: E402
from src.data.splitting import SPLITS, patient_level_split  # noqa: E402
from src.data.visualization import plot_split_distribution  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402
from src.utils.reproducibility import set_global_seed  # noqa: E402

logger = get_logger("prepare_dataset")

CLEANING_LOG_COLUMNS = [
    "pseudo_name", "folder_label", "patient_id", "sequence", "slice_number", "slice_variant",
    "image_width", "image_height", "file_sha256", "status", "exclusion_reason", "flags",
]


def main() -> int:
    config = load_config()
    seed = int(config["project"]["seed"])
    set_global_seed(seed)
    class_names = list(config["data"]["class_names"])
    rules = CleaningRules.from_config(config)
    dirs = OutputDirs.from_config(config, resolve_path)

    scan = scan_source(config)
    missing, extra = check_class_folders(scan.class_folders, rules.class_names)
    if missing:
        logger.error("Missing class folders in source: %s — aborting", missing)
        return 1
    if extra:
        logger.warning("Unexpected folders in source (their files will be excluded): %s", extra)

    cleaned = apply_cleaning_rules(scan.records, rules, scan.exact_duplicates)
    included = cleaned[cleaned["status"] == "included"]
    split_df = patient_level_split(included, config["data"]["split"], seed)

    reset_generated_dirs(dirs)
    metadata = materialise(split_df, dirs, class_names)
    metadata.to_csv(dirs.metadata / "dataset_metadata.csv", index=False)
    cleaned[CLEANING_LOG_COLUMNS].to_csv(dirs.metadata / "cleaning_log.csv", index=False)
    patient_splits = (
        split_df.groupby(["patient_id", "folder_label", "split"]).size().rename("images").reset_index()
        .sort_values(["folder_label", "split", "patient_id"])
    )
    patient_splits.to_csv(dirs.metadata / "patient_splits.csv", index=False)

    reports_dir = resolve_path(config["paths"]["reports"]) / "data_analysis"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_split_distribution(metadata.rename(columns={"class_name": "folder_label"}), class_names, list(SPLITS),
                            reports_dir / "split_distribution.png")

    problems = verify_prepared_dataset(metadata, dirs, class_names)

    excluded = cleaned[cleaned["status"] == "excluded"]
    summary = {
        "seed": seed,
        "source_files": int(len(cleaned)),
        "included": int(len(included)),
        "excluded": int(len(excluded)),
        "exclusion_reasons": excluded["exclusion_reason"].str.split(":").str[0].value_counts().to_dict(),
        "flagged_not_excluded": int((included["flags"] != "").sum()),
        "images_per_split_and_class": {
            split: metadata[metadata["split"] == split]["class_name"].value_counts().reindex(class_names, fill_value=0)
            .astype(int).to_dict()
            for split in SPLITS
        },
        "patients_per_split_and_class": {
            split: split_df[split_df["split"] == split].groupby("folder_label")["patient_id"].nunique()
            .reindex(class_names, fill_value=0).astype(int).to_dict()
            for split in SPLITS
        },
        "split_fraction_of_images": metadata["split"].value_counts(normalize=True).round(4).reindex(SPLITS).to_dict(),
        "verification_problems": problems,
    }
    (dirs.metadata / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if problems:
        logger.error("Verification FAILED with %d problem(s)", len(problems))
        return 1
    logger.info("Verification passed - dataset ready in %s", ", ".join(str(p) for p in dirs.splits.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
