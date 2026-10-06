"""Analyse the raw source dataset: statistics, quality checks and figures.

Usage (project root, venv active):
    python scripts/analyze_dataset.py

Writes to reports/data_analysis/:
    dataset_statistics.json / .txt   — all numbers, computed from the files
    file_audit.csv                   — per-file audit (pseudonymised, no patient names)
    exact_duplicates.csv, near_duplicates.csv
    class_distribution.png, images_per_patient.png, sequence_by_class.png,
    image_dimensions.png, sample_images.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import visualization as viz  # noqa: E402
from src.data.pipeline import scan_source  # noqa: E402
from src.data.statistics import compute_statistics, format_statistics  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger("analyze_dataset")

# Columns safe to export — patient names (patient_key, source paths/filenames) are excluded.
AUDIT_EXPORT_COLUMNS = [
    "pseudo_name", "folder_label", "filename_label", "patient_id", "sequence", "slice_number", "slice_variant",
    "extension", "parse_ok", "file_exists", "file_bytes", "readable", "read_error", "image_format", "image_mode",
    "image_width", "image_height", "is_grayscale_content", "file_sha256", "pixel_sha256", "phash",
]


def main() -> int:
    config = load_config()
    class_names = list(config["data"]["class_names"])
    out_dir = resolve_path(config["paths"]["reports"]) / "data_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    scan = scan_source(config)
    stats = compute_statistics(scan.records, scan.exact_duplicates, scan.near_duplicates, scan.class_folders, class_names)

    (out_dir / "dataset_statistics.json").write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    report = format_statistics(stats)
    (out_dir / "dataset_statistics.txt").write_text(report, encoding="utf-8")
    scan.records[AUDIT_EXPORT_COLUMNS].to_csv(out_dir / "file_audit.csv", index=False)
    scan.exact_duplicates[AUDIT_EXPORT_COLUMNS + ["duplicate_group"]].to_csv(out_dir / "exact_duplicates.csv", index=False)
    scan.near_duplicates.to_csv(out_dir / "near_duplicates.csv", index=False)

    readable = scan.records[scan.records["readable"] & scan.records["parse_ok"]]
    viz.plot_class_distribution(scan.records, class_names, out_dir / "class_distribution.png",
                                "Images per class (raw source)")
    viz.plot_images_per_patient(readable, class_names, out_dir / "images_per_patient.png")
    viz.plot_sequence_by_class(readable, class_names, out_dir / "sequence_by_class.png")
    viz.plot_dimension_distribution(scan.records, out_dir / "image_dimensions.png")
    viz.plot_sample_grid(readable, class_names, out_dir / "sample_images.png", per_class=6,
                         seed=int(config["project"]["seed"]))

    print(report)
    logger.info("Analysis written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
