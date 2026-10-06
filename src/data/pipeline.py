"""High-level orchestration shared by the analysis and preparation scripts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.discovery import assign_patient_ids, discover_images
from src.data.duplicates import find_exact_duplicate_groups, find_near_duplicate_pairs
from src.data.image_audit import audit_images
from src.utils.config import resolve_path


@dataclass
class SourceScan:
    """Everything known about the raw source before any cleaning decision."""

    records: pd.DataFrame
    class_folders: list[str]
    exact_duplicates: pd.DataFrame
    near_duplicates: pd.DataFrame


def source_directory(config: dict) -> Path:
    dataset = config["dataset"]
    return resolve_path(config["paths"]["raw_data"]) / dataset["extract_dir"] / dataset["source_subdir"]


def scan_source(config: dict) -> SourceScan:
    """Discover → pseudonymise → audit every file → detect duplicates."""
    records, class_folders = discover_images(source_directory(config), config["data"]["filename_pattern"])
    records = assign_patient_ids(records)
    records = audit_images(records)
    exact = find_exact_duplicate_groups(records, "pixel_sha256")
    near = find_near_duplicate_pairs(records, int(config["cleaning"]["near_duplicate_hamming"]))
    return SourceScan(records, class_folders, exact, near)
