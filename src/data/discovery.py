"""Discover images in the source dataset and parse labels / patient information.

The source folder layout is ``<source_dir>/<class_name>/<file>``. Each filename
encodes the patient, MRI sequence and slice (see ``data.filename_pattern`` in
``config.yaml``). Patient names are replaced by stable pseudonymous IDs so that
no personal name leaves ``dataset/raw``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

RECORD_COLUMNS = [
    "source_path",
    "source_filename",
    "folder_label",
    "filename_label",
    "patient_key",
    "sequence",
    "slice_number",
    "slice_variant",
    "extension",
    "parse_ok",
]


def normalise_patient_key(raw_name: str) -> str:
    """Normalise a patient name so spelling variants of spacing/case collapse together."""
    return re.sub(r"[^a-z0-9]", "", raw_name.lower())


def normalise_sequence(raw_sequence: str) -> str:
    """Map filename sequence tokens to canonical names (e.g. 'T2 Flair' -> 'FLAIR')."""
    token = raw_sequence.strip().upper()
    return "FLAIR" if "FLAIR" in token else token


def discover_images(source_dir: Path, filename_pattern: str) -> tuple[pd.DataFrame, list[str]]:
    """Scan ``source_dir/<class>/`` and return one record per file.

    Args:
        source_dir: Directory whose immediate sub-folders are class labels.
        filename_pattern: Regex with named groups ``patient``, ``sequence``,
            ``slice``, ``label`` (``variant`` optional).

    Returns:
        (records DataFrame, list of class folder names found on disk)
    """
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset source directory not found: {source_dir}\n"
            "Run `python scripts/download_dataset.py` first."
        )

    pattern = re.compile(filename_pattern, re.IGNORECASE)
    class_folders = sorted(p.name for p in source_dir.iterdir() if p.is_dir())
    rows: list[dict] = []

    for class_name in class_folders:
        for file_path in sorted((source_dir / class_name).iterdir()):
            if not file_path.is_file():
                continue
            match = pattern.match(file_path.name)
            rows.append(
                {
                    "source_path": file_path,
                    "source_filename": file_path.name,
                    "folder_label": class_name,
                    "filename_label": match["label"] if match else None,
                    "patient_key": normalise_patient_key(match["patient"]) if match else None,
                    "sequence": normalise_sequence(match["sequence"]) if match else None,
                    "slice_number": int(match["slice"]) if match else None,
                    "slice_variant": int(match["variant"]) if match and match["variant"] else 0,
                    "extension": file_path.suffix.lower(),
                    "parse_ok": match is not None,
                }
            )

    records = pd.DataFrame(rows, columns=RECORD_COLUMNS)
    logger.info("Discovered %d files in %d class folders under %s", len(records), len(class_folders), source_dir)
    return records, class_folders


def assign_patient_ids(records: pd.DataFrame) -> pd.DataFrame:
    """Add a pseudonymous ``patient_id`` column (e.g. ``HAE-P01``).

    IDs are deterministic: patients are numbered in sorted order of their
    normalised key within each class, so re-running gives identical IDs.
    """
    records = records.copy()
    records["patient_id"] = None
    for label, group in records[records["parse_ok"]].groupby("folder_label"):
        prefix = str(label)[:3].upper()
        for number, key in enumerate(sorted(group["patient_key"].unique()), start=1):
            mask = (records["folder_label"] == label) & (records["patient_key"] == key)
            records.loc[mask, "patient_id"] = f"{prefix}-P{number:02d}"
    records["pseudo_name"] = [_pseudonymous_filename(row, idx) for idx, row in records.iterrows()]
    if records["pseudo_name"].duplicated().any():
        clashes = records.loc[records["pseudo_name"].duplicated(keep=False), "pseudo_name"].tolist()
        raise ValueError(f"Pseudonymous filenames are not unique: {clashes[:5]}")
    return records


def _pseudonymous_filename(row: pd.Series, index: int) -> str:
    """Name-free output filename, e.g. ``HAE-P01_DWI_s004.png`` or ``NOR-P01_SWI_s034_v1.png``."""
    if not row["parse_ok"] or row["patient_id"] is None:
        return f"unparsed_{index:05d}{row['extension']}"
    variant = f"_v{int(row['slice_variant'])}" if row["slice_variant"] else ""
    return f"{row['patient_id']}_{row['sequence']}_s{int(row['slice_number']):03d}{variant}{row['extension']}"
