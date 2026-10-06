"""Rule-based validation of audited records.

Nothing is deleted from ``dataset/raw``. Every record receives:

* ``status``           — ``included`` or ``excluded``
* ``exclusion_reason`` — why an excluded record was dropped (clear justification only)
* ``flags``            — non-blocking observations that are reported but do not exclude
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class CleaningRules:
    class_names: tuple[str, ...]
    allowed_extensions: tuple[str, ...]
    min_file_bytes: int
    min_dimension: int
    max_dimension: int
    max_aspect_ratio: float

    @classmethod
    def from_config(cls, config: dict) -> "CleaningRules":
        cleaning = config["cleaning"]
        return cls(
            class_names=tuple(config["data"]["class_names"]),
            allowed_extensions=tuple(e.lower() for e in config["data"]["allowed_extensions"]),
            min_file_bytes=int(cleaning["min_file_bytes"]),
            min_dimension=int(cleaning["min_dimension"]),
            max_dimension=int(cleaning["max_dimension"]),
            max_aspect_ratio=float(cleaning["max_aspect_ratio"]),
        )


def check_class_folders(found: list[str], expected: tuple[str, ...]) -> tuple[list[str], list[str]]:
    """Return (missing expected class folders, unexpected extra folders)."""
    missing = [c for c in expected if c not in found]
    extra = [c for c in found if c not in expected]
    return missing, extra


def _first_failing_rule(row: pd.Series, rules: CleaningRules) -> str | None:
    """Blocking rules in priority order; returns the reason or None."""
    if row["folder_label"] not in rules.class_names:
        return "invalid_label: class folder not in configured class_names"
    if row["extension"] not in rules.allowed_extensions:
        return f"unsupported_format: extension {row['extension']}"
    if not row["file_exists"]:
        return "missing_file"
    if row["file_bytes"] is not None and row["file_bytes"] < rules.min_file_bytes:
        return f"empty_or_truncated: {row['file_bytes']} bytes < {rules.min_file_bytes}"
    if not row["readable"]:
        return f"corrupted_unreadable: {row['read_error']}"
    if not row["parse_ok"]:
        return "unparseable_filename: no patient ID, cannot be split leakage-safely"
    if row["filename_label"] != row["folder_label"]:
        return f"label_mismatch: filename says {row['filename_label']}, folder says {row['folder_label']}"
    width, height = row["image_width"], row["image_height"]
    if min(width, height) < rules.min_dimension or max(width, height) > rules.max_dimension:
        return f"unusable_dimensions: {width}x{height}"
    return None


def _flags(row: pd.Series, rules: CleaningRules) -> list[str]:
    flags = []
    if row["readable"]:
        aspect = max(row["image_width"], row["image_height"]) / min(row["image_width"], row["image_height"])
        if aspect > rules.max_aspect_ratio:
            flags.append(f"unusual_aspect_ratio:{aspect:.2f}")
        if not row["is_grayscale_content"]:
            flags.append("colour_content")
    return flags


def apply_cleaning_rules(records: pd.DataFrame, rules: CleaningRules, exact_duplicates: pd.DataFrame) -> pd.DataFrame:
    """Assign status / exclusion_reason / flags to every record.

    Exact pixel duplicates: if all copies share a label, the first copy (sorted
    filename) is kept and the rest are excluded; if copies carry *different*
    labels, the label is ambiguous and all copies are excluded.
    """
    df = records.copy()
    df["exclusion_reason"] = [_first_failing_rule(row, rules) for _, row in df.iterrows()]
    df["flags"] = [";".join(_flags(row, rules)) for _, row in df.iterrows()]

    for _, group in exact_duplicates.groupby("duplicate_group"):
        group = group.sort_values("source_filename")
        still_valid = group[df.loc[group.index, "exclusion_reason"].isna()]
        if still_valid.empty:
            continue
        if still_valid["folder_label"].nunique() > 1:
            labels = "/".join(sorted(still_valid["folder_label"].unique()))
            df.loc[still_valid.index, "exclusion_reason"] = f"conflicting_label_duplicate: identical image labelled {labels}"
            continue
        keeper = still_valid.iloc[0]
        for idx in still_valid.index[1:]:
            df.loc[idx, "exclusion_reason"] = f"exact_duplicate_of: {keeper['pseudo_name']}"

    df["status"] = df["exclusion_reason"].isna().map({True: "included", False: "excluded"})
    logger.info(
        "Cleaning: %d included, %d excluded",
        int((df["status"] == "included").sum()),
        int((df["status"] == "excluded").sum()),
    )
    return df
