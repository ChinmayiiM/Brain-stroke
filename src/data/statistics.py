"""Dataset statistics computed from audited records (never hard-coded)."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _dimension_summary(values: pd.Series) -> dict[str, float]:
    return {"min": int(values.min()), "max": int(values.max()), "mean": round(float(values.mean()), 1)}


def compute_statistics(
    records: pd.DataFrame,
    exact_duplicates: pd.DataFrame,
    near_duplicates: pd.DataFrame,
    class_folders: list[str],
    expected_classes: list[str],
) -> dict[str, Any]:
    """Summarise the raw (pre-cleaning) dataset."""
    readable = records[records["readable"]]
    filename_counts = records["source_filename"].value_counts()
    cross_patient = near_duplicates[~near_duplicates["same_patient"]] if not near_duplicates.empty else near_duplicates

    return {
        "total_files": int(len(records)),
        "class_folders_found": class_folders,
        "missing_class_folders": [c for c in expected_classes if c not in class_folders],
        "unexpected_class_folders": [c for c in class_folders if c not in expected_classes],
        "images_per_class": records["folder_label"].value_counts().sort_index().to_dict(),
        "class_distribution_pct": (records["folder_label"].value_counts(normalize=True).sort_index() * 100)
        .round(2)
        .to_dict(),
        "patients_per_class": records.groupby("folder_label")["patient_id"].nunique().to_dict(),
        "images_per_patient": records.groupby(["folder_label", "patient_id"]).size()
        .rename("images")
        .reset_index()
        .to_dict(orient="records"),
        "images_per_sequence_and_class": pd.crosstab(records["sequence"], records["folder_label"]).to_dict(),
        "image_width": _dimension_summary(readable["image_width"]) if len(readable) else None,
        "image_height": _dimension_summary(readable["image_height"]) if len(readable) else None,
        "dimension_counts": readable.groupby(["image_width", "image_height"]).size()
        .rename("count")
        .reset_index()
        .to_dict(orient="records"),
        "file_extensions": records["extension"].value_counts().to_dict(),
        "decoded_formats": readable["image_format"].value_counts().to_dict(),
        "colour_modes": readable["image_mode"].value_counts().to_dict(),
        "grayscale_content_images": int(readable["is_grayscale_content"].sum()),
        "missing_files": int((~records["file_exists"]).sum()),
        "corrupted_unreadable": int((records["file_exists"] & ~records["readable"]).sum()),
        "unparseable_filenames": int((~records["parse_ok"]).sum()),
        "label_mismatches": int(
            (records["parse_ok"] & (records["filename_label"] != records["folder_label"])).sum()
        ),
        "duplicate_filenames": int((filename_counts > 1).sum()),
        "exact_duplicate_groups": int(exact_duplicates["duplicate_group"].nunique()) if len(exact_duplicates) else 0,
        "exact_duplicate_files": int(len(exact_duplicates)),
        "redundant_exact_copies": int(len(exact_duplicates) - exact_duplicates["duplicate_group"].nunique())
        if len(exact_duplicates)
        else 0,
        "near_duplicate_pairs": int(len(near_duplicates)),
        "near_duplicate_pairs_cross_patient": int(len(cross_patient)),
    }


def format_statistics(stats: dict[str, Any]) -> str:
    """Human-readable text report."""
    lines = ["DATASET ANALYSIS (raw source, before cleaning)", "=" * 48, f"Total images: {stats['total_files']}", ""]
    lines.append("Images per class:")
    for label, count in stats["images_per_class"].items():
        lines.append(
            f"  {label:<14} {count:>5}  ({stats['class_distribution_pct'][label]:5.1f} %)  "
            f"patients: {stats['patients_per_class'].get(label, 0)}"
        )
    lines += ["", "Images per patient:"]
    for row in stats["images_per_patient"]:
        lines.append(f"  {row['folder_label']:<14} {row['patient_id']}  {row['images']:>4}")
    lines += ["", "Images per MRI sequence (rows) and class (columns):"]
    seq_table = pd.DataFrame(stats["images_per_sequence_and_class"]).fillna(0).astype(int)
    lines += ["  " + line for line in seq_table.to_string().splitlines()]
    w, h = stats["image_width"], stats["image_height"]
    if w and h:
        lines += [
            "",
            "Image dimensions:",
            f"  Width   min {w['min']}  max {w['max']}  mean {w['mean']}",
            f"  Height  min {h['min']}  max {h['max']}  mean {h['mean']}",
        ]
        for row in stats["dimension_counts"]:
            lines.append(f"  {row['image_width']}x{row['image_height']}: {row['count']}")
    lines += [
        "",
        f"File extensions:       {stats['file_extensions']}",
        f"Decoded formats:       {stats['decoded_formats']}",
        f"Colour modes:          {stats['colour_modes']}",
        f"Grayscale content:     {stats['grayscale_content_images']} images (R=G=B)",
        "",
        f"Missing class folders: {stats['missing_class_folders'] or 'none'}",
        f"Unexpected folders:    {stats['unexpected_class_folders'] or 'none'}",
        f"Missing files:         {stats['missing_files']}",
        f"Corrupted/unreadable:  {stats['corrupted_unreadable']}",
        f"Unparseable filenames: {stats['unparseable_filenames']}",
        f"Label mismatches:      {stats['label_mismatches']}",
        f"Duplicate filenames:   {stats['duplicate_filenames']}",
        f"Exact duplicates:      {stats['exact_duplicate_files']} files in {stats['exact_duplicate_groups']} groups "
        f"({stats['redundant_exact_copies']} redundant copies)",
        f"Near-duplicate pairs:  {stats['near_duplicate_pairs']} "
        f"({stats['near_duplicate_pairs_cross_patient']} across different patients)",
    ]
    return "\n".join(lines)
