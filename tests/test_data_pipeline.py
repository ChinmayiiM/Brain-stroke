"""Unit tests for discovery, auditing, duplicate detection, cleaning and splitting.

Test fixtures are tiny synthetic arrays written to pytest's tmp_path — they
exercise code paths only and are never part of the dataset.
"""

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data.cleaning import CleaningRules, apply_cleaning_rules, check_class_folders
from src.data.discovery import assign_patient_ids, discover_images
from src.data.duplicates import find_exact_duplicate_groups, find_near_duplicate_pairs
from src.data.image_audit import audit_images
from src.data.splitting import patient_level_split
from src.utils.config import load_config

PATTERN = load_config()["data"]["filename_pattern"]
RULES = CleaningRules(
    class_names=("A", "B"),
    allowed_extensions=(".png", ".jpg"),
    min_file_bytes=10,
    min_dimension=16,
    max_dimension=4096,
    max_aspect_ratio=2.0,
)


def _pattern_image(seed: int, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((size, size)) * 255).astype(np.uint8)


def _save(path, array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).convert("RGB").save(path)


@pytest.fixture()
def source_dir(tmp_path):
    root = tmp_path / "src"
    _save(root / "A" / "John Doe DWI-1.jpg_A_1.png", _pattern_image(1))
    _save(root / "A" / "John Doe SWI-2(1).jpg_A_2.png", _pattern_image(2))
    _save(root / "A" / "John Doe SWI-2(2).jpg_A_3.png", _pattern_image(2))  # exact duplicate
    _save(root / "A" / "Jane Roe T2 Flair-3.jpg_B_4.png", _pattern_image(3))  # label mismatch
    _save(root / "B" / "Max Mustermann GRE-5.jpg_B_5.png", _pattern_image(4))
    _save(root / "B" / "random_name.png", _pattern_image(5))  # unparseable
    (root / "B" / "Erika Muster DWI-6.jpg_B_6.png").write_bytes(b"not an image at all")  # corrupted
    return root


def _scan(source_dir):
    records, folders = discover_images(source_dir, PATTERN)
    return audit_images(assign_patient_ids(records)), folders


def test_discovery_parses_patient_sequence_and_variant(source_dir):
    records, folders = discover_images(source_dir, PATTERN)
    assert folders == ["A", "B"]
    row = records[records["source_filename"] == "John Doe SWI-2(1).jpg_A_2.png"].iloc[0]
    assert (row["patient_key"], row["sequence"], row["slice_number"], row["slice_variant"]) == ("johndoe", "SWI", 2, 1)
    flair = records[records["source_filename"].str.startswith("Jane")].iloc[0]
    assert flair["sequence"] == "FLAIR"
    assert not records[records["source_filename"] == "random_name.png"].iloc[0]["parse_ok"]


def test_pseudonymous_ids_are_deterministic_and_name_free(source_dir):
    records, _ = discover_images(source_dir, PATTERN)
    first = assign_patient_ids(records)
    second = assign_patient_ids(records.sample(frac=1.0, random_state=0)).loc[first.index]
    assert first["patient_id"].equals(second["patient_id"])
    names = " ".join(first["pseudo_name"])
    assert "john" not in names.lower() and "doe" not in names.lower()


def test_audit_detects_corruption(source_dir):
    records, _ = _scan(source_dir)
    bad = records[records["source_filename"].str.startswith("Erika")].iloc[0]
    assert not bad["readable"] and bad["read_error"]
    assert records[records["readable"]]["is_grayscale_content"].all()


def test_exact_and_near_duplicates(source_dir):
    records, _ = _scan(source_dir)
    exact = find_exact_duplicate_groups(records)
    assert exact["duplicate_group"].nunique() == 1 and len(exact) == 2
    near = find_near_duplicate_pairs(records, max_distance=0)
    assert (near["hamming"] == 0).any()


def test_cleaning_rules(source_dir):
    records, folders = _scan(source_dir)
    assert check_class_folders(folders, ("A", "B", "C")) == (["C"], [])
    cleaned = apply_cleaning_rules(records, RULES, find_exact_duplicate_groups(records)).set_index("source_filename")
    reasons = cleaned["exclusion_reason"]
    assert reasons["John Doe SWI-2(2).jpg_A_3.png"].startswith("exact_duplicate_of")
    assert pd.isna(reasons["John Doe SWI-2(1).jpg_A_2.png"])  # first copy kept
    assert reasons["Jane Roe T2 Flair-3.jpg_B_4.png"].startswith("label_mismatch")
    assert reasons["random_name.png"].startswith("unparseable_filename")
    assert reasons["Erika Muster DWI-6.jpg_B_6.png"].startswith(("corrupted_unreadable", "empty_or_truncated"))
    assert (cleaned.loc[reasons.notna(), "status"] == "excluded").all()


def test_conflicting_label_duplicates_are_all_excluded(tmp_path):
    root = tmp_path / "src"
    _save(root / "A" / "P One DWI-1.jpg_A_1.png", _pattern_image(9))
    _save(root / "B" / "P Two DWI-1.jpg_B_1.png", _pattern_image(9))
    records, _ = _scan(root)
    cleaned = apply_cleaning_rules(records, RULES, find_exact_duplicate_groups(records))
    assert cleaned["exclusion_reason"].str.startswith("conflicting_label_duplicate").all()


def _split_frame() -> pd.DataFrame:
    rows = []
    for label, sizes in {"A": [30, 20, 10, 5, 8], "B": [3, 7, 11, 5]}.items():
        for p, n in enumerate(sizes):
            rows += [{"folder_label": label, "patient_id": f"{label}-P{p}"}] * n
    return pd.DataFrame(rows)


def test_patient_level_split_has_no_leakage_and_full_coverage():
    df = patient_level_split(_split_frame(), {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42)
    assert (df.groupby("patient_id")["split"].nunique() == 1).all()
    coverage = df.groupby(["folder_label", "split"])["patient_id"].nunique().unstack(fill_value=0)
    assert (coverage >= 1).all().all()


def test_patient_level_split_is_deterministic():
    a = patient_level_split(_split_frame(), {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    b = patient_level_split(_split_frame(), {"train": 0.7, "val": 0.15, "test": 0.15}, seed=1)
    assert a["split"].equals(b["split"])


def test_split_requires_three_patients_per_class():
    df = pd.DataFrame({"folder_label": ["A"] * 4, "patient_id": ["A-P0", "A-P0", "A-P1", "A-P1"]})
    with pytest.raises(ValueError):
        patient_level_split(df, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=0)
