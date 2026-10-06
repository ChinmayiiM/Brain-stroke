"""Data-leakage checks run before any test-set metric is computed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import PROJECT_ROOT


def check_leakage(metadata: pd.DataFrame, selection_file: Path, near_duplicates_file: Path | None) -> dict[str, Any]:
    """Verify split isolation and that model selection happened without the test set.

    Returns a report; ``passed`` is False if any blocking check fails.
    """
    test = metadata[metadata["split"] == "test"]
    fit = metadata[metadata["split"].isin(["train", "validation"])]
    checks: dict[str, Any] = {}

    checks["test_paths_in_train_or_validation"] = int(test["image_path"].isin(fit["image_path"]).sum())
    checks["test_hashes_in_train_or_validation"] = int(test["file_sha256"].isin(fit["file_sha256"]).sum())
    checks["test_patients_in_train_or_validation"] = sorted(set(test["patient_id"]) & set(fit["patient_id"]))
    checks["test_paths_located_in_test_folder"] = bool(
        test["image_path"].str.startswith("dataset/test/").all()
    )

    selection = json.loads(selection_file.read_text(encoding="utf-8")) if selection_file.is_file() else None
    checks["model_selection_recorded_before_test"] = bool(selection) and selection.get("test_set_used") is False

    # Near-duplicates (pHash) spanning test and train/validation — reported, not blocking.
    cross_split_near = None
    if near_duplicates_file and near_duplicates_file.is_file():
        pairs = pd.read_csv(near_duplicates_file)
        if not pairs.empty:
            name_to_split = dict(zip(metadata["image_path"].map(lambda p: Path(p).name), metadata["split"]))
            split_a = pairs["pseudo_name_a"].map(name_to_split)
            split_b = pairs["pseudo_name_b"].map(name_to_split)
            crosses = ((split_a == "test") ^ (split_b == "test")) & split_a.notna() & split_b.notna()
            cross_split_near = pairs.loc[
                crosses, ["pseudo_name_a", "pseudo_name_b", "hamming", "same_patient"]
            ].to_dict(orient="records")
    checks["near_duplicate_pairs_test_vs_fit"] = cross_split_near

    checks["training_code_reads_test_split"] = _training_code_reads_test()
    checks["passed"] = bool(
        checks["test_paths_in_train_or_validation"] == 0
        and checks["test_hashes_in_train_or_validation"] == 0
        and not checks["test_patients_in_train_or_validation"]
        and checks["test_paths_located_in_test_folder"]
        and checks["model_selection_recorded_before_test"]
        and not checks["training_code_reads_test_split"]
    )
    return checks


def _training_code_reads_test() -> bool:
    """Static check: the training entry point must never request the test split."""
    source = (PROJECT_ROOT / "src" / "training" / "train.py").read_text(encoding="utf-8")
    return '"test"' in source or "'test'" in source
