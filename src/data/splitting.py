"""Leakage-safe, class-stratified, patient-level train/validation/test splitting.

Each patient belongs to exactly one class, so stratification is done per class:
the class's patients are partitioned into train/validation/test so that

1. every split receives at least one patient of every class (when the class
   has >= 3 patients), and
2. the *image* fractions are as close as possible to the configured ratios.

With only a handful of patients per class, the exact optimum is found by
exhaustive search over assignments; ties are broken by a seeded random order,
so the result is fully deterministic for a given seed.
"""

from __future__ import annotations

import itertools
import random

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

SPLITS = ("train", "validation", "test")
MAX_EXHAUSTIVE_PATIENTS = 12  # 3**12 ≈ 531k assignments — still fast


def _assignment_cost(counts: list[int], assignment: tuple[int, ...], targets: tuple[float, ...]) -> float:
    total = sum(counts)
    per_split = [0, 0, 0]
    for count, split_idx in zip(counts, assignment):
        per_split[split_idx] += count
    return sum(abs(per_split[i] / total - targets[i]) for i in range(3))


def _split_one_class(patient_counts: pd.Series, targets: tuple[float, ...], rng: random.Random) -> dict[str, str]:
    """Return {patient_id: split} for the patients of a single class."""
    patients = sorted(patient_counts.index)
    rng.shuffle(patients)  # seeded tie-breaking order
    counts = [int(patient_counts[p]) for p in patients]

    if len(patients) < 3:
        raise ValueError(
            f"Class has only {len(patients)} patient(s); at least 3 are required for a "
            "patient-level train/validation/test split."
        )
    if len(patients) > MAX_EXHAUSTIVE_PATIENTS:
        raise NotImplementedError(
            f"{len(patients)} patients in one class exceeds the exhaustive-search limit; "
            "use sklearn StratifiedGroupKFold-based splitting for larger datasets."
        )

    best, best_cost = None, float("inf")
    for assignment in itertools.product(range(3), repeat=len(patients)):
        if len(set(assignment)) < 3:  # every split needs >= 1 patient of this class
            continue
        cost = _assignment_cost(counts, assignment, targets)
        if cost < best_cost - 1e-12:
            best, best_cost = assignment, cost
    return {patient: SPLITS[idx] for patient, idx in zip(patients, best)}


def patient_level_split(
    df: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
    label_column: str = "folder_label",
    group_column: str = "patient_id",
) -> pd.DataFrame:
    """Add a ``split`` column to ``df`` (included records only are expected)."""
    targets = (float(ratios["train"]), float(ratios["val"]), float(ratios["test"]))
    if abs(sum(targets) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1, got {targets}")

    rng = random.Random(seed)
    patient_to_split: dict[str, str] = {}
    for label in sorted(df[label_column].unique()):
        counts = df[df[label_column] == label].groupby(group_column).size()
        patient_to_split.update(_split_one_class(counts, targets, rng))

    out = df.copy()
    out["split"] = out[group_column].map(patient_to_split)
    for label, group in out.groupby(label_column):
        fractions = group["split"].value_counts(normalize=True).reindex(SPLITS, fill_value=0)
        logger.info("%-14s %s", label, "  ".join(f"{s}={fractions[s]:.1%}" for s in SPLITS))
    return out
