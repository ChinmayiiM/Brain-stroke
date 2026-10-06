"""Materialise the cleaned dataset, the split folders and the metadata — then verify them."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.image_audit import file_sha256
from src.utils.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

SPLIT_DIR_KEYS = {"train": "train_data", "validation": "validation_data", "test": "test_data"}

METADATA_COLUMNS = [
    "image_path",
    "processed_path",
    "label",
    "class_name",
    "split",
    "patient_id",
    "sequence",
    "slice_number",
    "slice_variant",
    "image_width",
    "image_height",
    "format",
    "mode",
    "file_sha256",
]


@dataclass(frozen=True)
class OutputDirs:
    processed: Path
    splits: dict[str, Path]
    metadata: Path

    @classmethod
    def from_config(cls, config: dict, resolve) -> "OutputDirs":
        paths = config["paths"]
        return cls(
            processed=resolve(paths["processed_data"]),
            splits={split: resolve(paths[key]) for split, key in SPLIT_DIR_KEYS.items()},
            metadata=resolve(paths["metadata"]),
        )

    def generated_dirs(self) -> list[Path]:
        return [self.processed, *self.splits.values()]


def _relative(path: Path) -> str:
    """Project-relative POSIX path for portable metadata."""
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def reset_generated_dirs(dirs: OutputDirs) -> None:
    """Delete and recreate generated folders so re-runs never leave stale files.

    Refuses to touch anything outside ``<project>/dataset`` as a safety guard.
    """
    dataset_root = (PROJECT_ROOT / "dataset").resolve()
    for directory in dirs.generated_dirs():
        resolved = directory.resolve()
        if dataset_root not in resolved.parents:
            raise ValueError(f"Refusing to reset a directory outside {dataset_root}: {resolved}")
        if resolved.exists():
            shutil.rmtree(resolved)
        resolved.mkdir(parents=True)
    dirs.metadata.mkdir(parents=True, exist_ok=True)


def materialise(df: pd.DataFrame, dirs: OutputDirs, class_names: list[str]) -> pd.DataFrame:
    """Copy each included image (bytes unchanged) to processed/ and to its split folder.

    Returns the metadata DataFrame.
    """
    label_index = {name: i for i, name in enumerate(class_names)}
    rows = []
    for _, item in df.iterrows():
        processed_path = dirs.processed / item["folder_label"] / item["pseudo_name"]
        split_path = dirs.splits[item["split"]] / item["folder_label"] / item["pseudo_name"]
        for target in (processed_path, split_path):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["source_path"], target)
        rows.append(
            {
                "image_path": _relative(split_path),
                "processed_path": _relative(processed_path),
                "label": label_index[item["folder_label"]],
                "class_name": item["folder_label"],
                "split": item["split"],
                "patient_id": item["patient_id"],
                "sequence": item["sequence"],
                "slice_number": int(item["slice_number"]),
                "slice_variant": int(item["slice_variant"]),
                "image_width": int(item["image_width"]),
                "image_height": int(item["image_height"]),
                "format": item["image_format"],
                "mode": item["image_mode"],
                "file_sha256": item["file_sha256"],
            }
        )
    metadata = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    split_order = {s: i for i, s in enumerate(SPLIT_DIR_KEYS)}
    metadata = metadata.sort_values(
        ["split", "class_name", "image_path"], key=lambda col: col.map(split_order) if col.name == "split" else col
    ).reset_index(drop=True)
    logger.info("Materialised %d images into processed/ and split folders", len(metadata))
    return metadata


def verify_prepared_dataset(metadata: pd.DataFrame, dirs: OutputDirs, class_names: list[str]) -> list[str]:
    """Return a list of problems (empty list == dataset is consistent)."""
    problems: list[str] = []

    # 1. Every file exists, opens, is uncorrupted and matches its recorded hash.
    for _, row in metadata.iterrows():
        for column in ("image_path", "processed_path"):
            path = PROJECT_ROOT / row[column]
            if not path.is_file():
                problems.append(f"missing: {row[column]}")
                continue
            if file_sha256(path) != row["file_sha256"]:
                problems.append(f"hash mismatch: {row[column]}")
        try:
            with Image.open(PROJECT_ROOT / row["image_path"]) as img:
                img.verify()
        except Exception as exc:  # noqa: BLE001
            problems.append(f"unreadable: {row['image_path']} ({exc})")

        # 2. Folder name == class label == label index.
        folder_class = Path(row["image_path"]).parent.name
        if folder_class != row["class_name"] or class_names[row["label"]] != row["class_name"]:
            problems.append(f"label mismatch: {row['image_path']}")
        split_folder = Path(row["image_path"]).parent.parent.name
        if dirs.splits[row["split"]].name != split_folder:
            problems.append(f"split folder mismatch: {row['image_path']}")

    # 3. No patient and no identical image in more than one split.
    leaking_patients = metadata.groupby("patient_id")["split"].nunique()
    for patient in leaking_patients[leaking_patients > 1].index:
        problems.append(f"patient leakage across splits: {patient}")
    leaking_hashes = metadata.groupby("file_sha256")["split"].nunique()
    if (leaking_hashes > 1).any():
        problems.append(f"{int((leaking_hashes > 1).sum())} identical images appear in multiple splits")
    if metadata["file_sha256"].duplicated().any():
        problems.append("duplicate images remain in the prepared dataset")

    # 4. Files on disk == metadata (no stray or missing files), and only real class folders exist.
    for split, directory in dirs.splits.items():
        on_disk = {p.resolve() for p in directory.rglob("*") if p.is_file()}
        expected = {(PROJECT_ROOT / p).resolve() for p in metadata.loc[metadata["split"] == split, "image_path"]}
        if on_disk != expected:
            problems.append(f"{split}: {len(on_disk - expected)} stray / {len(expected - on_disk)} missing files")
        unknown = [d.name for d in directory.iterdir() if d.is_dir() and d.name not in class_names]
        if unknown:
            problems.append(f"{split}: unexpected class folders {unknown}")

    return problems
