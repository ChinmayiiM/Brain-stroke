"""Exact and near-duplicate image detection."""

from __future__ import annotations

import numpy as np
import pandas as pd


def find_exact_duplicate_groups(df: pd.DataFrame, hash_column: str = "pixel_sha256") -> pd.DataFrame:
    """Return rows whose hash occurs more than once, with a ``duplicate_group`` id.

    ``pixel_sha256`` (decoded pixels) catches identical images even when the
    files differ in encoding/metadata; ``file_sha256`` catches byte-identical files.
    """
    hashed = df[df[hash_column].notna()]
    dup_mask = hashed.duplicated(hash_column, keep=False)
    dups = hashed[dup_mask].copy()
    dups["duplicate_group"] = dups.groupby(hash_column).ngroup()
    return dups.sort_values(["duplicate_group", "source_filename"])


def _popcount64(values: np.ndarray) -> np.ndarray:
    """Vectorised bit count for an array of uint64."""
    counts = np.zeros(values.shape, dtype=np.uint8)
    v = values.copy()
    for _ in range(64):
        counts += (v & np.uint64(1)).astype(np.uint8)
        v >>= np.uint64(1)
    return counts


def find_near_duplicate_pairs(df: pd.DataFrame, max_distance: int, hash_column: str = "phash") -> pd.DataFrame:
    """All image pairs whose perceptual-hash Hamming distance is ``<= max_distance``.

    Exact pixel duplicates are included (distance 0). O(n²) in memory row by
    row, which is fine for datasets of a few thousand images.
    """
    valid = df[df[hash_column].notna()]
    hashes = valid[hash_column].astype("uint64").to_numpy()
    index = valid.index.to_numpy()
    pairs: list[tuple] = []
    for i in range(len(hashes) - 1):
        distances = _popcount64(np.bitwise_xor(hashes[i], hashes[i + 1 :]))
        for offset in np.nonzero(distances <= max_distance)[0]:
            pairs.append((index[i], index[i + 1 + offset], int(distances[offset])))

    pair_df = pd.DataFrame(pairs, columns=["index_a", "index_b", "hamming"])
    if pair_df.empty:
        return pair_df
    cols = ["pseudo_name", "folder_label", "patient_id", "sequence"]
    a = valid.loc[pair_df["index_a"], cols].reset_index(drop=True).add_suffix("_a")
    b = valid.loc[pair_df["index_b"], cols].reset_index(drop=True).add_suffix("_b")
    pair_df = pd.concat([pair_df, a, b], axis=1)
    pair_df["same_patient"] = pair_df["patient_id_a"] == pair_df["patient_id_b"]
    pair_df["same_label"] = pair_df["folder_label_a"] == pair_df["folder_label_b"]
    return pair_df.sort_values(["hamming", "index_a"]).reset_index(drop=True)
