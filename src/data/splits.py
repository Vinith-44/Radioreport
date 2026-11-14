"""Leakage assertions, parquet writing, and the fixed Step 0 evaluation subset."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def assert_no_leakage(df: pd.DataFrame) -> dict:
    """Hard checks (CLAUDE.md §3). Raises AssertionError on any violation; returns a summary."""
    assert set(df["split"].unique()) <= set(SPLITS), f"unexpected splits {set(df['split'].unique())}"
    key = df["dataset"] + "|" + df["study_id"].astype(str)
    dup = key[key.duplicated()]
    assert dup.empty, f"duplicate (dataset, study_id): {dup.head().tolist()}"

    pats = {s: set(df.loc[df["split"] == s, "patient_id"]) for s in SPLITS}
    for a, b in itertools.combinations(SPLITS, 2):
        overlap = pats[a] & pats[b]
        assert not overlap, f"PATIENT LEAKAGE {a}/{b}: {len(overlap)} patients, e.g. {sorted(overlap)[:5]}"

    # A study's prior must belong to the same patient, hence the same split.
    by_key = df.set_index(key)
    pri = df[df["prior_study_id"].notna()]
    pkey = pri["dataset"] + "|" + pri["prior_study_id"].astype(str)
    known = pkey.isin(by_key.index)
    if known.any():
        pr_split = by_key.loc[pkey[known], "split"].values
        assert (pr_split == pri.loc[known, "split"].values).all(), "prior study in a different split"
    return {s: {"studies": int((df["split"] == s).sum()), "patients": len(pats[s])} for s in SPLITS}


def write_splits(df: pd.DataFrame, out_dir: Path, suffix: str = "") -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for s in SPLITS:
        p = out_dir / f"{s}{suffix}.parquet"
        df[df["split"] == s].reset_index(drop=True).to_parquet(p, index=False)
        paths[s] = p
    return paths


def select_eval_ids(test: pd.DataFrame, max_n: int | None, seed: int) -> dict[str, list[str]]:
    """Seeded per-dataset subset of the test split, fixed once so every system is scored on the same ids."""
    out = {}
    for ds, g in test.groupby("dataset"):
        ids = sorted(g["study_id"].astype(str))
        if max_n and len(ids) > max_n:
            rng = np.random.default_rng(seed)
            ids = sorted(rng.choice(ids, size=max_n, replace=False).tolist())
        out[ds] = ids
    return out


def save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
