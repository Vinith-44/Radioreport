"""Which studies each Step 2 artifact covers, and where it lives.

  train : every TRAIN study with an image  -> the retrieval index + probe training (CLAUDE.md §3: train only)
  val   : every VAL study with an image    -> probe model selection / thresholds
  test  : ONLY the fixed Step 0 test subset (step0_eval_ids.json) -> queries at inference time
All arrays under <run>/embeddings and <run>/labels are row-aligned with <run>/embeddings/<split>_ids.parquet.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.text_clean import format_report

COLS = ["study_id", "patient_id", "dataset", "split", "image_path", "findings", "impression"]


def split_rows(ddir: Path, split: str, suffix: str = "") -> pd.DataFrame:
    df = pd.read_parquet(ddir / f"{split}{suffix}.parquet", columns=COLS)
    if split == "test":
        ids = json.loads((ddir / f"step0_eval_ids{suffix}.json").read_text(encoding="utf-8"))
        want = {str(i) for v in ids.values() for i in v}
        df = df[df["study_id"].astype(str).isin(want)]
        assert len(df) == len(want), f"{len(want) - len(df)} eval ids missing from test{suffix}.parquet"
    assert set(df["split"]) <= {split}, f"{split}{suffix}.parquet holds rows of {set(df['split'])}"
    df = df[df["image_path"].notna()]
    return df.sort_values(["dataset", "study_id"]).reset_index(drop=True)


def report_text(row: dict, mode: str = "findings+impression") -> str:
    if mode == "findings":
        return (row.get("findings") or "").strip()
    if mode == "findings+impression":
        return format_report(row.get("findings") or "", row.get("impression") or "")
    raise ValueError(mode)


def ids_sha(df: pd.DataFrame) -> str:
    return hashlib.sha256("\n".join(df["study_id"].astype(str)).encode()).hexdigest()


def load_split(run: Path, split: str, labels: bool = False):
    """(ids frame, embeddings[, CheXbert labels]) for one split, with alignment checked."""
    ids = pd.read_parquet(run / "embeddings" / f"{split}_ids.parquet")
    emb = np.load(run / "embeddings" / f"{split}.npy")
    assert len(ids) == len(emb), f"{split}: {len(ids)} ids vs {len(emb)} embeddings"
    if not labels:
        return ids, emb
    lab_p = run / "labels" / f"{split}.npy"
    meta = json.loads((run / "labels" / f"{split}.json").read_text(encoding="utf-8"))
    if meta["ids_sha256"] != ids_sha(ids):
        raise AssertionError(f"{split}: labels were computed for different studies than the embeddings; relabel")
    lab = np.load(lab_p)
    return ids, emb, lab
