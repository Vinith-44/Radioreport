"""CheXphoto (photos of CheXpert chest X-rays) paired with the clean CheXpert originals, for Step 6.

Observed layout of the Kaggle copy gajahterbang111/chexphoto-split (2026-11-14; an UNOFFICIAL re-upload):
  <root>/CheXphoto-v1.0_split_*/CheXphoto-v1.0_split_*/train.csv
  <root>/.../train/natural/iphone/patient00205/study1/view1_frontal.jpg
The CSV "Path" is matched to files by its tail from "train/" or "valid/". The clean original of
train/natural/<device>/patientX/studyY/viewZ.jpg is <clean_root>/train/patientX/studyY/viewZ.jpg (CheXpert small).
Labels follow CheXpert (1 / 0 / -1 uncertain / blank). LEAKAGE GUARD: CheXphoto-train photographs CheXpert-TRAIN
patients, some of whom were used for Model B's template reports, so every patient seen in training is excluded.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from src.data.synthetic import CHEXPERT_FINDINGS

LABELS = ["No Finding"] + CHEXPERT_FINDINGS   # CheXpert CSV order (13 conditions + No Finding first)
_TAIL = re.compile(r"((?:train|valid)/.*)$")


def tail(p: str) -> str | None:
    m = _TAIL.search(str(p).replace("\\", "/"))
    return m.group(1) if m else None


def clean_tail(photo_tail: str) -> str:
    """train/natural/iphone/patientX/studyY/v.jpg -> train/patientX/studyY/v.jpg"""
    parts = photo_tail.split("/")
    i = next(k for k, s in enumerate(parts) if s.startswith("patient"))
    return "/".join([parts[0]] + parts[i:])


def load_pairs(root: Path, clean_root: Path, kinds: list[str], exclude_patients: set[str], n: int | None,
               seed: int) -> tuple[pd.DataFrame, int]:
    """-> (pairs, number of photos dropped because their patient was in a training set)."""
    root, clean_root = Path(root), Path(clean_root)
    files = {}
    for f in root.rglob("*.jpg"):
        if f.name.startswith("._"):
            continue
        t = tail(f.as_posix())
        if t:
            files[t] = str(f)
    frames = []
    for csv in root.rglob("*.csv"):
        df = pd.read_csv(csv)
        if "Path" in df.columns:
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"no CheXphoto CSV with a 'Path' column under {root}")
    df = pd.concat(frames, ignore_index=True)
    df["tail"] = df["Path"].map(tail)
    df = df[df["tail"].notna() & df["tail"].isin(files)]
    df = df[df["tail"].str.contains("frontal")]
    df = df[df["tail"].map(lambda t: any(f"/{k}/" in f"/{t}" for k in kinds))]
    df["patient_id"] = "chexpert:" + df["tail"].str.extract(r"(patient\d+)")[0]
    seen = df["patient_id"].isin(exclude_patients)
    n_excluded = int(seen.sum())
    df = df[~seen]
    df["clean_path"] = [str(clean_root / clean_tail(t)) for t in df["tail"]]
    df = df[[Path(p).exists() for p in df["clean_path"]]]
    df = df.drop_duplicates("tail")
    if n and n < len(df):
        df = df.sample(n=n, random_state=seed)
    out = pd.DataFrame({"study_id": "chexphoto:" + df["tail"], "patient_id": df["patient_id"],
                        "photo_path": [files[t] for t in df["tail"]], "clean_path": df["clean_path"],
                        **{f"label:{c}": df[c].values for c in LABELS if c in df.columns}})
    return out.sort_values("study_id").reset_index(drop=True), n_excluded


def label_vector(row: dict, names: list[str]) -> list[int]:
    """CheXpert labels -> CheXbert 'rrg' convention: positive and uncertain -> 1, negative / blank -> 0."""
    out = []
    for n in names:
        v = row.get(f"label:{n}")
        out.append(1 if v in (1, 1.0, -1, -1.0) else 0)
    return out
