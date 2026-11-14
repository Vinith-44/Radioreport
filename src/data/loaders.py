"""Dataset loaders -> one study-level DataFrame per dataset with a shared schema.

Schema (one row per study):
  dataset, study_id, patient_id (namespaced "<dataset>:<id>"), split (train/val/test),
  image_path (first frontal), image_paths (all frontal), view, lateral_image_paths,
  indication, findings, impression, prior_study_id, study_datetime
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .text_clean import is_non_informative, normalize, parse_sections

SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validate": "val", "validation": "val", "test": "test"}
COLUMNS = ["dataset", "study_id", "patient_id", "split", "image_path", "image_paths", "view",
           "lateral_image_paths", "indication", "findings", "impression", "prior_study_id",
           "study_datetime"]
LATERAL_VIEWS = {"LATERAL", "LL", "RL", "LAO", "RAO", "LPO", "RPO", "LEFT LATERAL", "RIGHT LATERAL"}


class SchemaError(RuntimeError):
    pass


def _norm_split(s: str) -> str:
    key = str(s).strip().lower()
    if key not in SPLIT_ALIASES:
        raise SchemaError(f"Unknown split value {s!r}")
    return SPLIT_ALIASES[key]


def _require_columns(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SchemaError(f"{where}: missing columns {missing}. Available columns: {list(df.columns)}")


def _smoke_limit(df: pd.DataFrame, key: str, n: int | None) -> pd.DataFrame:
    """Deterministic per-split subset of studies (first n by sorted id) for --smoke."""
    if not n:
        return df
    keep = (df[[key, "split"]].drop_duplicates().sort_values(key)
            .groupby("split", group_keys=False).head(n)[key])
    return df[df[key].isin(set(keep))]


def finalize(df: pd.DataFrame, filters: dict, stats: Counter) -> pd.DataFrame:
    """Clean text, apply Step 0 filters, count every drop reason."""
    df = df.copy()
    for c in ["indication", "findings", "impression"]:
        df[c] = df[c].map(normalize)
    stats["studies_before_filters"] += len(df)

    # Masks use astype(bool): an empty object-dtype mask would otherwise be read as a column list.
    no_frontal = df["image_paths"].map(len).eq(0).astype(bool)
    stats["dropped_no_frontal_view"] += int(no_frontal.sum())
    df = df.loc[~no_frontal]

    min_alpha = int(filters.get("min_alpha_chars", 10))
    f_bad = df["findings"].map(lambda t: is_non_informative(t, min_alpha)).astype(bool)
    i_bad = df["impression"].map(lambda t: is_non_informative(t, min_alpha)).astype(bool)
    if filters.get("require_findings", True):
        stats["dropped_empty_or_noninformative_findings"] += int(f_bad.sum())
        df = df.loc[~f_bad].copy()
    else:
        both = f_bad & i_bad
        stats["dropped_empty_or_noninformative_report"] += int(both.sum())
        df = df.loc[~both].copy()
    # Keep studies with informative findings but blank out a non-informative impression ("As above.").
    df.loc[df["impression"].map(lambda t: is_non_informative(t, 1)).astype(bool), "impression"] = ""

    df["image_path"] = df["image_paths"].map(lambda p: p[0]) if len(df) else pd.Series(dtype=object)
    stats["studies_kept"] += len(df)
    return df[COLUMNS].reset_index(drop=True)


def _priors(df: pd.DataFrame) -> pd.Series:
    """prior_study_id = the patient's previous study by datetime (None if unknown / first study)."""
    if df["study_datetime"].isna().all():
        return pd.Series([None] * len(df), index=df.index, dtype=object)
    order = df.sort_values(["patient_id", "study_datetime", "study_id"])
    prev = order.groupby("patient_id")["study_id"].shift(1)
    prev[order["study_datetime"].isna()] = None
    return prev.reindex(df.index).astype(object).where(lambda s: s.notna(), None)


# --------------------------------------------------------------------------------------- IU-Xray
def load_iu_xray(cfg: dict, seed: int, stats: Counter, smoke_n: int | None = None,
                 frontal_views: list[str] | None = None) -> pd.DataFrame:
    root = Path(cfg["root"])
    reports = pd.read_csv(root / cfg["reports_csv"])
    proj = pd.read_csv(root / cfg["projections_csv"])
    _require_columns(reports, ["uid", "indication", "findings", "impression"], "IU reports_csv")
    _require_columns(proj, ["uid", "filename", "projection"], "IU projections_csv")
    img_dir = root / cfg["images_dir"]

    proj["projection"] = proj["projection"].astype(str).str.strip().str.lower()
    frontal = proj[proj["projection"] == "frontal"].groupby("uid")["filename"].apply(list)
    lateral = proj[proj["projection"] == "lateral"].groupby("uid")["filename"].apply(list)

    if cfg.get("r2gen_annotation"):
        ann = json.loads(Path(cfg["r2gen_annotation"]).read_text(encoding="utf-8"))
        uid_split = {}
        for split, items in ann.items():
            for it in items:
                m = re.match(r"CXR(\d+)_", it["id"])
                if not m:
                    raise SchemaError(f"Unexpected R2Gen id {it['id']!r}")
                uid_split[int(m.group(1))] = _norm_split(split)
        stats["split_method"] = "r2gen_annotation"
        n_all = len(reports)
        reports = reports[reports["uid"].isin(uid_split)].copy()
        stats["iu_not_in_r2gen_split"] = int(n_all - len(reports))
        reports["split"] = reports["uid"].map(uid_split)
    else:
        # Seeded split by patient. IU (OpenI) has one study per uid and no separate patient key,
        # so uid is used as the patient id (assumption recorded in the manifest).
        stats["split_method"] = "seeded_patient_split (NOT R2Gen)"
        rng = np.random.default_rng(seed)
        uids = np.array(sorted(reports["uid"].unique()))
        rng.shuffle(uids)
        fr = cfg.get("fallback_split", {"train": 0.7, "val": 0.1, "test": 0.2})
        n_tr, n_va = int(len(uids) * fr["train"]), int(len(uids) * fr["val"])
        m = {u: "train" for u in uids[:n_tr]} | {u: "val" for u in uids[n_tr:n_tr + n_va]} | \
            {u: "test" for u in uids[n_tr + n_va:]}
        reports = reports.copy()
        reports["split"] = reports["uid"].map(m)
    stats["patient_id_assumption"] = "IU: uid used as patient id (one study per uid)"

    reports = _smoke_limit(reports, "uid", smoke_n)
    df = pd.DataFrame({
        "dataset": "iu_xray",
        "study_id": reports["uid"].astype(str).values,
        "patient_id": ("iu_xray:" + reports["uid"].astype(str)).values,
        "split": reports["split"].values,
        "image_paths": [[str(img_dir / f) for f in frontal.get(u, [])] for u in reports["uid"]],
        "view": "FRONTAL",  # IU projections only say Frontal/Lateral, not PA/AP
        "lateral_image_paths": [[str(img_dir / f) for f in lateral.get(u, [])] for u in reports["uid"]],
        "indication": reports["indication"].values,
        "findings": reports["findings"].values,
        "impression": reports["impression"].values,
        "prior_study_id": None,
        "study_datetime": np.nan,
    })
    return df


# ------------------------------------------------------------------------------------ MIMIC-CXR
def _read_text(p: Path) -> str | None:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None


def load_mimic_cxr(cfg: dict, seed: int, stats: Counter, smoke_n: int | None = None,
                   frontal_views: list[str] | None = None) -> pd.DataFrame:
    frontal_views = {v.upper() for v in (frontal_views or ["PA", "AP"])}
    split = pd.read_csv(cfg["split_csv"], dtype={"dicom_id": str})
    meta = pd.read_csv(cfg["metadata_csv"], dtype={"dicom_id": str})
    _require_columns(split, ["dicom_id", "study_id", "subject_id", "split"], "MIMIC split_csv")
    _require_columns(meta, ["dicom_id", "ViewPosition", "StudyDate", "StudyTime"], "MIMIC metadata_csv")
    stats["split_method"] = "official mimic-cxr-2.0.0-split"

    img = split.merge(meta[["dicom_id", "ViewPosition", "StudyDate", "StudyTime"]], on="dicom_id", how="left")
    img["split"] = img["split"].map(_norm_split)
    img["view_u"] = img["ViewPosition"].fillna("").astype(str).str.upper()
    img["dt"] = pd.to_numeric(img["StudyDate"], errors="coerce") * 1e6 + \
        pd.to_numeric(img["StudyTime"], errors="coerce").fillna(0)

    n_split = img.groupby("study_id")["split"].nunique()
    if (n_split > 1).any():
        raise AssertionError(f"MIMIC: {int((n_split > 1).sum())} studies have images in >1 split")

    # Prior links are computed over ALL studies with metadata (before any filtering/smoke subset).
    st = img.groupby("study_id").agg(subject_id=("subject_id", "first"), split=("split", "first"),
                                     study_datetime=("dt", "min")).reset_index()
    st["patient_id"] = "mimic_cxr:" + st["subject_id"].astype(str)
    st["study_id_s"] = st["study_id"].astype(str)
    tmp = pd.DataFrame({"patient_id": st["patient_id"], "study_datetime": st["study_datetime"],
                        "study_id": st["study_id_s"]})
    st["prior_study_id"] = _priors(tmp).values

    st = _smoke_limit(st, "study_id", smoke_n)

    root = Path(cfg["images_root"])
    ext = cfg.get("image_ext", ".jpg")
    def _img_path(r) -> str:
        s = str(r.subject_id)
        return str(root / f"p{s[:2]}" / f"p{s}" / f"s{r.study_id}" / f"{r.dicom_id}{ext}")
    img = img[img["study_id"].isin(set(st["study_id"]))].copy()
    img["path"] = [_img_path(r) for r in img.itertuples()]
    # PA before AP when a study has both, then by dicom_id for determinism.
    img["vrank"] = img["view_u"].map(lambda v: 0 if v == "PA" else 1)
    img = img.sort_values(["study_id", "vrank", "dicom_id"])
    fr = img[img["view_u"].isin(frontal_views)].groupby("study_id")
    lat = img[img["view_u"].isin(LATERAL_VIEWS)].groupby("study_id")["path"].apply(list)
    fr_paths, fr_view = fr["path"].apply(list), fr["view_u"].first()

    rroot = Path(cfg["reports_root"])
    rpaths = [rroot / f"p{str(s)[:2]}" / f"p{s}" / f"s{t}.txt" for s, t in zip(st["subject_id"], st["study_id"])]
    with ThreadPoolExecutor(max_workers=16) as ex:
        texts = list(ex.map(_read_text, rpaths))
    secs = [parse_sections(t) if t is not None else None for t in texts]
    missing = [s is None for s in secs]
    stats["dropped_report_file_missing"] += int(sum(missing))
    keep = ~np.array(missing)
    st, secs = st[keep], [s for s in secs if s is not None]

    return pd.DataFrame({
        "dataset": "mimic_cxr",
        "study_id": st["study_id_s"].values,
        "patient_id": st["patient_id"].values,
        "split": st["split"].values,
        "image_paths": [fr_paths.get(s, []) for s in st["study_id"]],
        "view": [fr_view.get(s, None) for s in st["study_id"]],
        "lateral_image_paths": [lat.get(s, []) for s in st["study_id"]],
        "indication": [s["indication"] for s in secs],
        "findings": [s["findings"] for s in secs],
        "impression": [s["impression"] for s in secs],
        "prior_study_id": st["prior_study_id"].values,
        "study_datetime": st["study_datetime"].values,
    })


# ---------------------------------------------------------------------------------- ReXGradient
# View strings seen in the real test metadata: AP, PA, LATERAL, LL, UNKNOWN, None, N/A, POSTERO_ANTERIOR, LAT, ...
VIEW_ALIASES = {"POSTERO_ANTERIOR": "PA", "POSTEROANTERIOR": "PA", "ANTERO_POSTERIOR": "AP",
                "ANTEROPOSTERIOR": "AP", "LAT": "LATERAL", "LEFT_LATERAL": "LL"}
_UID_PATH_RE = re.compile(r"studies/([^/]+)/series/([^/]+)/instances/([^/]+)\.png$")


def _norm_view(v) -> str:
    s = str(v).strip().upper() if v is not None else "UNKNOWN"
    return VIEW_ALIASES.get(s, s)


def _index_images(images_root: Path) -> set[str]:
    """All file paths (relative to images_root, '/'-separated) - one scan instead of per-file stat calls."""
    out = set()
    for dirpath, _, files in os.walk(images_root):
        rel = Path(dirpath).relative_to(images_root).as_posix()
        for f in files:
            out.add(f if rel == "." else f"{rel}/{f}")
    return out


def _resolve_image(study_key: str, raw_path: str, available: set[str]) -> str | None:
    """Map an original ReXGradient ImagePath to a file that exists in the (possibly flattened) image folder."""
    m = _UID_PATH_RE.search(raw_path.replace("\\", "/"))
    cands = []
    if m:
        _, series, inst = m.groups()
        cands += [f"{study_key}_{inst}.png", f"{study_key}_{series}.png"]
    rel = raw_path.replace("\\", "/").split("deid_png/", 1)[-1]
    cands += [rel, f"deid_png/{rel}"]
    return next((c for c in cands if c in available), None)


def _load_rex_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        # The files contain bare NaN tokens (valid for Python's json module, not strict JSON).
        return json.load(f)


def inspect_rexgradient(cfg: dict) -> str:
    """Print-friendly summary of the real files/fields and image-name matching."""
    root, img_root = Path(cfg["root"]), Path(cfg["images_root"])
    lines = [f"ReXGradient metadata root: {root} (exists={root.exists()})",
             f"ReXGradient images root:   {img_root} (exists={img_root.exists()})"]
    available = _index_images(img_root) if img_root.exists() else set()
    lines.append(f"  image files found: {len(available)}; e.g. {sorted(available)[:2]}")
    for split, rel in cfg["metadata_files"].items():
        p = root / rel
        if not p.exists():
            lines.append(f"  [{split}] {p} MISSING")
            continue
        d = _load_rex_json(p)
        k0 = next(iter(d))
        lines.append(f"  [{split}] {rel}: {len(d)} studies; fields={list(d[k0])}")
        hit = sum(any(_resolve_image(k, ip, available) for ip in v.get("ImagePath") or []) for k, v in d.items())
        lines.append(f"      studies with >=1 image present in images_root: {hit}/{len(d)}")
    return "\n".join(lines)


def load_rexgradient(cfg: dict, seed: int, stats: Counter, smoke_n: int | None = None,
                     frontal_views: list[str] | None = None) -> pd.DataFrame:
    frontal_views = {v.upper() for v in (frontal_views or ["PA", "AP"])}
    root, img_root = Path(cfg["root"]), Path(cfg["images_root"])
    available = _index_images(img_root)
    stats["split_method"] = "official ReXGradient split files"
    stats["rexgradient_image_files_indexed"] = len(available)

    recs = []
    for split, rel in cfg["metadata_files"].items():
        d = _load_rex_json(root / rel)
        for key, v in d.items():
            missing = [f for f in ("PatientID", "Findings", "Impression", "ImagePath", "ImageViewPosition")
                       if f not in v]
            if missing:
                raise SchemaError(f"ReXGradient {rel} study {key}: missing fields {missing}; has {list(v)}")
            recs.append((key, _norm_split(split), v))
    ids = pd.Series([r[0] for r in recs])
    if ids.duplicated().any():
        raise AssertionError(f"ReXGradient: study ids in >1 split file, e.g. {ids[ids.duplicated()].head(3).tolist()}")
    # Official split must be patient-disjoint BEFORE any filtering (filters could hide an overlap).
    pats: dict[str, set] = {}
    for _, split, v in recs:
        pats.setdefault(split, set()).add(v["PatientID"])
    for a in pats:
        for b in pats:
            if a < b and pats[a] & pats[b]:
                raise AssertionError(f"ReXGradient PATIENT LEAKAGE {a}/{b} in official metadata: "
                                     f"{len(pats[a] & pats[b])} patients")
    if smoke_n:
        df_ids = pd.DataFrame({"k": ids, "split": [r[1] for r in recs]})
        keep = set(df_ids.sort_values("k").groupby("split").head(smoke_n * 20)["k"])  # extra: many lack images
        recs = [r for r in recs if r[0] in keep]

    views_seen: Counter = Counter()
    rows = []
    for key, split, v in recs:
        paths, views = list(v["ImagePath"] or []), [_norm_view(x) for x in (v["ImageViewPosition"] or [])]
        if len(paths) != len(views):
            raise SchemaError(f"ReXGradient study {key}: {len(paths)} paths vs {len(views)} views")
        views_seen.update(views)
        fr, lat, fr_view, n_unresolved = [], [], [], 0
        for p, vw in sorted(zip(paths, views), key=lambda t: (0 if t[1] == "PA" else 1, t[0])):
            if vw not in frontal_views and vw not in LATERAL_VIEWS:
                continue
            f = _resolve_image(key, p, available)
            if f is None:
                n_unresolved += 1
                continue
            (fr if vw in frontal_views else lat).append(str(img_root / f))
            if vw in frontal_views:
                fr_view.append(vw)
        if not fr and n_unresolved and any(vw in frontal_views for vw in views):
            stats["rexgradient_frontal_image_not_in_images_root"] += 1
        sd = pd.to_datetime(str(v.get("StudyDate")), format="%Y%m%d", errors="coerce")
        rows.append({
            "dataset": "rexgradient", "study_id": key, "patient_id": f"rexgradient:{v['PatientID']}",
            "split": split, "image_paths": fr, "view": fr_view[0] if fr_view else None,
            "lateral_image_paths": lat, "indication": v.get("Indication"),
            "findings": v.get("Findings"), "impression": v.get("Impression"),
            "study_datetime": sd.timestamp() if pd.notna(sd) else np.nan,
        })
    stats["rexgradient_view_values"] = dict(views_seen.most_common(20))
    out = pd.DataFrame(rows)
    out["prior_study_id"] = _priors(out).values
    return out


LOADERS = {"iu_xray": load_iu_xray, "mimic_cxr": load_mimic_cxr, "rexgradient": load_rexgradient}
