"""Step 0 sanity tests on tiny synthetic fixtures that mimic each dataset's file layout.

These test data plumbing only (cleaning, sections, splits, leakage, priors). They produce no metrics.
Run: python -m pytest tests -q
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loaders import finalize, load_iu_xray, load_mimic_cxr, load_rexgradient  # noqa: E402
from src.data.splits import assert_no_leakage, select_eval_ids  # noqa: E402
from src.data.text_clean import is_non_informative, parse_generated, parse_sections, remove_phi  # noqa: E402
from src.eval.radeval_wrapper import cached_metric, results_table, target_texts  # noqa: E402

FILTERS = {"require_findings": True, "frontal_views": ["PA", "AP"], "min_alpha_chars": 10}


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (8, 8), 128).save(path)


# ----------------------------------------------------------------------------------- text
def test_remove_phi():
    assert remove_phi("Compared to ___ there is no change.") == "Compared to there is no change."
    assert remove_phi("Heart size normal. XXXX opacity.") == "Heart size normal. opacity."
    assert remove_phi(None) == ""


@pytest.mark.parametrize("t", ["As above.", "None.", "---", "", None, "see above", "N/A", "___"])
def test_non_informative(t):
    assert is_non_informative(t)


def test_informative():
    assert not is_non_informative("The lungs are clear. No pleural effusion.")


MIMIC_REPORT = """                                 FINAL REPORT
 EXAMINATION:  CHEST (PA AND LAT)

 INDICATION:  ___F with cough  // eval for pneumonia

 COMPARISON:  ___

 FINDINGS:
 PA and lateral views of the chest. The lungs are clear without focal
 consolidation. No pleural effusion or pneumothorax.

 IMPRESSION:
 No acute cardiopulmonary process.
"""


def test_parse_sections_mimic():
    s = parse_sections(MIMIC_REPORT)
    assert s["indication"] == "F with cough // eval for pneumonia"
    assert s["findings"].startswith("PA and lateral views of the chest. The lungs are clear")
    assert "pneumothorax." in s["findings"]
    assert s["impression"] == "No acute cardiopulmonary process."
    assert "___" not in " ".join(s.values())


def test_parse_sections_no_findings():
    s = parse_sections("IMPRESSION: Stable cardiomegaly.")
    assert s["findings"] == "" and s["impression"] == "Stable cardiomegaly."


@pytest.mark.parametrize("text,f,i,ok", [
    ("FINDINGS: Lungs clear.\nIMPRESSION: Normal.", "Lungs clear.", "Normal.", True),
    ("**Findings:** Lungs clear.\n\n**Impression:** Normal.", "Lungs clear.", "Normal.", True),
    ("The lungs are clear.", "The lungs are clear.", "", False),
    ("findings: A\nimpression: B", "A", "B", True),
])
def test_parse_generated(text, f, i, ok):
    p = parse_generated(text)
    assert (p["findings"], p["impression"], p["parse_ok"]) == (f, i, ok)


# ------------------------------------------------------------------------------- fixtures
@pytest.fixture
def iu_root(tmp_path):
    root = tmp_path / "iu"
    reports, proj = [], []
    for uid in range(1, 41):
        findings = "As above." if uid == 3 else f"Heart size normal. Lungs clear case {uid}. XXXX."
        reports.append({"uid": uid, "MeSH": "", "Problems": "", "image": "", "indication": "XXXX-year-old cough",
                        "comparison": "", "findings": None if uid == 4 else findings, "impression": "No acute disease."})
        proj.append({"uid": uid, "filename": f"{uid}_IM-1.dcm.png", "projection": "Frontal"})
        proj.append({"uid": uid, "filename": f"{uid}_IM-2.dcm.png", "projection": "Lateral"})
        if uid != 5:  # uid 5 has no frontal image file listed -> dropped
            pass
    proj = [p for p in proj if not (p["uid"] == 5 and p["projection"] == "Frontal")]
    root.mkdir()
    pd.DataFrame(reports).to_csv(root / "indiana_reports.csv", index=False)
    pd.DataFrame(proj).to_csv(root / "indiana_projections.csv", index=False)
    for p in proj:
        _png(root / "images" / "images_normalized" / p["filename"])
    return root


def iu_cfg(root, ann=None):
    return {"root": str(root), "reports_csv": "indiana_reports.csv", "projections_csv": "indiana_projections.csv",
            "images_dir": "images/images_normalized", "r2gen_annotation": ann,
            "fallback_split": {"train": 0.7, "val": 0.1, "test": 0.2}}


@pytest.fixture
def mimic_root(tmp_path):
    root = tmp_path / "mimic"
    split, meta = [], []
    # subjects 10000001..10000006, 3 studies each; subject 6 is test, 5 val, rest train
    for si, subj in enumerate(range(10000001, 10000007)):
        sp = "test" if si == 5 else ("validate" if si == 4 else "train")
        for k in range(3):
            study = 50000000 + si * 10 + k
            for view, dicom in [("PA", f"d{study}a"), ("LATERAL", f"d{study}b")]:
                split.append({"dicom_id": dicom, "study_id": study, "subject_id": subj, "split": sp})
                meta.append({"dicom_id": dicom, "subject_id": subj, "study_id": study, "ViewPosition": view,
                             "StudyDate": 21800101 + (2 - k) * 100, "StudyTime": 101010.0})  # k=2 is earliest
                _png(root / "files" / f"p{str(subj)[:2]}" / f"p{subj}" / f"s{study}" / f"{dicom}.jpg")
            rp = root / "reports" / f"p{str(subj)[:2]}" / f"p{subj}" / f"s{study}.txt"
            rp.parent.mkdir(parents=True, exist_ok=True)
            if not (si == 0 and k == 0):  # one missing report file
                body = MIMIC_REPORT if k != 1 else "IMPRESSION: As above."  # k=1 -> no findings -> dropped
                rp.write_text(body, encoding="utf-8")
    pd.DataFrame(split).to_csv(root / "split.csv.gz", index=False)
    pd.DataFrame(meta).to_csv(root / "meta.csv.gz", index=False)
    return root


def mimic_cfg(root):
    return {"split_csv": str(root / "split.csv.gz"), "metadata_csv": str(root / "meta.csv.gz"),
            "reports_root": str(root / "reports"), "images_root": str(root / "files"), "image_ext": ".jpg"}


@pytest.fixture
def rex_root(tmp_path):
    """Mimics the real files: metadata/*_view_position.json + flattened images/<study key>_<instance uid>.png."""
    root = tmp_path / "rex"
    (root / "metadata").mkdir(parents=True)
    (root / "images").mkdir()
    for split, fname, pids in [("train", "train_metadata_view_position.json", range(0, 10)),
                               ("val", "valid_metadata_view_position.json", range(10, 12)),
                               ("test", "test_metadata_view_position.json", range(12, 16))]:
        d = {}
        for pid in pids:
            for k in range(2):
                acc, suid = f"A{pid}{k}", f"1.2.{pid}.{k}"
                key = f"pP{pid}_a{acc}_s{suid}"
                insts = [f"9.{pid}.{k}.0", f"9.{pid}.{k}.1", f"9.{pid}.{k}.2"]
                paths = [f"../deid_png/P{pid}/{acc}/studies/{suid}/series/8.{pid}/instances/{i}.png" for i in insts]
                views = ["POSTERO_ANTERIOR" if pid % 2 else "AP", "LATERAL", "UNKNOWN"]
                if not (pid == 0 and k == 1):  # this study's images are not in the images folder
                    for i in insts:
                        _png(root / "images" / f"{key}_{i}.png")
                d[key] = {"PatientID": f"P{pid}", "AccessionNumber": acc, "StudyInstanceUid": suid,
                          "PatientWeight": float("nan"), "StudyDate": 20200101 + k, "Indication": "Cough.",
                          "Comparison": "None.", "Findings": "The lungs are clear. No effusion.",
                          "Impression": "Normal chest.", "ImagePath": paths, "ImageViewPosition": views}
        (root / "metadata" / fname).write_text(json.dumps(d))  # json.dumps writes bare NaN, like the real files
    return root


def rex_cfg(root):
    return {"root": str(root / "metadata"), "images_root": str(root / "images"),
            "metadata_files": {"train": "train_metadata_view_position.json",
                               "val": "valid_metadata_view_position.json",
                               "test": "test_metadata_view_position.json"}}


# ---------------------------------------------------------------------------------- loaders
def test_iu_seeded_split(iu_root):
    st = Counter()
    df = finalize(load_iu_xray(iu_cfg(iu_root), 42, st), FILTERS, st)
    assert st["dropped_empty_or_noninformative_findings"] == 2  # uid 3 ("As above.") and uid 4 (None)
    assert st["dropped_no_frontal_view"] == 1                     # uid 5
    assert len(df) == 37
    assert "XXXX" not in " ".join(df["findings"]) + " ".join(df["indication"])
    assert all(len(l) == 1 for l in df["lateral_image_paths"])
    assert_no_leakage(df)
    df2 = finalize(load_iu_xray(iu_cfg(iu_root), 42, Counter()), FILTERS, Counter())
    assert df["split"].tolist() == df2["split"].tolist()  # deterministic


def test_iu_r2gen_split(iu_root, tmp_path):
    ann = {"train": [{"id": f"CXR{u}_IM-0001"} for u in range(1, 30)],
           "val": [{"id": f"CXR{u}_IM-0001"} for u in range(30, 34)],
           "test": [{"id": f"CXR{u}_IM-0001"} for u in range(34, 40)]}
    p = tmp_path / "annotation.json"
    p.write_text(json.dumps(ann))
    st = Counter()
    df = finalize(load_iu_xray(iu_cfg(iu_root, str(p)), 42, st), FILTERS, st)
    assert st["split_method"] == "r2gen_annotation" and st["iu_not_in_r2gen_split"] == 1  # uid 40
    assert set(df.loc[df["split"] == "test", "study_id"]) == {str(u) for u in range(34, 40)}


def test_mimic_loader(mimic_root):
    st = Counter()
    df = finalize(load_mimic_cxr(mimic_cfg(mimic_root), 42, st, frontal_views=["PA", "AP"]), FILTERS, st)
    assert st["dropped_report_file_missing"] == 1
    assert st["dropped_empty_or_noninformative_findings"] == 6  # k=1 study of every subject
    assert len(df) == 18 - 1 - 6
    assert set(df["view"]) == {"PA"} and all(len(l) == 1 for l in df["lateral_image_paths"])
    assert set(df["split"]) == {"train", "val", "test"}
    # Priors: k=2 is the earliest study (no prior); k=0 is latest, prior = k=1 (even though k=1 was dropped).
    s = df.set_index("study_id")
    assert s.loc["50000052", "prior_study_id"] is None
    assert s.loc["50000050", "prior_study_id"] == "50000051"
    assert "___" not in " ".join(df["indication"])
    assert_no_leakage(df)


def test_rexgradient_loader(rex_root):
    st = Counter()
    df = finalize(load_rexgradient(rex_cfg(rex_root), 42, st, frontal_views=["PA", "AP"]), FILTERS, st)
    assert st["rexgradient_frontal_image_not_in_images_root"] == 1 and st["dropped_no_frontal_view"] == 1
    assert len(df) == 31
    assert df["image_path"].str.endswith(".0.png").all()          # frontal = first image; UNKNOWN view ignored
    assert set(df["view"]) == {"PA", "AP"}                         # POSTERO_ANTERIOR normalised to PA
    assert all(len(l) == 1 for l in df["lateral_image_paths"])
    assert st["rexgradient_view_values"]["UNKNOWN"] == 32
    s = df.set_index("study_id")
    assert s.loc["pP1_aA11_s1.2.1.1", "prior_study_id"] == "pP1_aA10_s1.2.1.0"
    assert s.loc["pP1_aA10_s1.2.1.0", "prior_study_id"] is None
    assert_no_leakage(df)


def test_rexgradient_bad_fields(rex_root):
    p = rex_root / "metadata" / "test_metadata_view_position.json"
    d = json.loads(p.read_text())
    for v in d.values():
        v.pop("Findings")
    p.write_text(json.dumps(d))
    with pytest.raises(Exception, match="missing fields"):
        load_rexgradient(rex_cfg(rex_root), 42, Counter())


# --------------------------------------------------------------------------------- leakage
def test_leakage_detected(iu_root):
    df = finalize(load_iu_xray(iu_cfg(iu_root), 42, Counter()), FILTERS, Counter())
    tr = df[df["split"] == "train"].iloc[0]
    j = df.index[df["split"] == "test"][0]
    df.loc[j, "patient_id"] = tr["patient_id"]
    with pytest.raises(AssertionError, match="PATIENT LEAKAGE"):
        assert_no_leakage(df)


def test_duplicate_study_detected(iu_root):
    df = finalize(load_iu_xray(iu_cfg(iu_root), 42, Counter()), FILTERS, Counter())
    df = pd.concat([df, df.iloc[[0]]])
    with pytest.raises(AssertionError, match="duplicate"):
        assert_no_leakage(df)


def test_eval_ids_fixed_subset():
    test = pd.DataFrame({"dataset": ["a"] * 30 + ["b"] * 5, "study_id": [str(i) for i in range(35)]})
    e1, e2 = select_eval_ids(test, 10, 7), select_eval_ids(test, 10, 7)
    assert e1 == e2 and len(e1["a"]) == 10 and len(e1["b"]) == 5
    assert set(e1["a"]) <= {str(i) for i in range(30)}


# ------------------------------------------------------------------------------------ eval
def test_cached_metric_matches_per_sample_mean():
    hyps = {"A": ["x", "y", "z"], "B": ["x", "q", "w"]}
    per = {"A": [1.0, 0.0, 0.5], "B": [1.0, 1.0, None]}
    fn = cached_metric(per, hyps)
    assert fn(hyps["A"], ["r"] * 3) == pytest.approx(0.5)
    assert fn(hyps["B"], ["r"] * 3) == pytest.approx(1.0)       # None excluded
    assert fn(["x", "q", "z"], ["r"] * 3) == pytest.approx(2.5 / 3)  # swapped mix


def test_target_texts_empty_hyp():
    rows = [{"reference_findings": "Clear.", "reference_impression": "Normal.", "hyp_findings": "",
             "hyp_impression": ""}]
    assert target_texts(rows, "findings") == (["Clear."], ["."])
    assert target_texts(rows, "report") == (["Clear. Normal."], ["."])


def test_results_table_missing_metrics_show_dash():
    t = results_table([{"system": "s", "dataset": "d", "target": "findings", "n": 3, "dtype": "fp32",
                        "metrics": {"radgraph_partial": 0.25}, "parse_ok_rate": 1.0}])
    assert "| 25.00 |" in t and "| - |" in t


# ------------------------------------------------------------------------------ end to end
def test_prepare_script_smoke(iu_root, mimic_root, rex_root, tmp_path):
    cfg = yaml.safe_load((ROOT / "configs" / "step0.yaml").read_text())
    cfg["datasets"]["iu_xray"].update(iu_cfg(iu_root))
    cfg["datasets"]["mimic_cxr"].update(mimic_cfg(mimic_root))
    cfg["datasets"]["rexgradient"].update(rex_cfg(rex_root))
    for d in cfg["datasets"].values():  # test all three loaders regardless of what the real config enables
        d["enabled"] = True
    cp = tmp_path / "cfg.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    env = {**os.environ, "PROJECT_ROOT": str(tmp_path / "proj")}
    r = subprocess.run([sys.executable, "scripts/step0_prepare_data.py", "--config", str(cp), "--smoke"],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "LEAKAGE CHECK PASSED" in r.stdout
    d = tmp_path / "proj" / "data"
    test = pd.read_parquet(d / "test_smoke.parquet")
    cols = {"study_id", "patient_id", "image_path", "image_paths", "view", "indication", "findings",
            "impression", "prior_study_id", "dataset", "lateral_image_paths"}
    assert cols <= set(test.columns)
    ids = json.loads((d / "step0_eval_ids_smoke.json").read_text())
    assert set(ids) == {"iu_xray", "mimic_cxr", "rexgradient"}
    man = json.loads((d / "manifest_smoke.json").read_text())
    assert man["leakage_check"]["passed"]
    assert list(d.glob("run_info_*.json"))


# ---------------------------------------------------------------------------------- images
def test_16bit_xray_not_clipped(tmp_path):
    """ReXGradient PNGs are 16-bit; PIL's convert('RGB') would clip them to white."""
    import numpy as np
    from src.models.medgemma_zeroshot import load_image

    a = np.linspace(0, 4095, 64 * 64).reshape(64, 64).astype(np.uint16)
    p = tmp_path / "x16.png"
    Image.fromarray(a).save(p)
    rgb = np.asarray(load_image(str(p)))
    assert rgb.shape == (64, 64, 3) and rgb.dtype == np.uint8
    assert rgb.min() == 0 and rgb.max() == 255
    assert (rgb[..., 0] == 255).mean() < 0.01           # was 0.94 with plain convert("RGB")
    assert abs(float(rgb.mean()) - 127.5) < 2           # linear ramp keeps its mid-grey mean


def test_8bit_image_unchanged(tmp_path):
    import numpy as np
    from src.models.medgemma_zeroshot import load_image

    a = np.arange(256, dtype=np.uint8).reshape(16, 16)
    p = tmp_path / "x8.png"
    Image.fromarray(a, mode="L").save(p)
    assert (np.asarray(load_image(str(p)))[..., 0] == a).all()
