"""Step 1 sanity tests that need no GPU: train-subset selection/shrinking, multi-root image lookup, SFT examples,
train-only row selection, config inheritance, adapter digest. (The Unsloth trainer itself is checked by the
tensor-level sanity check at the start of scripts/step1_train_qlora.py, on Kaggle.)"""
from __future__ import annotations

import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.common import load_config  # noqa: E402
from src.data.loaders import load_rexgradient  # noqa: E402
from src.data.rexgradient_fetch import extract_wanted, first_frontal, select_subset, shrink_png  # noqa: E402
from src.models.medgemma_zeroshot import adapter_digest, build_prompt  # noqa: E402
from src.models.sft_data import SFTImageDataset, build_messages, select_train_rows  # noqa: E402
from test_step0 import rex_cfg, rex_root  # noqa: E402,F401  (Step 0 ReXGradient fixture)

CODE = Path(__file__).resolve().parents[1]
FV = ["PA", "AP"]


def _png16(w=300, h=200) -> bytes:
    a = (np.linspace(0, 4095, w * h).reshape(h, w)).astype(np.uint16)  # 12-bit ramp, like real X-rays
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, format="PNG")
    return buf.getvalue()


def _meta():
    p = lambda pid, i: f"../deid_png/P{pid}/A{pid}/studies/S{pid}/series/X{pid}/instances/U{pid}_{i}.png"  # noqa: E731
    return {
        "s_pa_ap": {"Findings": "The lungs are clear. No effusion.", "ImagePath": [p(1, 0), p(1, 1), p(1, 2)],
                    "ImageViewPosition": ["AP", "LATERAL", "POSTERO_ANTERIOR"]},
        "s_ap": {"Findings": "Mild cardiomegaly without edema.", "ImagePath": [p(2, 0), p(2, 1)],
                 "ImageViewPosition": ["LATERAL", "AP"]},
        "s_lat_only": {"Findings": "Clear lungs bilaterally here.", "ImagePath": [p(3, 0)],
                       "ImageViewPosition": ["LATERAL"]},
        "s_noninf": {"Findings": "None.", "ImagePath": [p(4, 0)], "ImageViewPosition": ["PA"]},
        "s_nan": {"Findings": float("nan"), "ImagePath": [p(5, 0)], "ImageViewPosition": ["PA"]},
    }


# ------------------------------------------------------------------------ train-subset fetch
def test_first_frontal_prefers_pa():
    m = _meta()
    assert first_frontal(m["s_pa_ap"]["ImagePath"], m["s_pa_ap"]["ImageViewPosition"], {"PA", "AP"}).endswith("U1_2.png")
    assert first_frontal(m["s_ap"]["ImagePath"], m["s_ap"]["ImageViewPosition"], {"PA", "AP"}).endswith("U2_1.png")
    assert first_frontal(m["s_lat_only"]["ImagePath"], m["s_lat_only"]["ImageViewPosition"], {"PA", "AP"}) is None


def test_select_subset_applies_step0_filters():
    sub = select_subset(_meta(), None, 42, FV, 10)
    assert set(sub) == {"s_pa_ap", "s_ap"}            # lateral-only, "None." and NaN findings are dropped
    assert sub["s_pa_ap"] == "P1/A1/studies/S1/series/X1/instances/U1_2.png"   # relative to deid_png/


def test_select_subset_seeded_and_sized():
    big = {f"k{i}": {"Findings": "The lungs are clear. No effusion.", "ImagePath": [f"../deid_png/P{i}/x.png"],
                     "ImageViewPosition": ["PA"]} for i in range(100)}
    a, b = select_subset(big, 10, 42, FV, 10), select_subset(big, 10, 42, FV, 10)
    assert a == b and len(a) == 10
    assert select_subset(big, 10, 7, FV, 10) != a


def test_shrink_png_16bit_not_clipped():
    out = Image.open(io.BytesIO(shrink_png(_png16(), 64)))
    assert out.mode == "L" and out.size == (64, 64)
    a = np.asarray(out)
    assert a.min() <= 5 and a.max() >= 250             # full 0..255 range, not clipped to white
    assert (a == 255).mean() < 0.1


def test_extract_with_transform(tmp_path):
    data = {"deid_png/P1/a.png": _png16(), "deid_png/P2/b.png": _png16(50, 40)}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, b in data.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(b)
            tf.addfile(ti, io.BytesIO(b))
    buf.seek(0)
    st = extract_wanted(buf, {"P1/a.png"}, tmp_path, transform=lambda b: shrink_png(b, 32))
    assert st["written"] == 1 and not st["missing"]
    assert Image.open(tmp_path / "P1/a.png").size == (32, 32) and not (tmp_path / "P2/b.png").exists()
    buf.seek(0)
    st2 = extract_wanted(buf, {"P1/a.png"}, tmp_path, transform=lambda b: shrink_png(b, 32))
    assert st2["already_present"] == 1 and st2["written"] == 0   # re-run does not rewrite


# ------------------------------------------------------------------------ multi-root loader
def test_rexgradient_extra_image_root(rex_root, tmp_path):
    root = rex_root
    # The study pP0_aA01_s1.2.0.1 has no image in the main folder; put its frontal image in an extra root,
    # in the original (unflattened) deid_png layout used by the fetch script.
    extra = tmp_path / "train_subset" / "deid_png"
    rel = "P0/A01/studies/1.2.0.1/series/8.0/instances/9.0.1.0.png"
    (extra / rel).parent.mkdir(parents=True)
    Image.new("L", (16, 16)).save(extra / rel)
    cfg = {**rex_cfg(root), "extra_images_roots": [str(extra), str(tmp_path / "does_not_exist")]}
    st = Counter()
    df = load_rexgradient(cfg, 42, st)
    row = df[df["study_id"] == "pP0_aA01_s1.2.0.1"].iloc[0]
    assert row["image_paths"] == [str(extra / rel)]
    assert st["rexgradient_image_roots"] == [cfg["images_root"], str(extra)]   # missing root skipped


# ------------------------------------------------------------------------ SFT examples
TEMPLATE = "Write the report.\nIndication: {indication}\nFINDINGS: ...\nIMPRESSION: ..."


def test_build_messages_matches_zero_shot_prompt():
    r = {"indication": "  Cough. ", "findings": "Clear lungs.", "impression": "Normal."}
    m = build_messages(r, TEMPLATE, "Not provided.")
    assert m[0]["role"] == "user" and m[0]["content"][0] == {"type": "image"}
    assert m[0]["content"][1]["text"] == build_prompt(TEMPLATE, r["indication"], "Not provided.")
    assert m[1] == {"role": "assistant", "content": [{"type": "text", "text": "FINDINGS: Clear lungs.\nIMPRESSION: Normal."}]}
    m2 = build_messages({"indication": "", "findings": "Clear.", "impression": ""}, TEMPLATE, "Not provided.")
    assert "Indication: Not provided." in m2[0]["content"][1]["text"]


def _df():
    rows = []
    for ds, n in [("rexgradient", 30), ("iu_xray", 8)]:
        for split in ["train", "val", "test"]:
            for i in range(n):
                rows.append({"dataset": ds, "split": split, "study_id": f"{ds}-{split}-{i}",
                             "patient_id": f"{ds}:{split}{i}", "image_path": "x.png", "indication": "",
                             "findings": "Clear lungs.", "impression": ""})
    return pd.DataFrame(rows)


def test_select_train_rows_train_only_seeded():
    df = _df()
    a = select_train_rows(df, {"rexgradient": 10, "iu_xray": "all"}, 42)
    assert set(a["split"]) == {"train"} and len(a) == 18
    assert a["dataset"].value_counts().to_dict() == {"rexgradient": 10, "iu_xray": 8}
    assert a["study_id"].tolist() == select_train_rows(df, {"rexgradient": 10, "iu_xray": "all"}, 42)["study_id"].tolist()
    assert "mimic_cxr" not in set(a["dataset"])


def test_sft_dataset_loads_image(tmp_path):
    p = tmp_path / "x.png"
    Image.fromarray(np.full((8, 8), 1000, np.uint16)).save(p)
    ds = SFTImageDataset([{"image_path": str(p), "indication": "", "findings": "Clear lungs.", "impression": ""}],
                         TEMPLATE, "Not provided.")
    ex = ds[0]
    img = ex["messages"][0]["content"][0]["image"]
    assert img.mode == "RGB" and len(list(ds)) == 1
    with pytest.raises(IndexError):
        ds[1]


# ------------------------------------------------------------------------ config + adapter digest
def test_step1_config_inherits_step0():
    c0, c1 = load_config(CODE / "configs/step0.yaml"), load_config(CODE / "configs/step1.yaml")
    assert c1["generation"]["prompt_template"] == c0["generation"]["prompt_template"]
    assert c1["generation"]["max_new_tokens"] == c0["generation"]["max_new_tokens"]
    assert c1["generation"]["dtype"] == "nf4" and c0["generation"]["dtype"] == "fp32"
    assert c1["datasets"] == c0["datasets"] and c1["seed"] == c0["seed"]
    t = c1["train"]
    assert t["per_device_batch"] * t["grad_accum"] == 16 and t["lora"] == {"r": 16, "alpha": 16, "dropout": 0.05}
    assert c1["models"][c1["model_a"]["base"]] == "google/medgemma-1.5-4b-it"
    assert t["finetune_vision_layers"] is False


def test_adapter_digest(tmp_path):
    assert adapter_digest(None) is None
    (tmp_path / "adapter_model.safetensors").write_bytes(b"abc")
    d1 = adapter_digest(str(tmp_path))
    (tmp_path / "adapter_model.safetensors").write_bytes(b"abd")
    assert d1 != adapter_digest(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        adapter_digest(str(tmp_path / "missing"))
