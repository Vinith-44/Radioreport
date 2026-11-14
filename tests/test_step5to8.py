"""Steps 5-8 tests (no GPU, no model downloads): prior-report prompt, winner rule, LoRA on a tiny Gemma 3 (skipped if
peft is missing), photo augmentations, the fixed photo test seed, CheXphoto pairing + leakage exclusion, translation
plumbing + label agreement, second-reader calibration / flagging / deletion simulation, label micro-F1."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))
sys.path.insert(0, str(CODE / "tests"))

from src.retrieval.rerank import CHEXBERT_LABELS  # noqa: E402


# ------------------------------------------------------------------ Step 5
def test_add_prior_and_prior_texts():
    from step4_train_projector import add_prior, prior_texts

    p = "Header line.\nIndication: Cough.\nReply in exactly this format:"
    assert add_prior(p, "FINDINGS: old.\nIMPRESSION: x.") == \
        "Header line.\nIndication: Cough.\nPrior report: FINDINGS: old. IMPRESSION: x.\nReply in exactly this format:"
    assert add_prior(p, "") == p and add_prior(p, None) == p
    pool = pd.DataFrame({"study_id": ["s1", "s2"], "findings": ["Old finding.", "New."], "impression": ["", ""]})
    rows = pd.DataFrame({"study_id": ["s2", "s3"], "prior_study_id": ["s1", "s9"]})
    assert prior_texts(rows, pool) == ["FINDINGS: Old finding.\nIMPRESSION: ", ""]   # s9 not in this split


def test_winner_rule():
    from step5_score import pick_winner

    a = {"x": {"radgraph_partial": 0.30, "f1chexbert_all_micro_f1": 0.40}}
    b = {"x": {"radgraph_partial": 0.25, "f1chexbert_all_micro_f1": 0.60}}
    assert pick_winner(a, b, 0.005)[0] == "model-a"
    b["x"]["radgraph_partial"] = 0.298                          # within tie -> CheXbert decides
    assert pick_winner(a, b, 0.005)[0] == "model-b"


def test_lora_on_tiny_gemma_trains_projector_and_lora_only(tmp_path):
    peft = pytest.importorskip("peft")
    import torch
    from test_step4 import TinyTok, _tiny_model

    from src.models.model_b import ModelB, prompt_pieces, target_ids

    _, vis, llm, proj = _tiny_model()
    llm = peft.get_peft_model(llm, peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                                                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                                   "gate_proj", "up_proj", "down_proj"]))
    model = ModelB(vis, llm, proj, torch.float32)
    tok = TinyTok()
    pre, post = prompt_pieces(tok, "Indication: x\nPrior report: FINDINGS: old.")
    tgt = [target_ids(tok, "FINDINGS: a. IMPRESSION: b.")]
    px = torch.randn(1, 3, 56, 56)
    lora = [p for n, p in llm.named_parameters() if p.requires_grad]
    assert lora and all("lora_" in n for n, p in llm.named_parameters() if p.requires_grad)
    opt = torch.optim.AdamW([{"params": list(proj.parameters()), "lr": 1e-2}, {"params": lora, "lr": 1e-3}])
    base = {n: p.clone() for n, p in llm.named_parameters() if not p.requires_grad}
    llm.train()
    l0 = None
    for _ in range(20):
        opt.zero_grad()
        loss = model(px, [pre], [post], tgt).loss
        loss.backward()
        opt.step()
        l0 = l0 if l0 is not None else float(loss.detach())
    assert float(loss.detach()) < l0
    assert all(torch.equal(p, base[n]) for n, p in llm.named_parameters() if not p.requires_grad)   # base frozen
    llm.save_pretrained(str(tmp_path / "lora"))
    assert (tmp_path / "lora" / "adapter_config.json").exists()


# ------------------------------------------------------------------ Step 6
def test_photo_augment_ops_and_determinism():
    from src.data import augment as A

    # textured test image (a smooth ramp would be left unchanged by blur, correctly)
    im = Image.fromarray((np.random.default_rng(0).random((100, 120)) * 255).astype(np.uint8))
    for name, fn, _ in A.PIPELINE:
        out = fn(im, random.Random(1))
        assert out.size == im.size and out.mode == "RGB", name
        assert not np.array_equal(np.asarray(out.convert("L")), np.asarray(im)), name
    a, ops_a = A.photo_augment(im, random.Random(7), return_ops=True)
    b, ops_b = A.photo_augment(im, random.Random(7), return_ops=True)
    assert ops_a == ops_b and len(ops_a) >= 2 and np.array_equal(np.asarray(a), np.asarray(b))
    assert [o for o in ops_a] == [n for n, _, _ in A.PIPELINE if n in ops_a]    # physical order kept


def test_load_for_siglip_with_transform(tmp_path):
    from src.data.augment import photo_augment
    from src.retrieval.embed import load_for_siglip

    p = tmp_path / "x.png"
    Image.fromarray((np.random.default_rng(0).random((300, 260)) * 255).astype(np.uint8)).save(p)
    out = load_for_siglip(str(p), transform=lambda g: photo_augment(g, random.Random(3)))
    assert out.size == (448, 448) and out.mode == "RGB"
    assert not np.array_equal(np.asarray(out), np.asarray(load_for_siglip(str(p))))


def test_photo_test_seed_is_per_study_and_fixed():
    from step6_make_photo_test import study_rng

    assert study_rng(2026, "a").random() == study_rng(2026, "a").random()
    assert study_rng(2026, "a").random() != study_rng(2026, "b").random()
    assert study_rng(2026, "a").random() != study_rng(1, "a").random()


def test_chexphoto_pairs_and_leakage_exclusion(tmp_path):
    from src.data.chexphoto import LABELS, clean_tail, label_vector, load_pairs

    root, clean = tmp_path / "cxp", tmp_path / "chexpert"
    sub = root / "CheXphoto-v1.0_split_1" / "CheXphoto-v1.0_split_1"
    recs = []
    for pid, lab in [("patient00001", 1.0), ("patient00002", -1.0), ("patient00003", 0.0)]:
        t = f"train/natural/iphone/{pid}/study1/view1_frontal.jpg"
        (sub / t).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (30, 30)).save(sub / t)
        (clean / clean_tail(t)).parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (30, 30)).save(clean / clean_tail(t))
        r = {c: np.nan for c in LABELS}
        r.update({"Path": f"CheXphoto-v1.0/{t}", "Pleural Effusion": lab})
        recs.append(r)
    pd.DataFrame(recs).to_csv(sub / "train.csv", index=False)
    pairs, n_ex = load_pairs(root, clean, ["natural"], {"chexpert:patient00001"}, None, 42)
    assert n_ex == 1 and len(pairs) == 2 and "chexpert:patient00001" not in set(pairs["patient_id"])
    assert all(Path(p).exists() for p in pairs["clean_path"]) and all(Path(p).exists() for p in pairs["photo_path"])
    v = [label_vector(r, CHEXBERT_LABELS) for r in pairs.to_dict("records")]
    eff = CHEXBERT_LABELS.index("Pleural Effusion")
    assert sorted(x[eff] for x in v) == [0, 1]       # uncertain -> 1 (rrg convention), negative -> 0
    assert clean_tail("train/natural/iphone/patient9/study2/view1_frontal.jpg") == "train/patient9/study2/view1_frontal.jpg"


def test_micro_f1():
    from step6_score import micro_f1

    y = np.array([[1, 0, 1], [0, 1, 0]], bool)
    assert micro_f1(y, y) == 1.0 and micro_f1(y, ~y) == 0.0
    assert micro_f1(y, np.array([[1, 0, 0], [0, 0, 0]], bool)) == pytest.approx(2 / 4)


# ------------------------------------------------------------------ Step 7
def test_translation_plumbing_and_agreement():
    from src.multilingual.translate import LANGS, label_agreement, translate_text

    assert LANGS["hindi"] == "hin_Deva" and LANGS["tamil"] == "tam_Taml" and LANGS["telugu"] == "tel_Telu"
    seen = []

    def fake(sents, src, tgt):
        seen.append((list(sents), src, tgt))
        return [f"<{tgt}>{s}" for s in sents]

    out = translate_text(fake, "Your lungs look clear. There is fluid on the left side.", "eng_Latn", "hin_Deva")
    assert out == "<hin_Deva>Your lungs look clear. <hin_Deva>There is fluid on the left side."
    assert seen[0][0] == ["Your lungs look clear.", "There is fluid on the left side."]
    assert translate_text(fake, "", "eng_Latn", "hin_Deva") == ""
    ref = [0] * 14
    ref[CHEXBERT_LABELS.index("Pleural Effusion")] = 1
    same = label_agreement(ref, list(ref))
    lost = label_agreement(ref, [0] * 14)
    assert same["exact"] and not lost["exact"] and lost["lost"] == ["Pleural Effusion"] and lost["added"] == []


# ------------------------------------------------------------------ Step 8
def test_second_reader_logic():
    from src.verify.second_reader import CIDX, calibrate, flag, mentioned, simulate_deletion

    ptx, eff = CIDX["Pneumothorax"], CIDX["Pleural Effusion"]
    cls = np.zeros((3, 14), np.int8)
    cls[0, eff] = 1        # "Small left effusion."
    cls[1, ptx] = 2        # "No pneumothorax."   (negative: NOT a mention)
    cls[2, :] = 0          # "Heart size normal."
    assert mentioned(cls) == {"Pleural Effusion"}
    sim = simulate_deletion(["Small left effusion.", "No pneumothorax.", "Heart size normal."], cls, random.Random(0))
    assert sim["deleted"] == "Pleural Effusion" and sim["kept_idx"] == [1, 2]
    assert sim["text"] == "No pneumothorax. Heart size normal."
    assert simulate_deletion(["Heart normal."], np.zeros((1, 14), np.int8), random.Random(0)) is None
    rng = np.random.default_rng(0)
    probs = rng.random((500, 14))
    ms = [set() for _ in range(500)]
    tau = calibrate(probs, ms, 0.02)
    assert abs(float((probs[:, eff] >= tau["Pleural Effusion"]).mean()) - 0.02) < 0.01
    row = np.zeros(14)
    row[eff] = 0.999
    row[ptx] = 0.999
    fl = flag(row, {"Pneumothorax"}, tau)                  # pneumothorax already in the report -> not flagged
    assert [f["finding"] for f in fl] == ["Pleural Effusion"]
    assert calibrate(probs[:5], ms[:5], 0.02)["Pleural Effusion"] == float("inf")   # too few absent studies
