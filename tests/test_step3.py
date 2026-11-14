"""Step 3 tests: sentence/side/zone parsing, raw CheXbert classes, MedGemma box parsing + coordinate mapping,
verdict rules, correctness vs reference, NIH CSV parsing, and step3_verify end-to-end on synthetic inputs."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "tests"))

from src.retrieval.rerank import CHEXBERT_LABELS  # noqa: E402
from src.verify.boxes import (iou, localize_prompt, pad_to_square, parse_boxes, patient_side,  # noqa: E402
                              to_image_box)
from src.verify.claims import chexbert_classes, claims_of, finding_sides, laterality, split_sentences, zone  # noqa: E402
from src.verify.verdict import (claim_correct, claim_verdict, drop_flagged, probe_state, sentence_verdict,  # noqa: E402
                                side_match)


# ------------------------------------------------------------------ text
def test_split_sentences():
    t = "Heart size is normal. A 2.5 cm nodule in the left lung.\nNo effusion. e.g. this stays"
    assert split_sentences(t) == ["Heart size is normal.", "A 2.5 cm nodule in the left lung.",
                                  "No effusion. e.g. this stays"]
    assert split_sentences("") == [] and split_sentences("  . ") == []


@pytest.mark.parametrize("s,side", [
    ("Small left pleural effusion.", "left"), ("Right-sided pneumothorax.", "right"),
    ("Bilateral lower lobe opacities.", "bilateral"), ("Opacities in both lungs.", "bilateral"),
    ("Left and right basilar atelectasis.", "bilateral"), ("No acute disease.", None),
    ("Cleft palate unrelated.", None), ("Effusion at the L base.", "left"),
])
def test_laterality(s, side):
    assert laterality(s) == side


def test_zone():
    assert zone("Right upper lobe opacity.") == "upper" and zone("Left basilar atelectasis.") == "lower"
    assert zone("Patchy opacity.") is None and zone("Upper and lower opacities.") is None


class _FakeScorer:
    """Mimics RadEval's F1CheXbert: tokenizer + model.cls_logits returning 13 four-way heads + 1 two-way head."""
    TARGET_NAMES = CHEXBERT_LABELS
    batch_size = 2
    device = "cpu"

    def __init__(self, table):
        import torch

        self.table, self.torch = table, torch
        outer = self

        class Tok:
            def __call__(self, texts, **kw):
                ids = torch.tensor([[outer._key(t)] for t in texts])
                return type("E", (), {"input_ids": ids, "attention_mask": torch.ones_like(ids)})()

        class Model:
            def cls_logits(self, input_ids, attention_mask=None):
                heads = []
                for h in range(14):
                    n = 4 if h < 13 else 2
                    z = torch.zeros(len(input_ids), n)
                    for b, k in enumerate(input_ids[:, 0].tolist()):
                        z[b, outer.table[k][h]] = 1.0
                    heads.append(z)
                return heads

        self.tokenizer, self.model = Tok(), Model()
        self.keys = {}

    def _key(self, t):
        return self.keys.setdefault(t, len(self.keys))


def test_chexbert_raw_classes_keep_negative():
    pytest.importorskip("torch")
    eff = CHEXBERT_LABELS.index("Pleural Effusion")
    rows = {0: [0] * 14, 1: [0] * 14, 2: [0] * 14}
    rows[0][eff] = 1          # positive
    rows[1][eff] = 2          # NEGATIVE (RadEval's get_labels would turn this into 0)
    rows[2][13] = 1           # No Finding
    sc = _FakeScorer(rows)
    out = chexbert_classes(["Left effusion.", "No effusion.", "Normal chest."], sc)
    assert out[0, eff] == 1 and out[1, eff] == 2 and out[2, 13] == 1
    c = claims_of("Small left effusion.", out[0])
    assert c == [{"finding": "Pleural Effusion", "polarity": "positive", "laterality": "left", "zone": None}]
    assert claims_of("No effusion.", out[1])[0]["polarity"] == "negative"
    assert finding_sides(["Small left effusion.", "No effusion."], out[:2]) == {"Pleural Effusion": {"left"}}


# ------------------------------------------------------------------ boxes
def test_prompt_is_official_template():
    p = localize_prompt("pleural effusion")
    assert "Where is the pleural effusion?" in p and '[{"box_2d": [800, 3, 840, 471], "label": "car"}' in p
    assert "{" in p and "{{" not in p


def test_parse_boxes_official_answer_with_thinking():
    ans = ('thought\nThe effusion blunts the angle.<unused95>Reasoning: fluid at the left base.\n'
           'Final Answer: ```json\n[{"box_2d": [600, 550, 900, 950], "label": "pleural effusion"}]\n```')
    boxes, st = parse_boxes(ans)
    assert st == "ok" and boxes == [{"box_2d": [600.0, 550.0, 900.0, 950.0], "label": "pleural effusion"}]


@pytest.mark.parametrize("ans,status,n", [
    ('Final Answer: [{"box_2d": [10, 20, 30, 40], "label": "x"}]', "ok", 1),       # no ``` fence
    ("Final Answer: ```json\n[]\n```", "empty", 0),                                 # model says absent
    ("I cannot find it.", "unparseable", None),
    ('```json[{"box_2d": [300, 400, 100, 200], "label": "x"}]```', "ok", 1),        # swapped corners re-ordered
    ('```json[{"box_2d": [0, 0, 1200, 50], "label": "x"}]```', "empty", 0),         # out of range dropped
])
def test_parse_boxes_variants(ans, status, n):
    boxes, st = parse_boxes(ans)
    assert st == status and (boxes is None if n is None else len(boxes) == n)
    if st == "ok" and "300, 400" in ans:
        assert boxes[0]["box_2d"] == [100.0, 200.0, 300.0, 400.0]


def test_padding_roundtrip_and_side():
    im = Image.new("L", (200, 100))                    # wide image: padded top/bottom
    sq, pad = pad_to_square(im)
    assert sq.size == (200, 200) and pad == {"size": 200, "top": 50, "left": 0, "h": 100, "w": 200}
    # box over the image's left half, full height of the original image, in padded 0-1000 coords
    b = to_image_box([250, 0, 750, 500], pad)
    assert b == pytest.approx([0.0, 0.0, 0.5, 1.0])
    assert patient_side([0.1, 0.2, 0.3, 0.6]) == "right"     # image-left = patient's right
    assert patient_side([0.7, 0.2, 0.9, 0.6]) == "left"
    assert patient_side([0.2, 0.2, 0.8, 0.6]) == "both" and patient_side([0.46, 0.2, 0.53, 0.6]) == "center"
    assert iou([0, 0, 1, 1], [0, 0, 0.5, 1]) == pytest.approx(0.5)


# ------------------------------------------------------------------ verdicts
POS, NEG = probe_state(0.9, 0.5, 0.5), probe_state(0.1, 0.5, 0.5)
WEAKPOS = probe_state(0.55, 0.5, 0.5)


def _c(f="Pleural Effusion", pol="positive", side=None, z=None):
    return {"finding": f, "polarity": pol, "laterality": side, "zone": z}


@pytest.mark.parametrize("claim,probe,box,verdict", [
    (_c(side="left"), POS, {"status": "ok", "sides": ["left"], "zones": ["lower"]}, "VERIFIED"),
    (_c(side="left"), POS, {"status": "ok", "sides": ["right"], "zones": ["lower"]}, "FLAGGED"),  # wrong side
    (_c(), NEG, {"status": "empty", "sides": [], "zones": []}, "FLAGGED"),                        # no support
    (_c(), POS, {"status": "empty", "sides": [], "zones": []}, "UNCERTAIN"),
    (_c(), NEG, {"status": "ok", "sides": ["left"], "zones": ["lower"]}, "UNCERTAIN"),
    (_c(), POS, None, "UNCERTAIN"),                                                               # box unknown
    (_c(), NEG, {"status": "unparseable", "sides": [], "zones": []}, "UNCERTAIN"),
    (_c(z="upper"), POS, {"status": "ok", "sides": ["left"], "zones": ["lower"]}, "UNCERTAIN"),   # zone mismatch
    (_c(side="bilateral"), POS, {"status": "ok", "sides": ["left"], "zones": ["lower"]}, "UNCERTAIN"),
    (_c(side="bilateral"), POS, {"status": "ok", "sides": ["left", "right"], "zones": ["lower"]}, "VERIFIED"),
    (_c(pol="negative"), NEG, None, "VERIFIED"),
    (_c(pol="negative"), POS, None, "FLAGGED"),
    (_c(pol="negative"), WEAKPOS, None, "UNCERTAIN"),
    (_c(pol="uncertain"), POS, None, "UNCERTAIN"),
    (_c("No Finding"), POS, None, "VERIFIED"),
    (_c("No Finding"), NEG, None, "FLAGGED"),
])
def test_claim_verdict_rules(claim, probe, box, verdict):
    assert claim_verdict(claim, probe, box)[0] == verdict


def test_sentence_verdict_and_side_match():
    assert sentence_verdict([]) == "NOT_CHECKED"
    assert sentence_verdict(["VERIFIED", "UNCERTAIN"]) == "UNCERTAIN"
    assert sentence_verdict(["VERIFIED", "FLAGGED", "UNCERTAIN"]) == "FLAGGED"
    assert side_match("left", ["right", "left"]) == "match" and side_match("left", ["right"]) == "contradiction"
    assert side_match(None, []) == "none"


@pytest.mark.parametrize("claim,ref,sides,ok", [
    (_c(), 1, set(), True), (_c(), 0, set(), False), (_c(), 2, set(), False), (_c(), 3, set(), None),
    (_c(side="left"), 1, {"right"}, False), (_c(side="left"), 1, {"left"}, True), (_c(side="left"), 1, set(), True),
    (_c(pol="negative"), 0, set(), True), (_c(pol="negative"), 1, set(), False), (_c(pol="uncertain"), 1, set(), None),
    (_c("No Finding"), 1, set(), True), (_c("No Finding"), 0, set(), False),
])
def test_claim_correct(claim, ref, sides, ok):
    assert claim_correct(claim, ref, sides) is ok


def test_drop_flagged():
    s = [{"text": "A.", "verdict": "VERIFIED"}, {"text": "B.", "verdict": "FLAGGED"}, {"text": "C.", "verdict": "NOT_CHECKED"}]
    assert drop_flagged(s) == "A. C."


# ------------------------------------------------------------------ NIH CSV
def test_read_nih_boxes(tmp_path):
    from scripts.step3_eval_boxes import read_nih_boxes

    p = tmp_path / "b.csv"
    p.write_text("Image Index,Finding Label,Bbox [x,y,w,h],,,\n00013118_008.png,Atelectasis,225.08,547.01,86.77,79.18,,,\n")
    df = read_nih_boxes(p)
    assert list(df.columns) == ["image", "label", "x", "y", "w", "h"] and df.iloc[0]["x"] == pytest.approx(225.08)


# ------------------------------------------------------------------ step3_verify end-to-end
def test_step3_verify_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from test_step2 import _write_run

    s2 = _write_run(tmp_path)                                   # step2 run r1: embeddings, labels, ids
    env = {**os.environ, "PROJECT_ROOT": str(tmp_path), "PYTHONIOENCODING": "utf-8"}
    for script in ["step2_train_probe.py", "step2_retrieve.py"]:
        r = subprocess.run([sys.executable, f"scripts/{script}", "--run-name", "r1"], cwd=CODE, env=env,
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-2000:]
    ids = [f"test{i}" for i in range(30)]
    v = tmp_path / "outputs/runs/step3/v1"
    (v / "source/rexgradient").mkdir(parents=True)
    with open(v / "source/rexgradient/predictions.jsonl", "w") as f:
        for i in ids:
            f.write(json.dumps({"study_id": i, "dataset": "rexgradient", "hypothesis": "FINDINGS: x\nIMPRESSION: y",
                                "hyp_findings": "Left effusion. Heart normal.", "hyp_impression": "No pneumothorax.",
                                "reference": "r", "reference_findings": "Right effusion.", "reference_impression": "",
                                "parse_ok": True, "hit_max_tokens": False, "dtype": "nf4", "model_id": "m"}) + "\n")
    (v / "verify_ids.json").write_text(json.dumps({"rexgradient": ids}))
    eff, ptx = "Pleural Effusion", "Pneumothorax"
    ref = {n: 0 for n in CHEXBERT_LABELS}
    ref[eff] = 1
    with open(v / "claims.jsonl", "w") as f, open(v / "boxes.jsonl", "w") as fb:
        for k, i in enumerate(ids):
            f.write(json.dumps({"study_id": i, "dataset": "rexgradient", "ref_classes": ref,
                                "ref_sides": {eff: ["right"]}, "sentences": [
                {"section": "findings", "text": "Left effusion.", "claims": [_c(eff, "positive", "left")]},
                {"section": "findings", "text": "Heart normal.", "claims": []},
                {"section": "impression", "text": "No pneumothorax.", "claims": [_c(ptx, "negative")]}]}) + "\n")
            # half the studies: box on the patient's RIGHT -> laterality contradiction -> FLAGGED
            side = "right" if k % 2 else "left"
            fb.write(json.dumps({"study_id": i, "dataset": "rexgradient", "finding": eff, "status": "ok",
                                 "boxes": [[0.1, 0.5, 0.3, 0.9]] if side == "right" else [[0.7, 0.5, 0.9, 0.9]],
                                 "sides": [side], "zones": ["lower"]}) + "\n")
    r = subprocess.run([sys.executable, "scripts/step3_verify.py", "--run-name", "v1"], cwd=CODE, env=env,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-2500:]
    reps = [json.loads(line) for line in open(v / "verify.jsonl")]
    assert len(reps) == 30
    for k, rep in enumerate(reps):
        s0 = rep["sentences"][0]
        assert s0["correct"] is False                         # left vs reference right -> wrong sentence
        if k % 2:
            assert s0["verdict"] == "FLAGGED" and "box only on the right" in s0["claims"][0]["reason"]
        assert rep["sentences"][1]["verdict"] == "NOT_CHECKED"
        assert set(s0["claims"][0]) >= {"probe", "box", "supporting_cases", "verdict", "reason"}
    drop = [json.loads(line) for line in open(v / "model-a+verified-drop/rexgradient/predictions.jsonl")]
    for k, d in enumerate(drop):
        if k % 2:
            assert d["hyp_findings"] == "Heart normal." and d["n_flagged_dropped"] >= 1
        assert d["hyp_impression"] in ("No pneumothorax.", "")
    ev = json.loads((v / "verify_eval.json").read_text())["all"]
    assert ev["wrong"] >= 30 and ev["wrong_caught_rate (FLAGGED or UNCERTAIN)"] is not None
