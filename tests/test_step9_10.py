"""Steps 9-10 tests: reward components, GRPO advantages/loss, the real PolicyB sample -> logprob -> update cycle on a
tiny Model B with LoRA (a rewarded completion becomes more likely, the reference does not move), the paired
CheXbert micro-F1 metric, HTML/box rendering, PDF export (skipped without reportlab) and the app pipeline's
orchestration with stub components."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE / "scripts"))
sys.path.insert(0, str(CODE / "tests"))

from src.retrieval.rerank import CHEXBERT_LABELS  # noqa: E402
from src.rl.rewards import (chexbert_sample_f1, combine, format_reward, length_penalty,  # noqa: E402
                            sentence_consistency)

torch = pytest.importorskip("torch")
from src.rl.grpo import group_advantages, grpo_loss, token_logprobs  # noqa: E402

E, P = CHEXBERT_LABELS.index("Pleural Effusion"), CHEXBERT_LABELS.index("Pneumothorax")


# ------------------------------------------------------------------ rewards
def test_format_and_length():
    assert format_reward("FINDINGS: Clear lungs. IMPRESSION: Normal.") == 1.0
    assert format_reward("FINDINGS: A. A. IMPRESSION: B.") == 0.5
    assert format_reward("just text") == 0.0
    assert length_penalty("a " * 100, 120, 1.0) == 0.0 and length_penalty("a " * 240, 120, 1.0) == pytest.approx(-1.0)


def test_chexbert_sample_f1():
    h = np.zeros((3, 14), int)
    r = np.zeros((3, 14), int)
    h[0, E] = r[0, E] = 1            # match
    h[1, E] = 1                      # hallucinated
    assert chexbert_sample_f1(h, r) == [1.0, 0.0, 1.0]   # row 3: both empty -> 1


def test_sentence_consistency():
    thr = np.full(14, 0.5)
    p = np.full(14, 0.1)
    p[E] = 0.95                                          # probe: effusion present, nothing else
    cls = np.zeros((3, 14), np.int8)
    cls[0, E] = 1                                        # "Small effusion."      -> supported (+1)
    cls[1, P] = 2                                        # "No pneumothorax."     -> supported (+1)
    cls[2, :] = 0                                        # no claim -> ignored
    assert sentence_consistency(cls, p, thr, 0.5) == 1.0
    bad = np.zeros((2, 14), np.int8)
    bad[0, E] = 2                                        # "No effusion." vs strong positive -> -1
    bad[1, P] = 1                                        # "Pneumothorax." vs strong negative -> -1
    assert sentence_consistency(bad, p, thr, 0.5) == -1.0
    assert sentence_consistency(np.zeros((0, 14), np.int8), p, thr, 0.5) == 0.0
    assert combine({"a": [1.0, None], "b": [0.5, 1.0]}, {"a": 2.0, "b": 1.0}) == [2.5, 1.0]


# ------------------------------------------------------------------ GRPO math
def test_group_advantages():
    a = group_advantages([1, 0, 0, 0, 2, 2, 2, 2], 4)
    assert a[0] > 0 and all(x < 0 for x in a[1:4]) and a[4:] == [0.0] * 4    # equal group -> no signal
    assert abs(sum(a[:4])) < 1e-6


def test_grpo_loss_direction():
    lp = torch.tensor([-1.0, -2.0], requires_grad=True)
    loss, kl = grpo_loss([lp], [lp.detach()], [1.0], beta=0.1)
    loss.backward()
    assert kl == pytest.approx(0.0) and (lp.grad < 0).all()   # positive advantage -> push log-probs UP
    logits = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    assert torch.allclose(token_logprobs(logits, torch.tensor([0, 1])), torch.log_softmax(logits, -1)[[0, 1], [0, 1]])


def _tiny_policy_b():
    peft = pytest.importorskip("peft")
    from test_step4 import TinyTok, _tiny_model

    from src.models.model_b import ModelB, end_of_turn_id, prompt_pieces
    from src.rl.policy import PolicyB

    _, vis, llm, proj = _tiny_model()
    llm = peft.get_peft_model(llm, peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                                                   target_modules=["q_proj", "v_proj", "down_proj"]))
    pol = object.__new__(PolicyB)          # bypass the from-disk loader; everything else is the real class
    tok = TinyTok()
    pol.llm, pol.proj, pol.tok, pol.device, pol.use_prior = llm, proj, tok, "cpu", False
    pol.model = ModelB(vis, llm, proj, torch.float32)
    for w in ["FINDINGS:", "clear", "lungs.", "IMPRESSION:", "normal."]:
        tok._id(w)
    pol.stops = {end_of_turn_id(tok), tok.eos_token_id}
    pre, post = prompt_pieces(tok, "Indication: x")
    px = torch.randn(1, 3, 56, 56)
    pol._inputs = lambda rows, g: {"px": px.repeat(len(rows), 1, 1, 1), "pre": [pre] * len(rows),
                                   "post": [post] * len(rows)}
    return pol, tok


def test_policy_b_sample_logprob_and_grpo_update():
    pol, tok = _tiny_policy_b()
    g = {"prompt_template": "", "empty_indication": ""}
    torch.manual_seed(0)
    comps, trunc, texts, b = pol.sample([{"image_path": "x"}], g, G=3, max_new=6, temperature=1.0)
    assert len(comps) == 3 and len(texts) == 3 and all(len(c) <= 6 for c in comps)
    lp = pol.logprobs(b, comps, 3)
    ref = pol.ref_logprobs(b, comps, 3)
    assert [x.shape[0] for x in lp] == [len(c) for c in comps]
    assert all(torch.allclose(a, r, atol=1e-5) for a, r in zip(lp, ref))   # LoRA B=0 at init -> policy == SFT
    target = tok(" ".join(["FINDINGS:", "clear", "lungs."]))["input_ids"] + [3]
    params = pol.llm.parameters()
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=5e-2)
    pol.llm.train()
    before = float(pol.logprobs(b, [target], 1)[0].sum().detach())
    for _ in range(15):     # the target completion gets advantage +1 vs a 'bad' one with -1
        lp2 = pol.logprobs(b, [target, [5, 5, 3]], 2)
        rf2 = pol.ref_logprobs(b, [target, [5, 5, 3]], 2)
        loss, _ = grpo_loss(lp2, rf2, group_advantages([1.0, 0.0], 2), beta=0.01)
        opt.zero_grad()
        loss.backward()
        opt.step()
    after = float(pol.logprobs(b, [target], 1)[0].sum().detach())
    ref_after = float(pol.ref_logprobs(b, [target], 1)[0].sum())
    assert after > before + 0.5
    assert ref_after == pytest.approx(before, abs=1e-4)                    # reference (adapter off) is untouched


def test_cached_micro_f1_matches_direct():
    from step9_score import cached_micro_f1

    rng = np.random.default_rng(0)
    R = rng.integers(0, 2, (20, 14))
    A, B = rng.integers(0, 2, (20, 14)), rng.integers(0, 2, (20, 14))
    ha, hb = [f"a{i}" for i in range(20)], [f"b{i}" for i in range(20)]
    fn = cached_micro_f1(R, {"x": A, "y": B}, {"x": ha, "y": hb})
    mixed = [ha[i] if i % 2 else hb[i] for i in range(20)]
    M = np.stack([A[i] if i % 2 else B[i] for i in range(20)]).astype(bool)
    direct = 2 * (M & R.astype(bool)).sum() / (M.sum() + R.sum())
    assert fn(mixed, []) == pytest.approx(direct)


# ------------------------------------------------------------------ app
def test_render_html_and_boxes():
    from src.app.render import draw_boxes, sentences_html

    h = sentences_html([{"section": "findings", "text": "Left <effusion>.", "verdict": "FLAGGED",
                         "claims": [{"reason": "box only on the right"}]},
                        {"section": "impression", "text": "Normal.", "verdict": "VERIFIED", "claims": []}])
    assert "&lt;effusion&gt;" in h and "#B71C1C" in h and "box only on the right" in h and "NOT a diagnosis" in h
    im = Image.new("L", (100, 80))
    out = draw_boxes(im, [{"finding": "Pleural Effusion", "box": [0.6, 0.5, 0.9, 0.9], "verdict": "FLAGGED"}])
    assert out.mode == "RGB" and np.asarray(out).sum() > 0


def test_pdf_export(tmp_path):
    pytest.importorskip("reportlab")
    from src.app.pdf import build_pdf

    res = {"system": "model-a@step1/a1", "indication": "Cough.",
           "sentences": [{"section": "findings", "text": "Small left effusion.", "verdict": "VERIFIED"},
                         {"section": "impression", "text": "Effusion.", "verdict": "UNCERTAIN"}],
           "cases": [{"study_id": "s1", "sim": 0.9, "agreement": 1.0, "snippet": "FINDINGS: effusion."}]}
    p = build_pdf(res, Image.new("RGB", (200, 160)), str(tmp_path / "r.pdf"), "Title", "Sub",
                  summary={"language": "hindi", "summary_en": "Some fluid.", "translation": "x", "faithful": False,
                           "lost": ["Pleural Effusion"], "added": []})
    data = Path(p).read_bytes()
    assert data.startswith(b"%PDF") and len(data) > 1500


def test_pipeline_report_orchestration(monkeypatch, tmp_path):
    from src.app import pipeline as PL
    from src.retrieval.index import FlatIP
    from src.verify import claims as C

    class FakeSys:
        name = "fake@test"

        def generate(self, rows, bs):
            return ["FINDINGS: Small left effusion. No pneumothorax.\nIMPRESSION: Effusion."]

    def fake_classes(texts, scorer=None, device=None):
        out = np.zeros((len(texts), 14), np.int8)
        for k, t in enumerate(texts):
            if "effusion" in t.lower():
                out[k, E] = 1
            if "pneumothorax" in t.lower():
                out[k, P] = 2
        return out

    monkeypatch.setattr(C, "chexbert_classes", fake_classes)
    pipe = object.__new__(PL.Pipeline)
    pipe.cfg = {"verify": {"strong_margin": 0.5, "side_center_band": 0.05},
                "retrieval": {"top_k": 5, "n_refs": 2, "report_text": "findings+impression"}}
    pipe.ac = {"boxes": {"enabled": False}}
    pipe._sys, pipe._cx = FakeSys(), object()
    probs = np.full(14, 0.1)
    probs[E] = 0.9
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(1, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb)
    db = rng.normal(size=(6, 8)).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    import pandas as pd

    lab = np.zeros((6, 14), np.int8)
    lab[:3, E] = 1
    pipe._index = (FlatIP(db), pd.DataFrame({"study_id": [f"t{i}" for i in range(6)],
                                             "patient_id": [f"p{i}" for i in range(6)]}), lab,
                   [f"FINDINGS: case {i}." for i in range(6)])
    pipe._probe = (None, {"thresholds": [0.5] * 14})
    pipe.probe_probs = lambda path: (probs, emb)
    r = pipe.report(Image.new("L", (64, 64)), "Cough.")
    v = {s["text"]: s["verdict"] for s in r["sentences"]}
    assert v["No pneumothorax."] == "VERIFIED"                      # probe agrees it is absent
    assert v["Small left effusion."] == "UNCERTAIN"                 # probe positive but no box (boxes disabled)
    assert len(r["cases"]) == 2 and all(c["agreement"] >= 0 for c in r["cases"])
    assert r["cases"][0]["agreement"] >= r["cases"][1]["agreement"]
