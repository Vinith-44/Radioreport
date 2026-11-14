"""Step 4 tests: template reports, CheXpert/NIH loaders on fake files, the 2x2 patch merge, embedding assembly,
and a TINY randomly initialised Model B (real transformers SigLIP + Gemma 3 classes) trained on CPU: loss falls,
only the projector learns, generate() returns new tokens only."""
from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from src.data.synthetic import PHRASES, load_chexpert, load_nih, template_report  # noqa: E402
from src.data.text_clean import parse_generated  # noqa: E402

torch = pytest.importorskip("torch")
from src.models.model_b import IMG, ModelB, Projector, assemble, prompt_pieces, target_ids  # noqa: E402


# ------------------------------------------------------------------ templates
def test_template_never_contradicts_labels():
    rng = random.Random(0)
    for _ in range(200):
        fs = list(PHRASES)
        pos = rng.sample(fs, rng.randint(0, 3))
        unc = rng.sample([f for f in fs if f not in pos], rng.randint(0, 2))
        neg = [f for f in fs if f not in pos and f not in unc and rng.random() < 0.3]
        rep = template_report(pos, unc, neg, rng)
        p = parse_generated(rep)
        assert p["parse_ok"] and p["findings"] and p["impression"]
        for f in pos + unc:
            neg_s = PHRASES[f][2]
            assert not neg_s or neg_s not in rep, (f, rep)
        if "Pleural Effusion" in pos + unc:
            assert "no pleural effusion" not in rep.lower()
        if not pos and not unc:
            assert "No acute cardiopulmonary abnormality." in rep
        assert not re.search(r"\b(left|right)\b", rep, re.I)     # labels carry no laterality


def test_template_examples():
    rng = random.Random(1)
    rep = template_report(["Cardiomegaly", "Pleural Effusion"], [], ["Pneumothorax"], rng)
    assert "IMPRESSION: Cardiomegaly and pleural effusion." in rep and "There is no pneumothorax." in rep
    nih = template_report([], [], [], rng, [("A hiatal hernia is present.", "hiatal hernia")])
    assert "A hiatal hernia is present." in nih and nih.endswith("IMPRESSION: Hiatal hernia.")


def _jpg(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (40, 32), 100).save(p)


def test_load_chexpert(tmp_path):
    rows = [
        ("CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg", "Frontal", "AP", {"Cardiomegaly": 1.0, "Pneumothorax": 0.0}),
        ("CheXpert-v1.0-small/train/patient00001/study1/view2_lateral.jpg", "Lateral", None, {}),
        ("CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg", "Frontal", "PA", {"Edema": -1.0}),
    ]
    cols = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", "No Finding"] + list(PHRASES)
    recs = []
    for path, fl, appa, labs in rows:
        r = {c: np.nan for c in cols}
        r.update({"Path": path, "Sex": "F", "Age": 50, "Frontal/Lateral": fl, "AP/PA": appa, **labs})
        recs.append(r)
        _jpg(tmp_path / path.replace("CheXpert-v1.0-small/", ""))
    pd.DataFrame(recs)[cols].to_csv(tmp_path / "train.csv", index=False)
    df = load_chexpert(tmp_path, "train", None, 42)
    assert len(df) == 2 and set(df["patient_id"]) == {"chexpert:patient00001", "chexpert:patient00002"}
    assert all(Path(p).exists() for p in df["image_path"])
    r1 = df[df["patient_id"] == "chexpert:patient00001"].iloc[0]["target"]
    assert "cardiomegaly" in r1.lower() and "There is no pneumothorax." in r1
    r2 = df[df["patient_id"] == "chexpert:patient00002"].iloc[0]["target"]
    assert "Possible pulmonary edema." in r2


def test_load_nih_uses_train_val_only(tmp_path):
    names = ["00000001_000.png", "00000002_000.png", "00000003_000.png"]
    for n in names:
        _jpg(tmp_path / "images_001" / "images" / n)
    pd.DataFrame({"Image Index": names, "Finding Labels": ["Effusion|Hernia", "No Finding", "Mass"],
                  "Follow-up #": 0, "Patient ID": [1, 2, 3], "Patient Age": 50, "Patient Gender": "M",
                  "View Position": ["PA", "AP", "PA"]}).to_csv(tmp_path / "Data_Entry_2017.csv", index=False)
    (tmp_path / "train_val_list.txt").write_text("00000001_000.png\n00000002_000.png\n")
    (tmp_path / "test_list.txt").write_text("00000003_000.png\n")
    df = load_nih(tmp_path, None, 42)
    assert len(df) == 2 and "nih:00000003_000.png" not in set(df["study_id"])
    t = df.set_index("study_id").loc["nih:00000001_000.png", "target"]
    assert "pleural effusion" in t.lower() and "hiatal hernia" in t.lower()


# ------------------------------------------------------------------ projector + assembly
def test_patch_merge_order():
    p = Projector(d_vis=1, d_llm=4, merge=2)
    x = torch.arange(16, dtype=torch.float32).view(1, 16, 1)      # 4x4 grid, value = row*4+col
    m = p.merge_patches(x)
    assert m.shape == (1, 4, 4)
    assert m[0, 0].tolist() == [0, 1, 4, 5] and m[0, 1].tolist() == [2, 3, 6, 7] and m[0, 3].tolist() == [10, 11, 14, 15]
    with pytest.raises(ValueError):
        p.merge_patches(torch.zeros(1, 9, 1))


def test_assemble_padding_and_labels():
    emb = torch.nn.Embedding(50, 3)
    img = torch.ones(2, 4, 3)
    tr = assemble(emb, img, [[1, 2], [1]], [[3], [3, 4]], [[7, 8, 9], [7]], pad_left=False)
    assert tr["inputs_embeds"].shape == (2, 10, 3)
    assert tr["labels"][0].tolist() == [-100] * 7 + [7, 8, 9]
    assert tr["labels"][1].tolist() == [-100] * 7 + [7] + [-100] * 2 and tr["attention_mask"][1].tolist() == [1] * 8 + [0, 0]
    ge = assemble(emb, img, [[1, 2], [1]], [[3], [3]], None, pad_left=True)     # lengths 7 and 6
    assert "labels" not in ge and ge["attention_mask"][1].tolist() == [0] + [1] * 6
    assert torch.equal(ge["inputs_embeds"][1, 0], torch.zeros(3))                # pad is on the left


# ------------------------------------------------------------------ tiny Model B end-to-end
class TinyTok:
    """Word-level stand-in for the Gemma tokenizer (same special-token conventions and chat template shape)."""
    def __init__(self):
        self.vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<end_of_turn>": 3, "<start_of_turn>": 4, "<unk>": 5}
        self.pad_token_id, self.eos_token_id, self.bos_token_id, self.unk_token_id = 0, 1, 2, 5
        self.inv = {v: k for k, v in self.vocab.items()}

    def _id(self, w):
        if w not in self.vocab:
            self.vocab[w] = len(self.vocab)
            self.inv[self.vocab[w]] = w
        return self.vocab[w]

    def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
        return f"<bos><start_of_turn>user\n{msgs[0]['content']}<end_of_turn>\n<start_of_turn>model\n"

    def __call__(self, s, add_special_tokens=False):
        return {"input_ids": [self._id(w) for w in re.findall(r"<[^>]+>|[^\s<]+", s)]}

    def convert_tokens_to_ids(self, t):
        return self.vocab.get(t, self.unk_token_id)

    def decode(self, ids, skip_special_tokens=False):
        ws = [self.inv.get(i, f"w{i}") for i in ids]   # a random LLM may emit any id
        if skip_special_tokens:
            ws = [w for w in ws if not (w.startswith("<") and w.endswith(">"))]
        return " ".join(ws)

    def batch_decode(self, rows, skip_special_tokens=False):
        return [self.decode(r.tolist(), skip_special_tokens) for r in rows]


def _tiny_model():
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig, SiglipVisionConfig, SiglipVisionModel

    torch.manual_seed(0)
    vis = SiglipVisionModel(SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                               num_attention_heads=2, image_size=56, patch_size=14)).eval()
    llm = Gemma3ForCausalLM(Gemma3TextConfig(vocab_size=200, hidden_size=48, intermediate_size=96, num_hidden_layers=2,
                                             num_attention_heads=2, num_key_value_heads=1, head_dim=24,
                                             sliding_window=64, max_position_embeddings=256,
                                             pad_token_id=0, eos_token_id=1, bos_token_id=2))
    for p in list(vis.parameters()) + list(llm.parameters()):
        p.requires_grad = False
    proj = Projector(32, 48, 2)
    return ModelB(vis, llm, proj, torch.float32), vis, llm, proj


def test_tiny_model_b_learns_only_projector_and_generates():
    model, vis, llm, proj = _tiny_model()
    tok = TinyTok()
    pre, post = prompt_pieces(tok, "Indication: Not provided.")
    assert tok.decode(pre) == "<bos> <start_of_turn> user"
    tgt = [target_ids(tok, "FINDINGS: clear lungs. IMPRESSION: normal."),
           target_ids(tok, "FINDINGS: big heart. IMPRESSION: cardiomegaly.")]
    assert tgt[0][-1] == 3   # <end_of_turn> supervised
    px = torch.randn(2, 3, 56, 56)
    llm_before = {k: v.clone() for k, v in llm.state_dict().items()}
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-2)
    llm.train()
    losses = []
    for _ in range(60):
        opt.zero_grad()
        out = model(px, [pre, pre], [post, post], tgt)
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.detach()))
    # a random frozen tiny LLM steered only through 4 image tokens has a loss floor (~4.7 measured); require a clear
    # decrease, and check the projector gradient against a finite difference instead of demanding a low loss
    assert losses[-1] < 0.95 * losses[0]
    assert all(p.grad is not None for p in proj.parameters())
    assert all(p.grad is None for p in llm.parameters()) and all(p.grad is None for p in vis.parameters())
    assert all(torch.equal(v, llm.state_dict()[k]) for k, v in llm_before.items())
    w = proj.fc2.bias
    opt.zero_grad()
    model(px, [pre, pre], [post, post], tgt).loss.backward()
    g, eps = float(w.grad[0]), 1e-3
    with torch.no_grad():
        w[0] += eps
        lp = float(model(px, [pre, pre], [post, post], tgt).loss)
        w[0] -= 2 * eps
        lm = float(model(px, [pre, pre], [post, post], tgt).loss)
        w[0] += eps
    assert abs(g - (lp - lm) / (2 * eps)) < 1e-3 + 0.02 * abs(g)        # gradient reaches the projector correctly
    llm.eval()
    gen = model.generate(px, [pre, pre], [post, post], tok, max_new_tokens=12)
    assert len(gen) == 2 and all(len(g.split()) <= 12 for g in gen)
    assert "user" not in gen[0] and "Indication:" not in gen[0]       # new tokens only, prompt not echoed


def test_convergence_and_coherence_rules():
    sys.path.insert(0, str(CODE / "scripts"))
    from step4_train_projector import coherence, converged

    falling = list(np.linspace(5, 1, 100))
    plateau = list(np.linspace(5, 1.2, 80)) + [1.2] * 40
    assert not converged(falling, 0.8, 0.05)["converged"]          # still falling fast at the end
    assert converged(plateau, 0.8, 0.05)["converged"]
    assert not converged([3.0] * 100, 0.8, 0.05)["converged"]      # flat but never learned
    c = coherence(["FINDINGS: Clear lungs. IMPRESSION: Normal.", "FINDINGS: A. A. IMPRESSION: B.", "rubbish"])
    assert c["format_ok"] == pytest.approx(2 / 3, abs=1e-3) and c["repeated_sentence"] == pytest.approx(1 / 3, abs=1e-3)
