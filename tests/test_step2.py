"""Step 2 tests: preprocessing parity (train-subset PNG vs full-size test PNG), FAISS == exact search, re-ranking
rules, probe training/thresholds, prompt with references, resumable embedding, and the probe + retrieve scripts
end-to-end on synthetic embeddings (no GPU, no MedSigLIP download)."""
from __future__ import annotations

import io
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

from src.data.rexgradient_fetch import shrink_png  # noqa: E402
from src.models.medgemma_zeroshot import build_prompt  # noqa: E402
from src.models.probe import auroc, best_thresholds, load_probe, predict, save_probe, train_probe  # noqa: E402
from src.retrieval.embed import embed_rows, load_for_siglip  # noqa: E402
from src.retrieval.index import FlatIP  # noqa: E402
from src.retrieval.rerank import CHEXBERT_LABELS, references_block, rerank  # noqa: E402


def _norm(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


# ------------------------------------------------------------------ preprocessing parity
def test_train_subset_and_test_images_get_identical_448_input(tmp_path):
    rng = np.random.default_rng(0)
    a = (rng.random((1100, 1300)) * 4095).astype(np.uint16)          # full-size 12-bit X-ray stand-in
    full = tmp_path / "full.png"
    Image.fromarray(a).save(full)
    small = tmp_path / "small.png"                                   # what the train-subset fetch stores
    small.write_bytes(shrink_png(full.read_bytes(), 896))
    x_full, x_small = np.asarray(load_for_siglip(str(full))), np.asarray(load_for_siglip(str(small)))
    assert x_full.shape == (448, 448, 3)
    assert np.array_equal(x_full, x_small)


# ------------------------------------------------------------------ index
def test_faiss_matches_exact_search():
    rng = np.random.default_rng(1)
    db, q = _norm(rng.normal(size=(500, 64))), _norm(rng.normal(size=(7, 64)))
    idx = FlatIP(db)
    s, i = idx.search(q, 10)
    exact = np.argsort(-(q @ db.T), axis=1)[:, :10]
    assert np.array_equal(i, exact) and np.allclose(s, np.take_along_axis(q @ db.T, exact, 1), atol=1e-5)
    with pytest.raises(ValueError):
        FlatIP(rng.normal(size=(5, 4)))


# ------------------------------------------------------------------ re-ranking
def test_rerank_rules():
    ci, cs = np.array([10, 11, 12, 13, 14]), np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    lab = np.array([[1, 0], [0, 1], [0, 1], [0, 1], [1, 1]])
    pats = ["pA", "pQ", "pB", "pC", "pD"]                   # pQ = the query's own patient
    texts = ["r10", "r11", "same text", "Same  text", "r14"]
    out = rerank(ci, cs, lab, np.array([0, 1]), pats, texts, "pQ", k=3)
    assert [o["index"] for o in out] == [12, 14, 10]        # full agreement first; 11 = own patient; 13 = duplicate
    assert out[0]["agreement"] == 1.0 and out[1]["agreement"] == 0.5
    sim = rerank(ci, cs, lab, None, pats, texts, "pQ", k=3)
    assert [o["index"] for o in sim] == [10, 12, 14]         # similarity order, still no own patient / duplicates


def test_references_block_truncates():
    b = references_block(["FINDINGS: " + "word " * 300, "FINDINGS: short."], 60)
    first, second = b.splitlines()
    assert first.startswith("[1] FINDINGS:") and first.endswith(" ...") and len(first) <= 60 + 10
    assert second == "[2] FINDINGS: short."


def test_build_prompt_with_references():
    t = "Indication: {indication}\nRefs:\n{references}\nEnd"
    assert build_prompt(t, "", "Not provided.", "[1] x") == "Indication: Not provided.\nRefs:\n[1] x\nEnd"
    with pytest.raises(ValueError):
        build_prompt(t, "c", "N", None)
    with pytest.raises(ValueError):
        build_prompt("Indication: {indication}", "c", "N", "[1] x")


# ------------------------------------------------------------------ probe
def test_auroc_matches_sklearn():
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(2)
    y, s = rng.integers(0, 2, 300), np.round(rng.random(300), 2)   # rounding -> ties
    assert abs(auroc(y, s) - sk.roc_auc_score(y, s)) < 1e-9
    assert np.isnan(auroc(np.zeros(5), np.arange(5)))


def test_probe_learns_and_roundtrips(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    w = rng.normal(size=(32, 14))
    xtr, xva = _norm(rng.normal(size=(2000, 32))), _norm(rng.normal(size=(400, 32)))
    ytr, yva = ((xtr @ w) > 0.3).astype(np.int8), ((xva @ w) > 0.3).astype(np.int8)
    model, info = train_probe(xtr, ytr, xva, yva, hidden=0, epochs=15, lr=1e-2, log=lambda *_: None)
    assert info["macro_auroc"] > 0.95 and len(info["thresholds"]) == 14
    save_probe(model, info, {"d_in": 32, "n_out": 14, "hidden": 0, "dropout": 0.0}, tmp_path / "probe")
    m2, info2 = load_probe(tmp_path / "probe")
    p1, b1 = predict(model, xva, info["thresholds"])
    p2, b2 = predict(m2, xva, info2["thresholds"])
    assert np.allclose(p1, p2) and np.array_equal(b1, b2) and set(np.unique(b1)) <= {0, 1}


def test_best_thresholds_maximise_f1():
    y = np.array([[1], [1], [0], [0]])
    p = np.array([[0.9], [0.6], [0.55], [0.1]])
    assert best_thresholds(y, p)[0] == pytest.approx(0.6)


# ------------------------------------------------------------------ resumable embedding
def test_embed_rows_resumes(tmp_path):
    pytest.importorskip("torch")
    paths = []
    for i in range(5):
        p = tmp_path / f"i{i}.png"
        Image.fromarray(np.full((40, 30), 10 * i, np.uint8)).save(p)
        paths.append(str(p))
    calls = []

    def fake(imgs):
        calls.append(len(imgs))
        return _norm(np.array([[float(np.asarray(im).mean()) + 1, 1.0] for im in imgs]))

    e1 = embed_rows(paths, fake, tmp_path / "parts", "train", chunk=2, batch_size=2, log=lambda *_: None)
    n1 = sum(calls)
    e2 = embed_rows(paths, fake, tmp_path / "parts", "train", chunk=2, batch_size=2, log=lambda *_: None)
    assert e1.shape == (5, 2) and np.array_equal(e1, e2) and n1 == 5 and sum(calls) == 5   # 2nd run: all cached


# ------------------------------------------------------------------ probe + retrieve scripts end-to-end
def _write_run(root: Path):
    """Synthetic Step 2 run: parquet splits + eval ids + embeddings/labels (as step2_embed/label would write)."""
    rng = np.random.default_rng(3)
    ddir = root / "data"
    ddir.mkdir(parents=True)
    rdir = root / "outputs/runs/step2/r1"
    w = rng.normal(size=(16, 14))
    for split, n in [("train", 400), ("val", 120), ("test", 30)]:
        df = pd.DataFrame({"study_id": [f"{split}{i}" for i in range(n)],
                           "patient_id": [f"rexgradient:{split}P{i // 2}" for i in range(n)],
                           "dataset": "rexgradient", "split": split, "image_path": [f"/x/{split}{i}.png" for i in range(n)],
                           "findings": [f"Finding text {i % 37}." for i in range(n)], "impression": ""})
        df.to_parquet(ddir / f"{split}.parquet")
        x = _norm(rng.normal(size=(n, 16)))
        (rdir / "embeddings").mkdir(parents=True, exist_ok=True)
        (rdir / "labels").mkdir(parents=True, exist_ok=True)
        np.save(rdir / "embeddings" / f"{split}.npy", x)
        keep = ["study_id", "patient_id", "dataset", "image_path"] + ([] if split == "test" else ["findings", "impression"])
        df[keep].to_parquet(rdir / "embeddings" / f"{split}_ids.parquet", index=False)
        if split != "test":
            from src.retrieval.store import ids_sha

            np.save(rdir / "labels" / f"{split}.npy", ((x @ w) > 0.2).astype(np.int8))
            (rdir / "labels" / f"{split}.json").write_text(json.dumps({"ids_sha256": ids_sha(df)}))
    (ddir / "step0_eval_ids.json").write_text(json.dumps({"rexgradient": [f"test{i}" for i in range(30)]}))
    return rdir


def test_probe_and_retrieve_scripts(tmp_path):
    pytest.importorskip("torch")
    rdir = _write_run(tmp_path)
    env = {**os.environ, "PROJECT_ROOT": str(tmp_path), "PYTHONIOENCODING": "utf-8"}
    for script in ["step2_train_probe.py", "step2_retrieve.py"]:
        r = subprocess.run([sys.executable, f"scripts/{script}", "--run-name", "r1"],
                           cwd=CODE, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-3000:]
    rows = [json.loads(line) for line in open(rdir / "retrieval.jsonl", encoding="utf-8")]
    assert len(rows) == 30
    tr = pd.read_parquet(rdir / "embeddings/train_ids.parquet").set_index("study_id")
    for row in rows:
        assert len(row["chosen"]) == 3 and row["references"].count("\n") == 2
        assert all(c["study_id"].startswith("train") for c in row["chosen"])        # index = train only
        texts = [" ".join(tr.loc[c["study_id"], "findings"].split()) for c in row["chosen"]]
        assert len(set(texts)) == 3                                                  # deduplicated
        ag = [c["agreement"] for c in row["chosen"]]
        assert ag == sorted(ag, reverse=True)                                        # re-ranked by agreement
    summ = json.loads((rdir / "retrieval_summary_rerank.json").read_text())
    assert summ["agree_chosen"] >= summ["agree_top_sim"]     # re-ranking can only raise label agreement
    info = json.loads((rdir / "probe/probe.json").read_text())
    assert info["labels"] == CHEXBERT_LABELS and info["n_train"] == 400
