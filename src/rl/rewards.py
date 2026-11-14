"""Step 9 GRPO rewards (CLAUDE.md §6 Step 9): weighted sum of
  (a) clinical accuracy vs the reference: RadGraph partial F1 (RadEval radeval.rewards.make_reward_fn("radgraph",
      key="radgraph_partial"), commit d412dc2) and per-sample CheXbert F1 (RadEval labeler, rrg labels;
      F1 = 2TP / (|hyp| + |ref|), 1.0 when both are empty),
  (b) verification consistency (Step 3 rules WITHOUT box evidence - a MedGemma box query per sampled sentence is
      far too slow inside RL): + per sentence whose claims the probe supports, - per sentence it contradicts,
      normalised by the number of sentences with a claim,
  (c) format: FINDINGS and IMPRESSION both present; minus a penalty if any sentence is repeated,
  (d) a small length penalty beyond a target number of words.
All components are computed on the hypothesis text only (the reference is used by (a) alone).
"""
from __future__ import annotations

import numpy as np

from src.data.text_clean import parse_generated
from src.verify.claims import NEG, POS, UNC, split_sentences


def chexbert_sample_f1(h: np.ndarray, r: np.ndarray) -> list[float]:
    """Row-wise F1 of binary label vectors (n, 14)."""
    out = []
    for a, b in zip(h.astype(bool), r.astype(bool)):
        den = int(a.sum() + b.sum())
        out.append(1.0 if den == 0 else 2.0 * float((a & b).sum()) / den)
    return out


def format_reward(text: str, repeat_penalty: float = 0.5) -> float:
    ok = 1.0 if parse_generated(text)["parse_ok"] else 0.0
    ss = [" ".join(s.lower().split()) for s in split_sentences(text.replace("FINDINGS:", "").replace("IMPRESSION:", ""))]
    return ok - (repeat_penalty if len(ss) != len(set(ss)) else 0.0)


def length_penalty(text: str, target_words: int, weight: float) -> float:
    n = len(text.split())
    return -weight * max(0, n - target_words) / max(1, target_words)


def sentence_consistency(sent_classes: np.ndarray, probe_p: np.ndarray, thresholds: np.ndarray, margin: float) -> float:
    """sent_classes: (n_sentences, 14) raw CheXbert classes of the hypothesis; probe_p/thresholds: (14,).
    A sentence scores +1 if every claim is supported (positive claim & probe positive, negative claim & probe
    negative), -1 if any claim is contradicted (positive claim & probe strongly negative, negative claim & probe
    strongly positive), 0 otherwise (incl. uncertain claims). Mean over sentences that make a claim; 0 if none."""
    strong_pos = probe_p >= thresholds + margin * (1 - thresholds)
    strong_neg = probe_p <= thresholds * (1 - margin)
    pos = probe_p >= thresholds
    scores = []
    for row in sent_classes:
        idx = [i for i, c in enumerate(row) if c != 0]
        if not idx:
            continue
        contra = any((row[i] == POS and strong_neg[i]) or (row[i] == NEG and strong_pos[i]) for i in idx)
        support = all((row[i] == POS and pos[i]) or (row[i] == NEG and not pos[i]) for i in idx if row[i] != UNC) \
            and any(row[i] != UNC for i in idx)
        scores.append(-1.0 if contra else 1.0 if support else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def combine(parts: dict[str, list[float]], weights: dict[str, float]) -> list[float]:
    keys = [k for k in weights if k in parts]
    n = len(parts[keys[0]])
    return [float(sum(weights[k] * (parts[k][i] if parts[k][i] is not None else 0.0) for k in keys)) for i in range(n)]


class RewardModel:
    """Loads RadGraph + CheXbert once; scores a batch of (hypothesis, reference, probe probabilities)."""

    def __init__(self, cfg: dict, thresholds: np.ndarray, device: str = "cuda"):
        from radeval.metrics.f1chexbert import F1CheXbert
        from radeval.rewards import make_reward_fn

        self.w, self.rc = cfg["weights"], cfg
        self.thr = np.asarray(thresholds, dtype=np.float32)
        self.radgraph = make_reward_fn("radgraph", key="radgraph_partial", reference_column="ground_truth")
        self.chexbert = F1CheXbert(device=device, batch_size=64)

    def __call__(self, hyps: list[str], refs: list[str], probe_p: np.ndarray) -> tuple[list[float], dict]:
        from src.retrieval.rerank import chexbert_labels
        from src.verify.claims import chexbert_classes

        h_find = [parse_generated(h)["findings"] or h for h in hyps]
        rg = self.radgraph(completions=h_find, ground_truth=refs)
        lab = chexbert_labels(h_find + refs)                           # binary rrg labels
        cx = chexbert_sample_f1(lab[: len(hyps)], lab[len(hyps):])
        sents = [split_sentences(h) for h in hyps]
        flat = [s for ss in sents for s in ss]
        cls = chexbert_classes(flat, self.chexbert) if flat else np.zeros((0, 14), np.int8)
        ver, k = [], 0
        for i, ss in enumerate(sents):
            ver.append(sentence_consistency(cls[k:k + len(ss)], probe_p[i], self.thr, float(self.rc["strong_margin"])))
            k += len(ss)
        parts = {"clinical": [((r if r is not None else 0.0) + c) / 2 for r, c in zip(rg, cx)],
                 "verify": ver, "format": [format_reward(h) for h in hyps],
                 "length": [length_penalty(h, int(self.rc["target_words"]), 1.0) for h in hyps]}
        return combine(parts, self.w), {**parts, "radgraph": rg, "chexbert_f1": cx}
