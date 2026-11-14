"""Fact-aware re-ranking of retrieved cases + the prompt block that carries them.

CheXbert labels: RadEval's labeler (radeval.metrics.f1chexbert.F1CheXbert, commit d412dc2), get_labels(mode="rrg"):
14 binary labels in TARGET_NAMES order (13 conditions + "No Finding"); "positive" and "uncertain" both -> 1.
That is exactly the labelling RadEval's CheXbert F1 uses, so re-ranking optimises the same notion of agreement
that the evaluation measures.
"""
from __future__ import annotations

import numpy as np

CHEXBERT_LABELS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
    "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture",
    "Support Devices", "No Finding",
]


def chexbert_labels(texts: list[str], device: str = "cuda", batch_size: int = 64) -> np.ndarray:
    from radeval.metrics.f1chexbert import F1CheXbert

    scorer = F1CheXbert(device=device, batch_size=batch_size)
    if list(scorer.TARGET_NAMES) != CHEXBERT_LABELS:
        raise AssertionError(f"RadEval CheXbert label order changed: {scorer.TARGET_NAMES}")
    lab = np.asarray(scorer.get_labels([t if t and t.strip() else "." for t in texts], mode="rrg"), dtype=np.int8)
    assert lab.shape == (len(texts), len(CHEXBERT_LABELS)) and set(np.unique(lab)) <= {0, 1}
    return lab


def rerank(cand_idx: np.ndarray, cand_sim: np.ndarray, cand_labels: np.ndarray, probe_bin: np.ndarray | None,
           cand_patients: list[str], cand_texts: list[str], query_patient: str, k: int = 3,
           dedupe: bool = True) -> list[dict]:
    """Order candidates by CheXbert-label agreement with the probe's prediction for the query image
    (fraction of the 14 labels that match), ties broken by image similarity. Never returns a study of the
    query's own patient; with dedupe, never returns two identical report texts.
    Per-candidate inputs (cand_labels, cand_patients, cand_texts) are aligned with cand_idx (search order).
    probe_bin=None -> plain similarity order (the "similarity-only" ablation)."""
    rows = []
    for j, (i, s) in enumerate(zip(cand_idx.tolist(), cand_sim.tolist())):
        if i < 0 or cand_patients[j] == query_patient:
            continue
        agree = 0.0 if probe_bin is None else float((cand_labels[j] == probe_bin).mean())
        rows.append({"index": int(i), "sim": float(s), "agreement": agree, "rank_in_search": j})
    rows.sort(key=lambda r: (-r["agreement"], -r["sim"], r["rank_in_search"]))
    out, seen = [], set()
    for r in rows:
        txt = " ".join(cand_texts[r["rank_in_search"]].split()).lower()
        if dedupe and txt in seen:
            continue
        seen.add(txt)
        out.append(r)
        if len(out) == k:
            break
    return out


def references_block(reports: list[str], max_chars: int) -> str:
    """'[1] FINDINGS: ...' lines; each report cut at a word boundary to max_chars."""
    lines = []
    for n, rep in enumerate(reports, 1):
        t = " ".join(rep.split())
        if len(t) > max_chars:
            t = t[:max_chars].rsplit(" ", 1)[0] + " ..."
        lines.append(f"[{n}] {t}")
    return "\n".join(lines)
