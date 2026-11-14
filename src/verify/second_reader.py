"""Step 8: second-reader mode - flag findings the image supports but a radiologist's report does not mention.

A finding f is "mentioned" when any sentence of the report has CheXbert class positive or uncertain for f
(raw classes, src/verify/claims.py). A possible miss is flagged when f is NOT mentioned and the Step 2 probe's
probability for f on the image is >= tau_f. tau_f is calibrated on the VAL split (never test): among val studies
whose report does not mention f, tau_f is the (1 - target_far) quantile of p_f, i.e. at most target_far of them
would be flagged for f. Simulation (evaluation only): delete every sentence that mentions one positive finding of a
real report; that finding must then be flagged.
"""
from __future__ import annotations

import random

import numpy as np

from src.retrieval.rerank import CHEXBERT_LABELS
from src.verify.claims import POS, UNC

CONDITIONS = [n for n in CHEXBERT_LABELS if n != "No Finding"]
CIDX = {n: CHEXBERT_LABELS.index(n) for n in CONDITIONS}


def mentioned(sent_classes: np.ndarray) -> set[str]:
    """sent_classes: (n_sentences, 14) raw CheXbert classes of one report."""
    if len(sent_classes) == 0:
        return set()
    return {n for n, i in CIDX.items() if np.isin(sent_classes[:, i], (POS, UNC)).any()}


def calibrate(probs: np.ndarray, mention_sets: list[set], target_far: float, min_absent: int = 20) -> dict[str, float]:
    """Per condition: tau = quantile_{1-target_far} of p over val studies NOT mentioning it (inf if too few)."""
    tau = {}
    for n, i in CIDX.items():
        p = np.asarray([probs[k, i] for k, m in enumerate(mention_sets) if n not in m])
        tau[n] = float(np.quantile(p, 1 - target_far)) if len(p) >= min_absent else float("inf")
    return tau


def flag(prob_row: np.ndarray, report_mentions: set[str], tau: dict[str, float]) -> list[dict]:
    return [{"finding": n, "p": round(float(prob_row[i]), 4), "tau": round(tau[n], 4)}
            for n, i in CIDX.items() if n not in report_mentions and prob_row[i] >= tau[n]]


def simulate_deletion(sentences: list[str], sent_classes: np.ndarray, rng: random.Random,
                      allowed: list[str] | None = None) -> dict | None:
    """Pick one positively reported condition, delete every sentence mentioning it (positive/uncertain).
    Returns {deleted, removed_also, text, kept_idx} or None (no positive condition / nothing would remain)."""
    pos = sorted({n for n, i in CIDX.items() if (sent_classes[:, i] == POS).any()} if len(sent_classes) else set())
    if allowed:
        pos = [n for n in pos if n in allowed]
    if not pos:
        return None
    f = rng.choice(pos)
    i = CIDX[f]
    keep = [k for k in range(len(sentences)) if sent_classes[k, i] not in (POS, UNC)]
    if not keep:
        return None
    still = mentioned(sent_classes[keep])
    before = mentioned(sent_classes)
    return {"deleted": f, "removed_also": sorted((before - still) - {f}), "kept_idx": keep,
            "text": " ".join(sentences[k] for k in keep)}
