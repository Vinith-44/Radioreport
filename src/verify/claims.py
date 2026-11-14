"""Sentence-level claims of a report: which finding, positive/negative/uncertain, which side, which zone.

Finding + polarity come from CheXbert's RAW 4-class heads (RadEval radeval.metrics.f1chexbert, commit d412dc2):
the 13 condition heads predict 0 blank, 1 positive, 2 negative, 3 uncertain; the 14th ("No Finding") head
predicts 0 blank, 1 positive. RadEval's get_labels() collapses negative and blank into 0 in both of its modes,
but verification must know that "No pneumothorax." is a NEGATIVE claim, so the raw classes are read here
(same tokenizer, same max_length=512, same model as get_labels). Laterality / zone are rule-based (regex).
"""
from __future__ import annotations

import re

import numpy as np

from src.retrieval.rerank import CHEXBERT_LABELS

BLANK, POS, NEG, UNC = 0, 1, 2, 3
POLARITY = {POS: "positive", NEG: "negative", UNC: "uncertain"}

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"(])|\n+")
_DECIMAL = re.compile(r"(\d)\.(\d)")


def split_sentences(text: str) -> list[str]:
    """Split on sentence-final punctuation followed by a capital/digit, or on newlines; decimals like 2.5 are
    kept intact. Empty pieces are dropped."""
    if not text or not text.strip():
        return []
    protected = _DECIMAL.sub(r"\1<DOT>\2", text.strip())
    parts = [p.replace("<DOT>", ".").strip() for p in _SENT_RE.split(protected)]
    return [p for p in parts if p and re.search(r"[A-Za-z]", p)]


_BILAT = re.compile(r"\b(bilateral(ly)?|both (lungs?|sides|bases|hemithora(x|ces)|lung fields|apices))\b", re.I)
_LEFT = re.compile(r"\bleft(-sided)?\b|\bL(?=\s+(lung|base|apex|pleural|upper|lower|mid))", re.I)
_RIGHT = re.compile(r"\bright(-sided)?\b|\bR(?=\s+(lung|base|apex|pleural|upper|lower|mid))", re.I)
_UPPER = re.compile(r"\b(upper|apical|apex|apices)\b", re.I)
_LOWER = re.compile(r"\b(lower|basilar|basal|bases?|base|retrocardiac|costophrenic)\b", re.I)


def laterality(sentence: str) -> str | None:
    """'left' | 'right' | 'bilateral' | None (not stated). 'left and right' counts as bilateral."""
    if _BILAT.search(sentence):
        return "bilateral"
    l, r = bool(_LEFT.search(sentence)), bool(_RIGHT.search(sentence))
    if l and r:
        return "bilateral"
    return "left" if l else "right" if r else None


def zone(sentence: str) -> str | None:
    u, lo = bool(_UPPER.search(sentence)), bool(_LOWER.search(sentence))
    return None if u == lo else ("upper" if u else "lower")


def chexbert_classes(texts: list[str], scorer=None, device: str = "cuda") -> np.ndarray:
    """Raw CheXbert classes, shape (n, 14), values in {0,1,2,3} (No Finding column in {0,1})."""
    import torch

    if scorer is None:
        from radeval.metrics.f1chexbert import F1CheXbert

        scorer = F1CheXbert(device=device, batch_size=64)
    if list(scorer.TARGET_NAMES) != CHEXBERT_LABELS:
        raise AssertionError(f"CheXbert label order changed: {scorer.TARGET_NAMES}")
    out = []
    texts = [t if t and t.strip() else "." for t in texts]
    with torch.no_grad():
        for i in range(0, len(texts), scorer.batch_size):
            enc = scorer.tokenizer(texts[i:i + scorer.batch_size], padding=True, truncation=True, max_length=512,
                                   return_tensors="pt")
            heads = scorer.model.cls_logits(enc.input_ids.to(scorer.device),
                                            attention_mask=enc.attention_mask.to(scorer.device))
            out.append(torch.stack([h.argmax(dim=1) for h in heads], dim=1).cpu().numpy())
    arr = np.concatenate(out).astype(np.int8) if out else np.zeros((0, 14), np.int8)
    assert arr.shape[1] == len(CHEXBERT_LABELS) and set(np.unique(arr)) <= {0, 1, 2, 3}
    return arr


def claims_of(sentence: str, classes_row) -> list[dict]:
    """All non-blank CheXbert findings of one sentence, with the sentence's side/zone."""
    side, z = laterality(sentence), zone(sentence)
    out = []
    for name, c in zip(CHEXBERT_LABELS, [int(x) for x in classes_row]):
        if c == BLANK:
            continue
        out.append({"finding": name, "polarity": POLARITY[c], "laterality": side, "zone": z})
    return out


def finding_sides(sentences: list[str], classes: np.ndarray) -> dict[str, set]:
    """Per finding, the sides stated in POSITIVE mentions (used for the reference report's laterality)."""
    sides: dict[str, set] = {}
    for s, row in zip(sentences, classes):
        side = laterality(s)
        for name, c in zip(CHEXBERT_LABELS, row):
            if int(c) == POS and side:
                sides.setdefault(name, set()).add(side)
    return sides
