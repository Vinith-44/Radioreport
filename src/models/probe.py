"""Probe = small linear / MLP head on frozen MedSigLIP image embeddings -> 14 CheXbert labels (CLAUDE.md §5).

Targets are CheXbert labels of the TRAIN reports (findings + impression), never CheXpert/NIH dataset labels
used as report ground truth. The epoch is chosen by macro AUROC on the VAL split and per-label decision
thresholds maximise F1 on VAL; the test split is not touched here. Shared with Step 3 (verification).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC (ties averaged); nan if only one class is present."""
    y = np.asarray(y).astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    ss = np.asarray(s)[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def best_thresholds(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per label, the probability threshold that maximises F1 on (y, p); 0.5 if the label never occurs."""
    th = np.full(y.shape[1], 0.5, dtype=np.float32)
    for c in range(y.shape[1]):
        if y[:, c].sum() == 0:
            continue
        best = -1.0
        for t in np.unique(np.round(p[:, c], 4)):
            pr = p[:, c] >= t
            tp = float((pr & (y[:, c] == 1)).sum())
            f1 = 2 * tp / max(1.0, pr.sum() + y[:, c].sum())
            if f1 > best:
                best, th[c] = f1, float(t)
    return th


def build(d_in: int, n_out: int, hidden: int, dropout: float):
    import torch.nn as nn

    if hidden <= 0:
        return nn.Linear(d_in, n_out)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_out))


def train_probe(xtr, ytr, xva, yva, hidden=512, dropout=0.1, epochs=30, lr=1e-3, weight_decay=1e-4,
                batch_size=256, seed=42, device="cpu", log=print) -> tuple[object, dict]:
    import torch

    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    xtr_t, ytr_t = torch.tensor(xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32)
    xva_t = torch.tensor(xva, dtype=torch.float32, device=device)
    model = build(xtr.shape[1], ytr.shape[1], hidden, dropout).to(device)
    # pos_weight balances rare labels; clipped so a label with 3 positives cannot dominate.
    pos = ytr_t.sum(0)
    pos_weight = ((len(ytr_t) - pos) / pos.clamp_min(1)).clamp(1, 50).to(device)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = {"macro_auroc": -1.0}
    best_state = None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr_t), generator=g)
        tot = 0.0
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            opt.zero_grad()
            loss = lossf(model(xtr_t[b].to(device)), ytr_t[b].to(device))
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        model.eval()
        with torch.no_grad():
            pva = torch.sigmoid(model(xva_t)).cpu().numpy()
        aucs = [auroc(yva[:, c], pva[:, c]) for c in range(yva.shape[1])]
        macro = float(np.nanmean(aucs)) if not all(np.isnan(aucs)) else float("nan")
        log(f"  epoch {ep + 1:3d}  train loss {tot / len(perm):.4f}  val macro AUROC {macro:.4f}")
        if best_state is None or macro > best["macro_auroc"]:
            best = {"macro_auroc": macro, "epoch": ep + 1, "per_label_auroc": aucs}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pva = torch.sigmoid(model(xva_t)).cpu().numpy()
    best["thresholds"] = best_thresholds(yva, pva).tolist()
    return model.cpu(), best


def save_probe(model, info: dict, cfg: dict, out: Path) -> None:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "probe.pt")
    (out / "probe.json").write_text(json.dumps({**info, "config": cfg}, indent=2), encoding="utf-8")


def load_probe(out: Path):
    import torch

    info = json.loads((out / "probe.json").read_text(encoding="utf-8"))
    c = info["config"]
    model = build(c["d_in"], c["n_out"], c["hidden"], c["dropout"])
    model.load_state_dict(torch.load(out / "probe.pt", map_location="cpu"))
    model.eval()
    return model, info


def predict(model, x: np.ndarray, thresholds) -> tuple[np.ndarray, np.ndarray]:
    import torch

    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(x, dtype=torch.float32))).numpy()
    return p, (p >= np.asarray(thresholds)[None, :]).astype(np.int8)
