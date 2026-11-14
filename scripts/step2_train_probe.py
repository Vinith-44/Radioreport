"""Step 2c: train the Probe (14 CheXbert labels from frozen MedSigLIP embeddings). Seconds on CPU/GPU.

  python scripts/step2_train_probe.py --run-name r1 [--smoke]

Train split = fit, val split = epoch selection (macro AUROC) + per-label F1 thresholds. Writes <run>/probe/
(probe.pt, probe.json with val AUROC per label). Test is not used. Reused by Step 3.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common import load_config, run_dir, set_seed, write_run_info  # noqa: E402
from src.models.probe import save_probe, train_probe  # noqa: E402
from src.retrieval.rerank import CHEXBERT_LABELS  # noqa: E402
from src.retrieval.store import load_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2.yaml")
    ap.add_argument("--run-name", default="r1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    pc = dict(cfg["probe"])
    if args.smoke:
        pc["epochs"] = 3
    rdir = run_dir(cfg, "step2", args.run_name + ("_smoke" if args.smoke else ""))
    tr_ids, xtr, ytr = load_split(rdir, "train", labels=True)
    va_ids, xva, yva = load_split(rdir, "val", labels=True)
    assert not set(tr_ids["patient_id"]) & set(va_ids["patient_id"]), "train/val patient overlap"
    write_run_info(rdir, cfg, args, {"step": "step2_train_probe", "probe": pc, "n_train": len(xtr), "n_val": len(xva)})
    print(f"probe: train {xtr.shape}, val {xva.shape}", flush=True)

    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, info = train_probe(xtr, ytr, xva, yva, hidden=int(pc["hidden"]), dropout=float(pc["dropout"]),
                              epochs=int(pc["epochs"]), lr=float(pc["lr"]), weight_decay=float(pc["weight_decay"]),
                              batch_size=int(pc["batch_size"]), seed=seed, device=dev)
    save_probe(model, {**info, "labels": CHEXBERT_LABELS, "n_train": len(xtr), "n_val": len(xva),
                       "val_positives": yva.sum(0).tolist()},
               {**pc, "d_in": int(xtr.shape[1]), "n_out": len(CHEXBERT_LABELS)}, rdir / "probe")
    print(f"\nbest epoch {info['epoch']}: val macro AUROC {info['macro_auroc']:.4f}")
    for name, a, n in zip(CHEXBERT_LABELS, info["per_label_auroc"], yva.sum(0)):
        print(f"  {name:28s} AUROC {'  n/a' if np.isnan(a) else f'{a:.3f}'}   (val positives: {int(n)})")


if __name__ == "__main__":
    main()
