"""Step 2b: CheXbert labels (RadEval's labeler, 14 labels, rrg mode) of the TRAIN and VAL reference reports.

  python scripts/step2_label_reports.py --run-name r1 [--smoke]

Row-aligned with embeddings/<split>_ids.parquet (run step2_embed.py first). Used as probe targets (train/val) and
for re-ranking retrieved train cases. Test reports are not labelled here (they are only ever scoring references).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import load_config, run_dir, set_seed, write_run_info  # noqa: E402
from src.retrieval.rerank import CHEXBERT_LABELS, chexbert_labels  # noqa: E402
from src.retrieval.store import ids_sha, report_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2.yaml")
    ap.add_argument("--run-name", default="r1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["train", "val"], choices=["train", "val"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    mode = cfg["retrieval"]["report_text"]
    rdir = run_dir(cfg, "step2", args.run_name + ("_smoke" if args.smoke else ""))
    ldir = rdir / "labels"
    ldir.mkdir(parents=True, exist_ok=True)
    write_run_info(rdir, cfg, args, {"step": "step2_label_reports", "report_text": mode})
    for split in args.splits:
        ids = pd.read_parquet(rdir / "embeddings" / f"{split}_ids.parquet")
        meta = {"ids_sha256": ids_sha(ids), "n": len(ids), "report_text": mode, "labels": CHEXBERT_LABELS,
                "labeler": "radeval.metrics.f1chexbert.F1CheXbert get_labels(mode='rrg')"}
        mp = ldir / f"{split}.json"
        if mp.exists() and json.loads(mp.read_text(encoding="utf-8")) == meta and (ldir / f"{split}.npy").exists():
            print(f"[{split}] labels up to date.")
            continue
        t0 = time.time()
        lab = chexbert_labels([report_text(r, mode) for r in ids.to_dict("records")])
        np.save(ldir / f"{split}.npy", lab)
        mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        prev = dict(zip(CHEXBERT_LABELS, np.round(lab.mean(0) * 100, 1).tolist()))
        print(f"[{split}] {len(ids)} reports labelled in {time.time() - t0:.0f}s; % positive: {prev}", flush=True)


if __name__ == "__main__":
    main()
