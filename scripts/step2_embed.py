"""Step 2a: MedSigLIP-448 image embeddings for train (index + probe), val (probe selection), test subset (queries).

  python scripts/step2_embed.py --run-name r1 --smoke
  python scripts/step2_embed.py --run-name r1            # resumable: finished 2048-image chunks are kept

Writes outputs/runs/step2/<run>[_smoke]/embeddings/{train,val,test}.npy (+ _ids.parquet, .json).
Test rows keep NO report text in the ids file: queries are images only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common import data_dir, load_config, run_dir, set_seed, write_run_info  # noqa: E402
from src.retrieval.embed import PREPROC_VERSION, embed_rows  # noqa: E402
from src.retrieval.store import ids_sha, split_rows  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2.yaml")
    ap.add_argument("--run-name", default="r1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    ap.add_argument("--dtype", choices=["fp16", "fp32"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    rc = cfg["retrieval"]
    dtype = args.dtype or rc["dtype"]
    suffix = "_smoke" if args.smoke else ""
    ddir = data_dir(cfg)
    rdir = run_dir(cfg, "step2", args.run_name + suffix)
    edir = rdir / "embeddings"
    edir.mkdir(parents=True, exist_ok=True)
    write_run_info(rdir, cfg, args, {"step": "step2_embed", "encoder": rc["encoder"], "dtype": dtype,
                                     "preprocessing": PREPROC_VERSION})

    from src.models.medgemma_zeroshot import hf_token
    from src.retrieval.embed import SigLIPEmbedder

    emb_fn = None
    for split in args.splits:
        df = split_rows(ddir, split, suffix)
        meta = {"encoder": rc["encoder"], "dtype": dtype, "preprocessing": PREPROC_VERSION, "n": len(df),
                "ids_sha256": ids_sha(df), "datasets": df["dataset"].value_counts().to_dict()}
        mp = edir / f"{split}.json"
        if mp.exists() and (edir / f"{split}.npy").exists():
            old = json.loads(mp.read_text(encoding="utf-8"))
            old.pop("dim", None)
            if old == meta:
                print(f"[{split}] up to date ({len(df)} images).")
                continue
            if {k: old.get(k) for k in ("encoder", "dtype", "preprocessing")} != \
                    {k: meta[k] for k in ("encoder", "dtype", "preprocessing")}:
                raise SystemExit(f"[{split}] existing embeddings used other settings {old}; use a new --run-name")
        if emb_fn is None:
            t0 = time.time()
            emb_fn = SigLIPEmbedder(rc["encoder"], dtype=dtype, token=hf_token())
            print(f"loaded {rc['encoder']} ({dtype}) in {time.time() - t0:.0f}s", flush=True)
        print(f"[{split}] {len(df)} images: {meta['datasets']}", flush=True)
        t0 = time.time()
        # Chunk folder keyed by the exact study list, so a changed split can never reuse stale chunks.
        e = embed_rows(df["image_path"].tolist(), emb_fn, edir / "parts" / meta["ids_sha256"][:16], split,
                       batch_size=int(rc["batch_size"]),
                       num_workers=int(rc["num_workers"]))
        assert e.shape[0] == len(df) and np.isfinite(e).all()
        assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-3)
        np.save(edir / f"{split}.npy", e)
        keep = ["study_id", "patient_id", "dataset", "image_path"] + ([] if split == "test" else ["findings", "impression"])
        df[keep].to_parquet(edir / f"{split}_ids.parquet", index=False)
        mp.write_text(json.dumps({**meta, "dim": int(e.shape[1])}, indent=2), encoding="utf-8")
        print(f"[{split}] done: {e.shape} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
