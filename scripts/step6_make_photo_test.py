"""Step 6a: a FIXED phone-photo / WhatsApp-style copy of the Step 0 test subset (report metrics need references).

  python scripts/step6_make_photo_test.py [--smoke]

Each test image -> to 8-bit -> 896x896 -> photo_augment with a per-study seed (robust.photo_seed + sha of the study
id), saved as PNG under <project>/data/photo_test/<dataset>/. Writes data/test_photo[_smoke].parquet: the test rows
of the eval subset with image_path pointing at the corrupted copy, plus the corruptions applied. Every system sees
exactly the same corrupted images. These synthetic photos are never used for training (training draws fresh
random corruptions), but they come from the same corruption family - CheXphoto is the real-photo check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.common import data_dir, load_config, write_run_info  # noqa: E402
from src.data.augment import photo_augment  # noqa: E402
from src.models.medgemma_zeroshot import load_image  # noqa: E402


def study_rng(seed: int, study_id: str) -> random.Random:
    return random.Random(seed * 1_000_003 + int(hashlib.sha256(study_id.encode()).hexdigest()[:12], 16))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step6.yaml")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sfx = "_smoke" if args.smoke else ""
    ddir = data_dir(cfg)
    seed = int(cfg["robust"]["photo_seed"])
    test = pd.read_parquet(ddir / f"test{sfx}.parquet")
    ids = json.loads((ddir / f"step0_eval_ids{sfx}.json").read_text(encoding="utf-8"))
    want = {str(i) for v in ids.values() for i in v}
    sub = test[test["study_id"].astype(str).isin(want)].copy()
    out_dir = ddir / f"photo_test{sfx}"
    paths, ops = [], []
    for r in sub.to_dict("records"):
        sid = str(r["study_id"])
        p = out_dir / r["dataset"] / (hashlib.sha256(sid.encode()).hexdigest()[:20] + ".png")
        rng = study_rng(seed, sid)
        im = load_image(r["image_path"]).convert("L").resize((896, 896), Image.BILINEAR)
        aug, applied = photo_augment(im, rng, return_ops=True)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            aug.save(p)
        paths.append(str(p))
        ops.append(",".join(applied))
    sub["clean_image_path"] = sub["image_path"]
    sub["image_path"] = paths
    sub["image_paths"] = [[p] for p in paths]
    sub["photo_ops"] = ops
    sub.to_parquet(ddir / f"test_photo{sfx}.parquet", index=False)
    write_run_info(ddir, cfg, args, {"step": "step6_make_photo_test", "n": len(sub), "photo_seed": seed})
    print(f"{len(sub)} photo-style test images -> {out_dir}; ops used: "
          f"{pd.Series([o for s in ops for o in s.split(',')]).value_counts().to_dict()}")


if __name__ == "__main__":
    main()
