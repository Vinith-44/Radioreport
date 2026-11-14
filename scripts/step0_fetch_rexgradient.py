"""Step 0 (data): fetch the official ReXGradient-160K test (and optionally val) images from Hugging Face.

Run ON KAGGLE (needs ~10 GB per split of disk; streams the 155 GB archive once, stores only wanted PNGs):
  python scripts/step0_fetch_rexgradient.py --smoke          # first ~300 MB of the archive (pipeline check)
  python scripts/step0_fetch_rexgradient.py                  # test split (default)
  python scripts/step0_fetch_rexgradient.py --splits test val

Needs HF_TOKEN (Kaggle Secret) with access to the gated repo. Output (fetch.out_dir in the config):
  <out_dir>/metadata/{train,valid,test}_metadata_view_position.json
  <out_dir>/deid_png/<PatientID>/.../<UID>.png
  <out_dir>/fetch_manifest.json
Then save <out_dir> as a PRIVATE Kaggle dataset (license forbids redistribution) and point
datasets.rexgradient.root / images_root at it.
Re-runnable: files already on disk are not re-written; the stream stops as soon as every wanted file exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config, write_run_info  # noqa: E402
from src.data.rexgradient_fetch import (PARTS, ConcatParts, download_metadata, extract_wanted,  # noqa: E402
                                        hf_open_part_factory, load_meta, wanted_from_metadata, zstd_reader)
from src.models.medgemma_zeroshot import hf_token  # noqa: E402

META_FILES = {"train": "train_metadata_view_position.json", "val": "valid_metadata_view_position.json",
              "test": "test_metadata_view_position.json"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step0.yaml")
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["train", "val", "test"])
    ap.add_argument("--out-dir", help="override fetch.out_dir")
    ap.add_argument("--smoke", action="store_true", help="stream only the first ~300 MB")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fcfg = cfg["datasets"]["rexgradient"]["fetch"]
    out = Path(args.out_dir or fcfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    tok = hf_token()
    if not tok:
        raise SystemExit("HF_TOKEN not set (Kaggle: Add-ons -> Secrets -> HF_TOKEN).")

    print("downloading metadata ...", flush=True)
    paths = download_metadata(fcfg["repo"], tok, out / "metadata", list(META_FILES.values()))
    meta_dir = paths[0].parent
    wanted: set[str] = set()
    per_split = {}
    for s in args.splits:
        w = wanted_from_metadata(load_meta(meta_dir / META_FILES[s]))
        per_split[s] = len(w)
        wanted |= w
    img_dir = out / "deid_png"
    if args.smoke:
        # Test studies are spread over the whole archive; add train so the smoke run proves files get written.
        wanted |= wanted_from_metadata(load_meta(meta_dir / META_FILES["train"]))
        img_dir = out / "smoke_deid_png"
    print(f"wanted images: {per_split} (total {len(wanted)}){' + train (smoke only)' if args.smoke else ''}",
          flush=True)
    write_run_info(out, cfg, args, {"step": "step0_fetch_rexgradient", "wanted": per_split})

    limit = 300 * 2**20 if args.smoke else None
    raw = ConcatParts(hf_open_part_factory(fcfg["repo"], tok), PARTS, byte_limit=limit)
    t0 = time.time()

    def report(st: dict) -> None:
        gb = raw.total / 1e9
        rate = gb / max(1e-9, time.time() - t0)
        eta = (155.0 - gb) / rate / 60 if rate > 0 else float("nan")
        print(f"  {gb:6.1f}/155 GB streamed ({rate * 1000:.0f} MB/s, ETA {eta:.0f} min) | "
              f"found {st['found']}/{st['wanted']} | written {st['written']} | part {raw.i}", flush=True)

    try:
        stats = extract_wanted(zstd_reader(raw), wanted, img_dir, report=report,
                               progress_every=2000 if args.smoke else 20000)
    except (EOFError, tarfile.ReadError) as e:
        if not args.smoke:
            raise
        n = sum(1 for _ in img_dir.rglob("*.png")) if img_dir.exists() else 0
        stats = {"note": f"smoke byte limit reached ({type(e).__name__})", "smoke_pngs_written": n}
    stats.update({"splits": args.splits, "wanted_per_split": per_split, "streamed_gb": round(raw.total / 1e9, 2),
                  "minutes": round((time.time() - t0) / 60, 1), "smoke": args.smoke})
    missing = stats.get("missing", [])
    stats["missing_count"] = len(missing)
    stats["missing_examples"] = missing[:20]
    stats.pop("missing", None)
    (out / "fetch_manifest.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    if missing and not args.smoke:
        raise SystemExit(f"{len(missing)} wanted images were not found in the archive (see fetch_manifest.json).")


if __name__ == "__main__":
    main()
