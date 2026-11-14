"""Step 0a: clean IU-Xray / MIMIC-CXR / ReXGradient into patient-disjoint train/val/test parquet.

  python scripts/step0_prepare_data.py --config configs/step0.yaml --inspect   # check paths/columns
  python scripts/step0_prepare_data.py --config configs/step0.yaml --smoke     # ~50 studies/split/dataset
  python scripts/step0_prepare_data.py --config configs/step0.yaml             # full

Outputs (under <project_root>/data/): {train,val,test}[_smoke].parquet, manifest[_smoke].json,
step0_eval_ids[_smoke].json. Re-running overwrites them (data prep is deterministic given the seed).
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.common import data_dir, load_config, set_seed, write_run_info  # noqa: E402
from src.data.loaders import LOADERS, finalize, inspect_rexgradient  # noqa: E402
from src.data.splits import assert_no_leakage, save_json, select_eval_ids, write_splits  # noqa: E402


def locate_rexgradient(dcfg: dict) -> dict:
    """If the configured ReXGradient paths are missing, look for the fetched folder under /kaggle/input
    (a notebook's output is mounted at a path that depends on Kaggle's layout)."""
    base = Path("/kaggle/input")

    def find(name: str, need_meta: bool):
        for depth in range(0, 5):
            for hit in base.glob("*/" * depth + name):
                if (hit / "deid_png").exists() and (not need_meta or (hit / "metadata").exists()):
                    return hit
        return None

    out = dict(dcfg)
    if not Path(dcfg["root"]).exists() and (hit := find("rexgradient-160k-test-images", True)):
        print(f"[rexgradient] configured root missing; using {hit}", flush=True)
        out.update(root=str(hit / "metadata"), images_root=str(hit / "deid_png"))
    extra = [x for x in (dcfg.get("extra_images_roots") or []) if Path(x).exists()]
    if not extra and (hit := find("rexgradient-160k-train-subset", False)):
        print(f"[rexgradient] train-subset images found at {hit}", flush=True)
        extra = [str(hit / "deid_png")]
    out["extra_images_roots"] = extra
    return out


def inspect(cfg: dict) -> None:
    ds = cfg["datasets"]
    for name, c in ds.items():
        print(f"\n=== {name} (enabled={c.get('enabled')})")
        for k, v in c.items():
            if isinstance(v, str) and ("/" in v or "\\" in v):
                p = Path(v) if Path(v).is_absolute() else Path(c.get("root", "")) / v
                print(f"  {k}: {p}  exists={p.exists()}")
    if ds.get("rexgradient", {}).get("enabled"):
        print("\n" + inspect_rexgradient(ds["rexgradient"]))
    iu = ds.get("iu_xray", {})
    if iu.get("enabled") and (Path(iu["root"]) / iu["reports_csv"]).exists():
        print("\nIU reports columns:", list(pd.read_csv(Path(iu["root"]) / iu["reports_csv"], nrows=2).columns))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step0.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--datasets", nargs="*", help="subset of dataset names (default: all enabled)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg["datasets"].get("rexgradient", {}).get("enabled"):
        cfg["datasets"]["rexgradient"] = locate_rexgradient(cfg["datasets"]["rexgradient"])
    if args.inspect:
        inspect(cfg)
        return
    seed = int(cfg["seed"])
    set_seed(seed)
    suffix = "_smoke" if args.smoke else ""
    smoke_n = int(cfg["eval"]["smoke_n"]) if args.smoke else None
    out = data_dir(cfg)
    filters = cfg["filters"]

    frames, manifest = [], {"seed": seed, "smoke": args.smoke, "filters": filters, "datasets": {}}
    for name, dcfg in cfg["datasets"].items():
        if not dcfg.get("enabled") or (args.datasets and name not in args.datasets):
            continue
        stats: Counter = Counter()
        print(f"[{name}] loading ...", flush=True)
        raw = LOADERS[name](dcfg, seed, stats, smoke_n=smoke_n, frontal_views=filters["frontal_views"])
        df = finalize(raw, filters, stats)
        frames.append(df)
        stats_out = {k: (v if not isinstance(v, (int,)) else int(v)) for k, v in stats.items()}
        stats_out["kept_per_split"] = df["split"].value_counts().to_dict()
        stats_out["with_prior_link"] = int(df["prior_study_id"].notna().sum())
        stats_out["with_lateral"] = int(df["lateral_image_paths"].map(len).gt(0).sum())
        stats_out["with_indication"] = int(df["indication"].str.len().gt(0).sum())
        manifest["datasets"][name] = stats_out
        print(f"[{name}] {stats_out}", flush=True)

    if not frames:
        raise SystemExit("No datasets enabled/selected.")
    df = pd.concat(frames, ignore_index=True)

    # Image files must exist (spot-check up to 200 per dataset; full check is too slow on Kaggle input FS).
    for name, g in df.groupby("dataset"):
        sample = g["image_path"].sample(min(200, len(g)), random_state=seed)
        missing = [p for p in sample if not Path(p).exists()]
        manifest["datasets"][name]["image_spotcheck_missing"] = f"{len(missing)}/{len(sample)}"
        if missing:
            raise FileNotFoundError(f"[{name}] {len(missing)}/{len(sample)} sampled images missing, e.g. {missing[:3]}")

    summary = assert_no_leakage(df)
    manifest["leakage_check"] = {"passed": True, "per_split": summary}
    print("LEAKAGE CHECK PASSED:", summary, flush=True)

    paths = write_splits(df, out, suffix)
    eval_ids = select_eval_ids(df[df["split"] == "test"], cfg["eval"]["max_test_studies"], seed)
    if args.smoke:
        eval_ids = {k: v[:smoke_n] for k, v in eval_ids.items()}
    save_json(eval_ids, out / f"step0_eval_ids{suffix}.json")
    manifest["eval_ids"] = {k: len(v) for k, v in eval_ids.items()}
    min_n = cfg["eval"]["min_test_studies"]
    manifest["eval_meets_min_n"] = {k: len(v) >= min_n for k, v in eval_ids.items()}
    save_json(manifest, out / f"manifest{suffix}.json")
    write_run_info(out, cfg, args, {"step": "step0_prepare", "outputs": {k: str(v) for k, v in paths.items()}})

    print("eval subset sizes:", manifest["eval_ids"])
    for k, ok in manifest["eval_meets_min_n"].items():
        if not ok and not args.smoke:
            print(f"WARNING: {k} has {manifest['eval_ids'][k]} test studies < {min_n} (Step 0 acceptance).")
    print("wrote", {k: str(v) for k, v in paths.items()})


if __name__ == "__main__":
    main()
