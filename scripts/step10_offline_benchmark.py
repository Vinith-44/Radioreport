"""Step 10b: offline build - latency and memory of the final system on THIS machine (e.g. a laptop).

  python scripts/step10_offline_benchmark.py [--smoke] [--no-verify]

Model A finals run on the 4-bit (nf4) base + LoRA (configs/step1.yaml generation.dtype = nf4); Model B finals are
already small (Gemma 3 1B + MedSigLIP). bitsandbytes 4-bit needs an NVIDIA GPU: on a CPU-only machine the script
says so and measures the unquantised fallback instead (much slower) - it never reports a 4-bit number it did not
measure. Writes outputs/runs/step10/offline/benchmark.json and the STEP10 block of outputs/RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.common import data_dir, lib_versions, load_config, project_root, run_dir, write_run_info  # noqa: E402
from src.eval.scoring import update_results_md  # noqa: E402


def rss_gb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**30, 2)
    except ImportError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step10.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    import torch

    cfg = load_config(args.config)
    oc = cfg["offline"]
    cuda = torch.cuda.is_available()
    if not cuda and cfg["generation"]["dtype"] == "nf4":
        print("NOTE: no NVIDIA GPU -> bitsandbytes 4-bit is unavailable; measuring the fp32 CPU fallback.")
        cfg["generation"]["dtype"] = "fp32"
    cfg["app"]["boxes"]["enabled"] = False      # boxes = extra MedGemma calls; timed separately in the app
    rdir = run_dir(cfg, "step10", "offline" + ("_smoke" if args.smoke else ""))
    from src.app.pipeline import Pipeline

    t0 = time.time()
    pipe = Pipeline(cfg, args.smoke)
    _ = pipe.system
    load_s = time.time() - t0
    sfx = "_smoke" if args.smoke else ""
    test = pd.read_parquet(data_dir(cfg) / f"test{sfx}.parquet")
    ids = json.loads((data_dir(cfg) / f"step0_eval_ids{sfx}.json").read_text(encoding="utf-8"))
    want = [i for v in ids.values() for i in v][: 2 + (3 if args.smoke else int(oc["n_reports"]))]
    rows = test[test["study_id"].astype(str).isin(set(want))].to_dict("records")
    verify = bool(oc["with_verification"]) and not args.no_verify
    times = []
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for k, r in enumerate(rows):
        with Image.open(r["image_path"]) as im:
            from src.models.medgemma_zeroshot import to_rgb8

            img = to_rgb8(im)
        t = time.time()
        if verify:
            pipe.report(img, r.get("indication") or "")
        else:
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img.save(f.name)
                pipe.system.generate([{"image_path": f.name, "indication": r.get("indication") or ""}], 1)
        if k >= 2:   # first two = warm-up
            times.append(time.time() - t)
    res = {"device": torch.cuda.get_device_name(0) if cuda else platform.processor() or "cpu",
           "system": pipe.system.name, "dtype": cfg["generation"]["dtype"], "with_verification": verify,
           "load_s": round(load_s, 1), "n_timed": len(times),
           "latency_s_median": round(statistics.median(times), 2) if times else None,
           "latency_s_p90": round(sorted(times)[int(0.9 * (len(times) - 1))], 2) if times else None,
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None,
           "process_rss_gb": rss_gb(), "versions": lib_versions()}
    (rdir / "benchmark.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_run_info(rdir, cfg, args, {"step": "step10_offline_benchmark", **{k: v for k, v in res.items() if k != "versions"}})
    body = "\n".join([
        f"## Step 10 — offline build{' (SMOKE: not results)' if args.smoke else ''}",
        f"Device: {res['device']}; system {res['system']} ({res['dtype']}); verification "
        f"{'on' if verify else 'off'}, boxes off.", "",
        "| load (s) | median latency (s/report) | p90 (s) | peak GPU (GB) | process RSS (GB) | n |", "|---|---|---|---|---|---|",
        f"| {res['load_s']} | {res['latency_s_median']} | {res['latency_s_p90']} | {res['peak_gpu_gb']} | "
        f"{res['process_rss_gb']} | {res['n_timed']} |"])
    if not args.smoke:
        update_results_md(project_root(cfg) / "outputs" / "RESULTS.md", body, "STEP10")
    print("\n" + body)


if __name__ == "__main__":
    main()
