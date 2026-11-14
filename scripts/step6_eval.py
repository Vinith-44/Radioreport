"""Step 6b: generate reports for {winner, its robust twin} x {clean test, photo-style test, CheXphoto clean, CheXphoto
photo}. Resumable (one JSONL per arm, finished studies skipped).

  python scripts/step6_eval.py --run-name e1 [--smoke]

winner = outputs/runs/step5/<robust.winner_run>/winner.json; robust twin = same type, runs/step6/<robust.robust_run>.
Writes outputs/runs/step6/<run>/<arm>/<dataset>/predictions.jsonl (report sets, Step 0 row format) and
<arm>/chexphoto/predictions.jsonl (+ the CheXpert labels of each photo). The winner's clean-test predictions are
copied from its own run (identical rows) instead of being generated again.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from src.common import (append_jsonl, data_dir, load_config, project_root, read_jsonl, run_dir,  # noqa: E402
                        set_seed, write_run_info)
from src.data.text_clean import format_report, parse_generated  # noqa: E402


def training_chexpert_patients(cfg: dict) -> set[str]:
    """Every CheXpert patient in any Model B training set (train_rows.csv of Steps 4-6 runs)."""
    out = set()
    for f in (project_root(cfg) / "outputs" / "runs").glob("step*/*/train_rows.csv"):
        df = pd.read_csv(f)
        ids = df.loc[df["study_id"].astype(str).str.startswith("chexpert:"), "study_id"].astype(str)
        out |= set("chexpert:" + ids.str.extract(r"(patient\d+)")[0].dropna())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step6.yaml")
    ap.add_argument("--run-name", default="e1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    rc = cfg["robust"]
    sfx = "_smoke" if args.smoke else ""
    ddir = data_dir(cfg)
    rdir = run_dir(cfg, "step6", args.run_name + sfx)
    wfile = run_dir(cfg, "step5", rc["winner_run"] + sfx) / "winner.json"
    if not wfile.exists():
        raise SystemExit(f"no {wfile} (run step5_score.py first)")
    win = json.loads(wfile.read_text(encoding="utf-8"))["winner_spec"]
    robust = {**win, "step": "step6", "run": rc["robust_run"], "system": win["system"] + "-robust"}
    systems = {"baseline": win, "robust": robust}
    write_run_info(rdir, cfg, args, {"step": "step6_eval", "systems": systems})
    eval_ids = json.loads((ddir / f"step0_eval_ids{sfx}.json").read_text(encoding="utf-8"))

    sets = {"clean": pd.read_parquet(ddir / f"test{sfx}.parquet")}
    pp = ddir / f"test_photo{sfx}.parquet"
    if not pp.exists():
        raise SystemExit(f"no {pp.name} (run step6_make_photo_test.py first)")
    sets["photo"] = pd.read_parquet(pp)
    cx = rc["chexphoto"]
    pairs = None
    if Path(cx["root"]).exists():
        from src.data.chexphoto import load_pairs

        pairs, n_ex = load_pairs(Path(cx["root"]), Path(cx["clean_root"]), list(cx["kinds"]),
                                 training_chexpert_patients(cfg), 8 if args.smoke else int(cx["n"]), seed)
        print(f"CheXphoto: {len(pairs)} photo/clean pairs (excluded {n_ex} photos of training patients)", flush=True)
        pairs.to_parquet(rdir / "chexphoto_pairs.parquet", index=False)
    else:
        print(f"CheXphoto not found at {cx['root']} - only the synthetic photo test is evaluated")

    from src.models.systems import System, spec_dir

    for label, spec in systems.items():
        sysm = None
        # report test sets
        for sname, df in sets.items():
            arm = rdir / f"{label}-{sname}"
            for ds, ids in eval_ids.items():
                out = arm / ds / "predictions.jsonl"
                src = spec_dir(cfg, spec, args.smoke) / spec["system"] / ds / "predictions.jsonl"
                if label == "baseline" and sname == "clean" and src.exists() and not out.exists():
                    if len(read_jsonl(src)) >= len(ids):
                        out.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy(src, out)
                        print(f"[{arm.name}/{ds}] copied the winner's own clean-test predictions")
                        continue
                done = {r["study_id"] for r in read_jsonl(out)}
                sub = df[(df["dataset"] == ds) & df["study_id"].astype(str).isin(set(ids))]
                todo = sub[~sub["study_id"].astype(str).isin(done)].to_dict("records")
                print(f"[{arm.name}/{ds}] {len(done)} done, {len(todo)} to go", flush=True)
                if not todo:
                    continue
                if sysm is None:
                    sysm = System(cfg, spec, args.smoke)
                if sysm.use_prior:
                    from step4_train_projector import prior_texts  # noqa: E402

                    pri = dict(zip(sub["study_id"].astype(str), prior_texts(sub, df[df["dataset"] == ds])))
                    todo = [{**r, "prior": pri[str(r["study_id"])]} for r in todo]
                out.parent.mkdir(parents=True, exist_ok=True)
                for i in range(0, len(todo), 32):
                    ch = todo[i:i + 32]
                    texts = sysm.generate(ch, args.batch_size)
                    rows = []
                    for r, t in zip(ch, texts):
                        p = parse_generated(t)
                        rows.append({"study_id": str(r["study_id"]), "dataset": ds, "model": f"{label}-{sname}",
                                     "model_id": sysm.name, "dtype": cfg["generation"]["dtype"],
                                     "reference_findings": r["findings"], "reference_impression": r["impression"],
                                     "reference": format_report(r["findings"], r["impression"]), "hypothesis": t,
                                     "hyp_findings": p["findings"], "hyp_impression": p["impression"],
                                     "parse_ok": p["parse_ok"], "hit_max_tokens": False,
                                     "photo_ops": r.get("photo_ops")})
                    append_jsonl(out, rows)
                    print(f"  [{arm.name}/{ds}] {len(done) + i + len(ch)}/{len(sub)}", flush=True)
        # CheXphoto: label agreement only (no reference reports exist)
        if pairs is not None:
            for kind, col in (("clean", "clean_path"), ("photo", "photo_path")):
                out = rdir / f"{label}-chexphoto_{kind}" / "chexphoto" / "predictions.jsonl"
                done = {r["study_id"] for r in read_jsonl(out)}
                todo = [r for r in pairs.to_dict("records") if r["study_id"] not in done]
                print(f"[{label}-chexphoto_{kind}] {len(done)} done, {len(todo)} to go", flush=True)
                if not todo:
                    continue
                if sysm is None:
                    sysm = System(cfg, spec, args.smoke)
                out.parent.mkdir(parents=True, exist_ok=True)
                for i in range(0, len(todo), 32):
                    ch = todo[i:i + 32]
                    texts = sysm.generate([{"image_path": r[col], "indication": "", "prior": ""} for r in ch],
                                          args.batch_size)
                    append_jsonl(out, [{"study_id": r["study_id"], "image": r[col], "hypothesis": t,
                                        "labels": {k: r[k] for k in r if k.startswith("label:")}}
                                       for r, t in zip(ch, texts)])
        del sysm
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
