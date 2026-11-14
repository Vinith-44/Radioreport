"""Generic test-set generation for any system spec (src/models/systems.py). Resumable. Used by Step 9 (and 10).

  python scripts/generate_system.py --config configs/step9.yaml --type a --step step9 --run g1 \
      --system model-a-grpo --out-step step9 --out-run g1 [--smoke] [--test-file test]

Writes outputs/runs/<out-step>/<out-run>[_smoke]/<system>/<dataset>/predictions.jsonl in the Step 0 row format
(fixed Step 0 test subset, shared prompt, greedy decoding, generation.max_new_tokens).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from src.common import append_jsonl, data_dir, load_config, read_jsonl, run_dir, set_seed, write_run_info  # noqa: E402
from src.data.text_clean import format_report, parse_generated  # noqa: E402


def generate_for_spec(cfg: dict, spec: dict, out_dir: Path, smoke: bool, test_file: str = "test",
                      batch_size: int = 8) -> None:
    from src.models.systems import System

    sfx = "_smoke" if smoke else ""
    ddir = data_dir(cfg)
    test = pd.read_parquet(ddir / f"{test_file}{sfx}.parquet")
    eval_ids = json.loads((ddir / f"step0_eval_ids{sfx}.json").read_text(encoding="utf-8"))
    sysm = None
    for ds, ids in eval_ids.items():
        out = out_dir / ds / "predictions.jsonl"
        done = {r["study_id"] for r in read_jsonl(out)}
        sub = test[(test["dataset"] == ds) & test["study_id"].astype(str).isin(set(ids))]
        todo = sub[~sub["study_id"].astype(str).isin(done)].to_dict("records")
        print(f"[{out_dir.name}/{ds}] {len(done)} done, {len(todo)} to go", flush=True)
        if not todo:
            continue
        if sysm is None:
            sysm = System(cfg, spec, smoke)
        if sysm.use_prior:
            from step4_train_projector import prior_texts

            pri = dict(zip(sub["study_id"].astype(str), prior_texts(sub, test[test["dataset"] == ds])))
            todo = [{**r, "prior": pri[str(r["study_id"])]} for r in todo]
        out.parent.mkdir(parents=True, exist_ok=True)
        mx = int(cfg["generation"]["max_new_tokens"])
        for i in range(0, len(todo), 32):
            ch = todo[i:i + 32]
            texts = sysm.generate(ch, batch_size)
            rows = []
            for r, t in zip(ch, texts):
                p = parse_generated(t)
                rows.append({"study_id": str(r["study_id"]), "dataset": ds, "model": out_dir.name,
                             "model_id": sysm.name, "dtype": cfg["generation"]["dtype"],
                             "reference_findings": r["findings"], "reference_impression": r["impression"],
                             "reference": format_report(r["findings"], r["impression"]), "hypothesis": t,
                             "hyp_findings": p["findings"], "hyp_impression": p["impression"], "parse_ok": p["parse_ok"],
                             "hit_max_tokens": len(t.split()) >= mx})
            append_jsonl(out, rows)
            print(f"  [{out_dir.name}/{ds}] {len(done) + i + len(ch)}/{len(sub)}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--type", choices=["a", "b"], required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--out-step", required=True)
    ap.add_argument("--out-run", required=True)
    ap.add_argument("--test-file", default="test")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    spec = {"type": args.type, "step": args.step, "run": args.run, "system": args.system}
    out_dir = run_dir(cfg, args.out_step, args.out_run + ("_smoke" if args.smoke else "")) / args.system
    write_run_info(out_dir.parent, cfg, args, {"step": "generate_system", "spec": spec})
    generate_for_spec(cfg, spec, out_dir, args.smoke, args.test_file, args.batch_size)


if __name__ == "__main__":
    main()
