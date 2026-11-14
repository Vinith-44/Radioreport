"""Step 5b: Model B test predictions on the fixed Step 0 test subset. Resumable.

  python scripts/step5_generate.py --run-name b2 [--smoke]
  python scripts/step5_generate.py --run-name b2 --test-file test_photo --system model-b-photo   # Step 6

Same row format as scripts/step0_zeroshot_generate.py (so the shared scoring code applies), same prompt template,
greedy decoding and max_new_tokens as Model A; plus the prior report when Model B was trained with it.
Writes outputs/runs/<run_step>/<run>[_smoke]/<system>/<dataset>/predictions.jsonl.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import pandas as pd  # noqa: E402

from src.common import append_jsonl, data_dir, load_config, read_jsonl, run_dir, set_seed, write_run_info  # noqa: E402
from src.data.text_clean import format_report, parse_generated  # noqa: E402

LOCKED = ("model_run", "projector_sha256", "prompt_template", "max_new_tokens", "use_prior", "test_file")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step5.yaml")
    ap.add_argument("--run-name", default="b2", help="trained Model B run (under runs/<run_step>)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--system", help="output folder name (default model_b_eval.system)")
    ap.add_argument("--test-file", default="test", help="test parquet stem, e.g. test_photo (Step 6)")
    ap.add_argument("--out-step", help="write under runs/<out-step>/<run> (default: the model's run_step)")
    args = ap.parse_args()

    import torch

    from step4_train_projector import AlignDS, add_prior, collate, prior_texts  # noqa: F401

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    ev, g = cfg["model_b_eval"], cfg["generation"]
    sfx = "_smoke" if args.smoke else ""
    stage = cfg.get("run_step", "step5")
    mrun = run_dir(cfg, stage, args.run_name + sfx)
    if not (mrun / "projector.pt").exists():
        raise SystemExit(f"no trained Model B at {mrun} (finish training first)")
    system = args.system or ev["system"]
    out_root = run_dir(cfg, args.out_step or stage, args.run_name + sfx)
    ddir = data_dir(cfg)
    test = pd.read_parquet(ddir / f"{args.test_file}{sfx}.parquet")
    eval_ids = json.loads((ddir / f"step0_eval_ids{sfx}.json").read_text(encoding="utf-8"))
    meta_model = json.loads((mrun / "projector.json").read_text(encoding="utf-8"))
    use_prior = bool(meta_model.get("use_prior"))
    settings = {"model_run": str(mrun), "projector_sha256": sha(mrun / "projector.pt"),
                "prompt_template": g["prompt_template"], "max_new_tokens": int(ev["max_new_tokens"]),
                "use_prior": use_prior, "test_file": args.test_file}
    write_run_info(out_root, cfg, args, {"step": f"{stage}_generate_model_b", **settings})

    plan = []
    for ds, ids in eval_ids.items():
        od = out_root / system / ds
        od.mkdir(parents=True, exist_ok=True)
        mp = od / "generation_meta.json"
        if mp.exists():
            old = json.loads(mp.read_text(encoding="utf-8"))
            diff = {k: (old.get(k), settings[k]) for k in LOCKED if old.get(k) != settings[k]}
            if diff:
                raise SystemExit(f"[{ds}] settings differ from the existing predictions: {diff}; use a new --system")
        else:
            mp.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        done = {r["study_id"] for r in read_jsonl(od / "predictions.jsonl")}
        sub = test[(test["dataset"] == ds) & test["study_id"].astype(str).isin(set(ids))].copy()
        assert len(sub) == len(ids), f"[{ds}] eval ids missing from {args.test_file}{sfx}.parquet"
        sub["prior"] = prior_texts(sub, test[test["dataset"] == ds]) if use_prior else ""
        sub["target"] = ""
        todo = sub[~sub["study_id"].astype(str).isin(done)]
        print(f"[{system}/{ds}] {len(done)} done, {len(todo)} to go", flush=True)
        if len(todo):
            plan.append((ds, od / "predictions.jsonl", todo))
    if not plan:
        print("Nothing to do.")
        return

    from torch.utils.data import DataLoader

    from src.models.medgemma_zeroshot import hf_token
    from src.models.model_b import load_trained

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc, tok, _ = load_trained(mrun, device, hf_token())
    mx = int(ev["max_new_tokens"])
    for ds, pred_p, todo in plan:
        rows = todo.to_dict("records")
        dl = DataLoader(AlignDS(rows, proc, tok, g["prompt_template"], g["empty_indication"], with_target=False,
                                use_prior=use_prior), batch_size=int(ev["batch_size"]), collate_fn=collate, num_workers=2)
        t0, n = time.time(), 0
        for b in dl:
            texts = model.generate(b["px"].to(device), b["pre"], b["post"], tok, mx)
            lines = []
            for i, t in zip(b["i"], texts):
                r = rows[i]
                p = parse_generated(t)
                n_tok = len(tok(t, add_special_tokens=False)["input_ids"])
                lines.append({"study_id": str(r["study_id"]), "dataset": ds, "model": system,
                              "model_id": f"model-b:{mrun.name}", "dtype": "/".join(meta_model["dtypes"]),
                              "indication": r["indication"], "prior_used": bool(use_prior and r["prior"]),
                              "reference_findings": r["findings"], "reference_impression": r["impression"],
                              "reference": format_report(r["findings"], r["impression"]), "hypothesis": t,
                              "hyp_findings": p["findings"], "hyp_impression": p["impression"], "parse_ok": p["parse_ok"],
                              "n_new_tokens": n_tok, "hit_max_tokens": n_tok >= mx})
            append_jsonl(pred_p, lines)
            n += len(lines)
            rate = (time.time() - t0) / n
            print(f"[{system}/{ds}] {n}/{len(rows)}  {rate:.2f}s/study  ETA {rate * (len(rows) - n) / 60:.0f} min",
                  flush=True)
    for ds, pred_p, _ in plan:
        rs = read_jsonl(pred_p)
        print(f"\n=== {system}/{ds}: {len(rs)} predictions, parse_ok {sum(r['parse_ok'] for r in rs)}/{len(rs)}")
        for r in rs[:2]:
            print(f"--- {r['study_id']}\nREF: {r['reference'][:300]}\nHYP: {r['hypothesis'][:300]}")


if __name__ == "__main__":
    main()
