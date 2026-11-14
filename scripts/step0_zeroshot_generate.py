"""Step 0b: zero-shot MedGemma generation on the fixed Step 0 test subset. Resumable.

  python scripts/step0_zeroshot_generate.py --model medgemma-1.5 --run-name zs --smoke
  python scripts/step0_zeroshot_generate.py --model medgemma-1.5 --run-name zs
  python scripts/step0_zeroshot_generate.py --model medgemma-1   --run-name zs

Writes <project_root>/outputs/runs/<step>/<run-name>[_smoke]/<system>/<dataset>/predictions.jsonl, one line
per study (study_id, reference, hypothesis, ...). Re-running with the same --run-name skips finished studies.
Step 1 reuses this exact code path with --step step1 --adapter <LoRA dir> (see scripts/step1_generate.py), so
fine-tuned and zero-shot predictions differ only in the adapter.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.common import (append_jsonl, data_dir, load_config, read_jsonl, run_dir,  # noqa: E402
                        set_seed, write_run_info)
from src.data.text_clean import format_report, parse_generated  # noqa: E402
from src.models.medgemma_zeroshot import (adapter_digest, build_prompt, generate_batch,  # noqa: E402
                                          load_image, load_model)

# Settings that must not change within a run (otherwise one predictions file would mix two systems).
LOCKED = ("model_id", "dtype", "prompt_template", "empty_indication", "max_new_tokens", "do_sample",
          "adapter_sha256", "retrieval_sha256", "test_file")


def _collate(batch):
    return batch


class _DS:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return r, load_image(r["image_path"])


def main(defaults: dict | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step0.yaml")
    ap.add_argument("--model", help="key under models: in the config (required)")
    ap.add_argument("--run-name", default="zeroshot")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "nf4"])
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--step", default="step0", help="outputs/runs/<step>/...")
    ap.add_argument("--test-file", default="test", help="test parquet stem (Step 6: test_photo)")
    ap.add_argument("--adapter", help="LoRA adapter directory (Step 1+)")
    ap.add_argument("--system", help="output folder / system name (default: --model)")
    ap.add_argument("--retrieval", help="Step 2: retrieval.jsonl (study_id -> references block); switches to "
                                        "generation.prompt_template_retrieval")
    ap.set_defaults(**(defaults or {}))
    args = ap.parse_args()
    if not args.model:
        ap.error("--model is required")
    system = args.system or args.model

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    g = dict(cfg["generation"])
    if args.dtype:
        g["dtype"] = args.dtype
    if args.batch_size:
        g["batch_size"] = args.batch_size
    model_id = cfg["models"][args.model]
    suffix = "_smoke" if args.smoke else ""

    ddir = data_dir(cfg)
    test = pd.read_parquet(ddir / f"{args.test_file}{suffix}.parquet")
    eval_ids = json.loads((ddir / f"step0_eval_ids{suffix}.json").read_text(encoding="utf-8"))
    datasets = args.datasets or list(eval_ids)

    rdir = run_dir(cfg, args.step, args.run_name + suffix)
    a_sha = adapter_digest(args.adapter)
    write_run_info(rdir, cfg, args, {"step": f"{args.step}_generate", "model_id": model_id, "generation": g,
                                     "system": system, "adapter": args.adapter, "adapter_sha256": a_sha})
    refs: dict[str, str] = {}
    r_sha = None
    template = g["prompt_template"]
    if args.retrieval:
        import hashlib

        raw = Path(args.retrieval).read_bytes()
        r_sha = hashlib.sha256(raw).hexdigest()
        refs = {r["study_id"]: r["references"] for r in read_jsonl(Path(args.retrieval))}
        template = g["prompt_template_retrieval"]
    settings = {"model_id": model_id, "dtype": g["dtype"], "prompt_template": template,
                "retrieval_sha256": r_sha, "test_file": args.test_file,
                "empty_indication": g["empty_indication"], "max_new_tokens": g["max_new_tokens"],
                "do_sample": g["do_sample"], "adapter": args.adapter, "adapter_sha256": a_sha}

    # Work out what is left before loading the model.
    plan = []
    for ds in datasets:
        out_dir = rdir / system / ds
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_p = out_dir / "generation_meta.json"
        if meta_p.exists():
            old = json.loads(meta_p.read_text(encoding="utf-8"))
            diff = {k: (old.get(k), settings[k]) for k in LOCKED if old.get(k) != settings[k]}
            if diff:
                raise SystemExit(f"[{ds}] settings differ from the existing run {rdir.name}: {diff}. "
                                 f"Use a new --run-name.")
        else:
            meta_p.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        pred_p = out_dir / "predictions.jsonl"
        done = {r["study_id"] for r in read_jsonl(pred_p)}
        sub = test[(test["dataset"] == ds) & (test["study_id"].astype(str).isin(set(eval_ids[ds])))]
        assert len(sub) == len(eval_ids[ds]), f"[{ds}] eval ids missing from test parquet"
        todo = sub[~sub["study_id"].astype(str).isin(done)]
        if args.retrieval:
            miss = [i for i in todo["study_id"].astype(str) if i not in refs]
            if miss:
                raise SystemExit(f"[{ds}] {len(miss)} studies have no retrieved references, e.g. {miss[:3]}")
        print(f"[{system}/{ds}] {len(done)} done, {len(todo)} to go -> {pred_p}", flush=True)
        if len(todo):
            plan.append((ds, pred_p, todo))
    if not plan:
        print("Nothing to do.")
        return

    import torch
    from torch.utils.data import DataLoader

    t0 = time.time()
    model, processor = load_model(model_id, g["dtype"], args.adapter)
    print(f"loaded {model_id} ({g['dtype']}){' + adapter ' + args.adapter if args.adapter else ''} in {time.time() - t0:.0f}s; "
          f"device map: {getattr(model, 'hf_device_map', None) and set(model.hf_device_map.values())}", flush=True)

    n_seen, n_empty = 0, 0
    for ds, pred_p, todo in plan:
        rows = todo.to_dict("records")
        dl = DataLoader(_DS(rows), batch_size=g["batch_size"], num_workers=g["num_workers"],
                        collate_fn=_collate, shuffle=False)
        t_ds, n_ds = time.time(), 0
        for batch in dl:
            recs, images = [b[0] for b in batch], [b[1] for b in batch]
            prompts = [build_prompt(template, r["indication"], g["empty_indication"],
                                    refs[str(r["study_id"])] if args.retrieval else None) for r in recs]
            tb = time.time()
            outs = generate_batch(model, processor, images, prompts, g["max_new_tokens"], g["do_sample"])
            dt = (time.time() - tb) / len(recs)
            lines = []
            for r, o, pr in zip(recs, outs, prompts):
                p = parse_generated(o["text"])
                lines.append({
                    "study_id": str(r["study_id"]), "dataset": ds, "model": system, "model_id": model_id,
                    "dtype": g["dtype"], "adapter_sha256": a_sha, "retrieval_sha256": r_sha,
                    "indication": r["indication"], "prompt": pr if args.retrieval else None,
                    "reference_findings": r["findings"], "reference_impression": r["impression"],
                    "reference": format_report(r["findings"], r["impression"]),
                    "hypothesis": o["text"], "hyp_findings": p["findings"], "hyp_impression": p["impression"],
                    "parse_ok": p["parse_ok"], "n_new_tokens": o["n_new_tokens"],
                    "hit_max_tokens": o["hit_max_tokens"], "sec_per_study": round(dt, 3),
                })
            append_jsonl(pred_p, lines)
            n_ds += len(recs)
            # Empty-output guard (fp16 overflow symptom) over the first outputs of this process.
            if n_seen < g["empty_check_n"]:
                n_seen += len(outs)
                n_empty += sum(not o["text"] for o in outs)
                if n_seen >= g["empty_check_n"] and n_empty / n_seen > g["empty_abort_frac"]:
                    raise SystemExit(f"ABORT: {n_empty}/{n_seen} empty outputs with dtype={g['dtype']}. "
                                     f"Use --dtype fp32 (and a new --run-name).")
            rate = (time.time() - t_ds) / n_ds
            print(f"[{system}/{ds}] {n_ds}/{len(rows)}  {rate:.2f}s/study  "
                  f"ETA {rate * (len(rows) - n_ds) / 60:.0f} min  "
                  f"mem {[round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())]} GB",
                  flush=True)
        print(f"[{system}/{ds}] finished.", flush=True)

    # Show a few raw outputs (CLAUDE.md §8: show the smoke output).
    for ds, pred_p, _ in plan:
        rs = read_jsonl(pred_p)
        print(f"\n=== {system}/{ds}: {len(rs)} predictions, parse_ok={sum(r['parse_ok'] for r in rs)}/{len(rs)}, "
              f"hit_max_tokens={sum(r['hit_max_tokens'] for r in rs)}")
        for r in rs[:2]:
            print(f"--- study {r['study_id']}\nREF: {r['reference'][:400]}\nHYP: {r['hypothesis'][:400]}")


if __name__ == "__main__":
    main()
