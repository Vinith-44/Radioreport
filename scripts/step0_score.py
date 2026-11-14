"""Step 0c: score saved predictions with RadEval, run the paired test, write the results table.

  python scripts/step0_score.py --run-name zeroshot --smoke
  python scripts/step0_score.py --run-name zeroshot

Reads   outputs/runs/step0/<run>/<model>/<dataset>/predictions.jsonl
Writes  .../<model>/<dataset>/metrics.json, per_sample.jsonl
        outputs/runs/step0/<run>/significance.json, results_table.md
        <project_root>/outputs/RESULTS.md (Step 0 section)
Resumable: a (model, dataset) whose metrics.json matches the current predictions is not re-scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import data_dir, load_config, project_root, read_jsonl, run_dir, set_seed, write_run_info  # noqa: E402
from src.eval.radeval_wrapper import Scorer, paired_test, results_table, target_texts  # noqa: E402

BEGIN, END = "<!-- STEP0:BEGIN -->", "<!-- STEP0:END -->"


def _digest(rows: list[dict], metrics_cfg: dict) -> str:
    h = hashlib.sha256(json.dumps(metrics_cfg, sort_keys=True).encode())
    for r in rows:
        h.update(f"{r['study_id']}\x00{r['hypothesis']}\x00{r['reference']}\n".encode("utf-8"))
    return h.hexdigest()


def update_results_md(path: Path, body: str) -> None:
    text = path.read_text(encoding="utf-8") if path.exists() else "# RESULTS\n"
    block = f"{BEGIN}\n{body}\n{END}"
    if BEGIN in text:
        text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step0.yaml")
    ap.add_argument("--run-name", default="zeroshot")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-green", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="score incomplete prediction files (marked PARTIAL; never for final numbers)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    sc = cfg["scoring"]
    suffix = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step0", args.run_name + suffix)
    eval_ids = json.loads((data_dir(cfg) / f"step0_eval_ids{suffix}.json").read_text(encoding="utf-8"))

    if args.smoke:
        mcfg = {"default": sc["smoke_metrics"], "per_sample": [m for m in sc["per_sample_metrics"]
                                                               if m in sc["smoke_metrics"]], "green": False}
    else:
        mcfg = {"default": sc["default_metrics"], "per_sample": sc["per_sample_metrics"],
                "green": bool(sc["green"]) and not args.no_green}
    write_run_info(rdir, cfg, args, {"step": "step0_score", "metrics": mcfg})

    # Collect prediction files and check completeness against the fixed eval ids.
    preds: dict[tuple[str, str], list[dict]] = {}
    for p in sorted(rdir.glob("*/*/predictions.jsonl")):
        model, ds = p.parent.parent.name, p.parent.name
        rows = {r["study_id"]: r for r in read_jsonl(p)}
        want = eval_ids.get(ds, [])
        have = [rows[i] for i in want if i in rows]
        if len(have) < len(want):
            msg = f"[{model}/{ds}] {len(have)}/{len(want)} predictions"
            if not args.allow_partial:
                print(f"SKIP {msg} - incomplete (finish generation, or --allow-partial).")
                continue
            print(f"PARTIAL {msg}")
        preds[(model, ds)] = have
    if not preds:
        raise SystemExit(f"No complete prediction files under {rdir}")

    scorer = None
    entries, per_sample_all = [], {}
    for (model, ds), rows in preds.items():
        out = rdir / model / ds
        digest = _digest(rows, mcfg)
        mpath = out / "metrics.json"
        cached = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else None
        if cached and cached.get("digest") == digest:
            print(f"[{model}/{ds}] metrics up to date (cached).")
            result = cached
            per = {t: {} for t in sc["targets"]}
            for r in read_jsonl(out / "per_sample.jsonl"):
                for t in sc["targets"]:
                    for m, v in r.get(t, {}).items():
                        per[t].setdefault(m, []).append(v)
        else:
            if scorer is None:
                scorer = Scorer(mcfg["default"], mcfg["per_sample"], mcfg["green"])
            result = {"model": model, "dataset": ds, "n": len(rows), "digest": digest, "metrics_cfg": mcfg,
                      "partial": len(rows) < len(eval_ids[ds]),
                      "dtype": sorted({r["dtype"] for r in rows}),
                      "model_id": sorted({r["model_id"] for r in rows}),
                      "parse_ok_rate": sum(r["parse_ok"] for r in rows) / len(rows),
                      "hit_max_tokens_rate": sum(r["hit_max_tokens"] for r in rows) / len(rows),
                      "empty_hyp_findings": sum(not r["hyp_findings"].strip() for r in rows),
                      "targets": {}}
            per = {}
            for t in sc["targets"]:
                refs, hyps = target_texts(rows, t)
                print(f"[{model}/{ds}/{t}] scoring {len(rows)} studies ...", flush=True)
                agg, ps = scorer.score(refs, hyps)
                result["targets"][t] = agg
                per[t] = ps
            mpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
            with open(out / "per_sample.jsonl", "w", encoding="utf-8") as f:
                for i, r in enumerate(rows):
                    f.write(json.dumps({"study_id": r["study_id"],
                                        **{t: {m: v[i] for m, v in per[t].items()} for t in per}}) + "\n")
        per_sample_all[(model, ds)] = per
        for t in sc["targets"]:
            entries.append({"system": f"zero-shot {model}" + (" (PARTIAL)" if result["partial"] else ""),
                            "dataset": ds, "target": t, "n": result["n"], "dtype": ",".join(result["dtype"]),
                            "metrics": result["targets"][t], "parse_ok_rate": result["parse_ok_rate"]})

    # Paired test: MedGemma 1 (baseline) vs MedGemma 1.5 on the same studies, primary target = findings.
    sig = {}
    models = sorted({m for m, _ in preds})
    if {"medgemma-1", "medgemma-1.5"} <= set(models):
        for ds in sorted({d for _, d in preds}):
            a, b = preds.get(("medgemma-1", ds)), preds.get(("medgemma-1.5", ds))
            if not a or not b or [r["study_id"] for r in a] != [r["study_id"] for r in b]:
                continue
            refs, ha = target_texts(a, "findings")
            _, hb = target_texts(b, "findings")
            print(f"\n[{ds}] paired test medgemma-1 vs medgemma-1.5 (findings)")
            sig[ds] = paired_test({"medgemma-1": ha, "medgemma-1.5": hb},
                                  {"medgemma-1": per_sample_all[("medgemma-1", ds)]["findings"],
                                   "medgemma-1.5": per_sample_all[("medgemma-1.5", ds)]["findings"]},
                                  refs, 1000 if args.smoke else sc["significance_n_samples"],
                                  sc["significance_level"], seed)
        (rdir / "significance.json").write_text(json.dumps(sig, indent=2, default=str), encoding="utf-8")

    table = results_table(sorted(entries, key=lambda e: (e["dataset"], e["target"], e["system"])))
    sig_lines = []
    for ds, s in sig.items():
        sc15 = s["scores"].get("medgemma-1.5", {})
        ps = ", ".join(f"{k.removesuffix('_pvalue')} p={v:.4f}" for k, v in sc15.items() if k.endswith("_pvalue"))
        sig_lines.append(f"- {ds}: medgemma-1.5 vs medgemma-1 (findings, paired AR, {s['n_samples']} trials): {ps}")
    min_n = cfg["eval"]["min_test_studies"]
    small = sorted({e["dataset"] for e in entries if e["n"] < min_n})
    body = "\n".join([
        f"## Step 0 — zero-shot baselines{' (SMOKE: not results)' if args.smoke else ''}",
        f"Run: `{rdir}`. Scores x100 except RadCliQ-v1. Primary target = FINDINGS.",
        "", table, "",
        "Paired significance (RadGraph/RaTEScore/GREEN only; CheXbert/BLEU are corpus-level):" if sig_lines else "",
        *sig_lines,
        "" if not small else f"\n**Below the {min_n}-study acceptance threshold:** {', '.join(small)}",
    ])
    (rdir / "results_table.md").write_text(body + "\n", encoding="utf-8")
    if not args.smoke:
        update_results_md(project_root(cfg) / "outputs" / "RESULTS.md", body)
    print("\n" + body)


if __name__ == "__main__":
    main()
