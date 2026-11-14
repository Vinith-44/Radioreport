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
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import data_dir, load_config, project_root, run_dir, set_seed, write_run_info  # noqa: E402
from src.eval.radeval_wrapper import results_table  # noqa: E402
from src.eval.scoring import (LazyScorer, collect, metrics_config, paired, pvalues,  # noqa: E402
                              score_system, update_results_md)


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

    mcfg = metrics_config(sc, args.smoke, args.no_green)
    write_run_info(rdir, cfg, args, {"step": "step0_score", "metrics": mcfg})

    # Collect prediction files and check completeness against the fixed eval ids.
    preds: dict[tuple[str, str], list[dict]] = {}
    for sys_dir in sorted(p for p in rdir.iterdir() if p.is_dir()):
        for ds, rows in collect(sys_dir, eval_ids, args.allow_partial).items():
            preds[(sys_dir.name, ds)] = rows
    if not preds:
        raise SystemExit(f"No complete prediction files under {rdir}")

    scorer = LazyScorer(mcfg)
    entries, per_sample_all = [], {}
    for (model, ds), rows in preds.items():
        result, per = score_system(rdir / model / ds, model, ds, rows, len(eval_ids[ds]), mcfg, sc["targets"], scorer)
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
            if not a or not b:
                continue
            print(f"\n[{ds}] paired test medgemma-1 vs medgemma-1.5 (findings)")
            r = paired("medgemma-1", a, per_sample_all[("medgemma-1", ds)], "medgemma-1.5", b,
                       per_sample_all[("medgemma-1.5", ds)], 1000 if args.smoke else sc["significance_n_samples"],
                       sc["significance_level"], seed)
            if r:
                sig[ds] = r
        (rdir / "significance.json").write_text(json.dumps(sig, indent=2, default=str), encoding="utf-8")

    table = results_table(sorted(entries, key=lambda e: (e["dataset"], e["target"], e["system"])))
    sig_lines = []
    for ds, s in sig.items():
        ps = ", ".join(f"{k} p={v:.4f}" for k, v in pvalues(s, "medgemma-1.5").items())
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
        update_results_md(project_root(cfg) / "outputs" / "RESULTS.md", body, "STEP0")
    print("\n" + body)


if __name__ == "__main__":
    main()
