"""Step 2f: retrieval ablation on Model A - with vs without retrieved references, same test studies, paired.

  python scripts/step2_score.py --run-name r1 [--smoke]

Baseline arm  = outputs/runs/step1/<retrieval.step1_run>/model-a          (Step 1 predictions, no retrieval)
Other arms    = outputs/runs/step2/<run>/model-a+retrieval[-sim]           (step2_generate.py)
Decision per dataset with the pre-registered rule in configs/step2.yaml (ablation): KEEP retrieval only if no
paired-tested clinical metric (RadGraph, RaTEScore, GREEN) is significantly lower and corpus-level CheXbert
micro-F1 (5 / 14) is not lower by more than ablation.max_chexbert_drop. Writes ablation.json, results_table.md,
and the Step 2 block of outputs/RESULTS.md (not in smoke mode).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import data_dir, load_config, project_root, run_dir, set_seed, write_run_info  # noqa: E402
from src.eval.radeval_wrapper import SIG_METRICS, results_table  # noqa: E402
from src.eval.scoring import (LazyScorer, collect, metrics_config, paired, pvalues,  # noqa: E402
                              score_system, update_results_md)

CHEXBERT = ("f1chexbert_5_micro_f1", "f1chexbert_all_micro_f1")


def decide(base: dict, cand: dict, pv: dict, alpha: float, max_drop: float) -> tuple[bool, list[str]]:
    """base/cand: findings-target metrics; pv: paired p-values of cand vs base. Returns (keep, reasons)."""
    why = []
    for m in sorted(SIG_METRICS):
        if m in base and m in cand and m in pv and cand[m] < base[m] and pv[m] < alpha:
            why.append(f"{m} significantly lower ({100 * base[m]:.2f} -> {100 * cand[m]:.2f}, p={pv[m]:.4f})")
    for m in CHEXBERT:
        if m in base and m in cand and base[m] - cand[m] > max_drop:
            why.append(f"{m} lower by {100 * (base[m] - cand[m]):.2f} points (> {100 * max_drop:.1f})")
    return not why, why


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2.yaml")
    ap.add_argument("--run-name", default="r1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-green", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    sc = cfg["scoring"]
    sfx = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step2", args.run_name + sfx)
    s1dir = run_dir(cfg, "step1", cfg["retrieval"]["step1_run"] + sfx)
    eval_ids = json.loads((data_dir(cfg) / f"step0_eval_ids{sfx}.json").read_text(encoding="utf-8"))
    mcfg = metrics_config(sc, args.smoke, args.no_green)
    write_run_info(rdir, cfg, args, {"step": "step2_score", "metrics": mcfg, "baseline_dir": str(s1dir)})

    base_name = cfg["model_a"]["system"]
    arms = {base_name: (s1dir / base_name, collect(s1dir / base_name, eval_ids, False))}
    for d in sorted(rdir.glob(cfg["ablation"]["system"] + "*")):
        if d.is_dir():
            arms[d.name] = (d, collect(d, eval_ids, args.allow_partial))
    if not arms[base_name][1]:
        raise SystemExit(f"No complete Step 1 predictions under {s1dir / base_name}.")
    if len(arms) < 2:
        raise SystemExit(f"No retrieval predictions under {rdir} (run scripts/step2_generate.py).")

    scorer = LazyScorer(mcfg)
    res, per, entries = {}, {}, []
    for name, (folder, preds) in arms.items():
        for ds, rows in preds.items():
            r, p = score_system(folder / ds, name, ds, rows, len(eval_ids[ds]), mcfg, sc["targets"], scorer)
            res[(name, ds)], per[(name, ds)] = r, p
            for t in sc["targets"]:
                entries.append({"system": name + (" (PARTIAL)" if r["partial"] else ""), "dataset": ds, "target": t,
                                "n": r["n"], "dtype": ",".join(r["dtype"]), "metrics": r["targets"][t],
                                "parse_ok_rate": r["parse_ok_rate"]})

    alpha, max_drop = float(sc["significance_level"]), float(cfg["ablation"]["max_chexbert_drop"])
    sig, decisions = {}, {}
    for name, (_, preds) in arms.items():
        if name == base_name:
            continue
        for ds in sorted(preds):
            if ds not in arms[base_name][1]:
                continue
            print(f"\n[{ds}] paired test {base_name} (no retrieval) vs {name} (findings)")
            s = paired(base_name, arms[base_name][1][ds], per[(base_name, ds)], name, preds[ds], per[(name, ds)],
                       1000 if args.smoke else int(sc["significance_n_samples"]), alpha, seed)
            if s is None:
                decisions[f"{name}/{ds}"] = {"keep": False, "why": ["different studies than the baseline"]}
                continue
            sig[f"{name}/{ds}"] = s
            keep, why = decide(res[(base_name, ds)]["targets"]["findings"], res[(name, ds)]["targets"]["findings"],
                               pvalues(s, name), alpha, max_drop)
            if res[(name, ds)]["partial"]:
                keep, why = False, why + ["partial predictions"]
            decisions[f"{name}/{ds}"] = {"keep": keep, "why": why, "p_values": pvalues(s, name)}
    (rdir / "significance.json").write_text(json.dumps(sig, indent=2, default=str), encoding="utf-8")
    (rdir / "ablation.json").write_text(json.dumps(decisions, indent=2), encoding="utf-8")

    table = results_table(sorted(entries, key=lambda e: (e["dataset"], e["target"], e["system"])))
    lines = [f"- {k}: **{'KEEP' if d['keep'] else 'DROP'}**" + (f" ({'; '.join(d['why'])})" if d["why"] else
                                                               " (no clinical metric hurt)")
             for k, d in decisions.items()]
    sig_lines = [f"- {k}: " + ", ".join(f"{m} p={v:.4f}" for m, v in pvalues(s, k.split('/')[0]).items())
                 for k, s in sig.items()]
    body = "\n".join([
        f"## Step 2 — retrieval ablation on Model A{' (SMOKE: not results)' if args.smoke else ''}",
        f"Run: `{rdir}`; no-retrieval arm: `{s1dir / base_name}`. Scores x100 except RadCliQ-v1. Target = FINDINGS.",
        "", table, "",
        "Decision (pre-registered rule, configs/step2.yaml `ablation`):", *lines, "",
        f"Paired significance vs {base_name} (findings):" if sig_lines else "", *sig_lines,
    ])
    (rdir / "results_table.md").write_text(body + "\n", encoding="utf-8")
    if not args.smoke:
        update_results_md(project_root(cfg) / "outputs" / "RESULTS.md", body, "STEP2")
    print("\n" + body)


if __name__ == "__main__":
    main()
