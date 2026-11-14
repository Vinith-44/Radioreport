"""Step 1c: score Model A and test it against zero-shot MedGemma 1.5 on the SAME test studies.

  python scripts/step1_score.py --run-name a1 --baseline-run zeroshot --smoke
  python scripts/step1_score.py --run-name a1 --baseline-run zeroshot

Reads   outputs/runs/step1/<run>/model-a/<dataset>/predictions.jsonl
        outputs/runs/step0/<baseline-run>/{medgemma-1.5,medgemma-1}/<dataset>/predictions.jsonl
Writes  metrics.json / per_sample.jsonl next to each predictions file (cached by digest),
        outputs/runs/step1/<run>/significance.json, results_table.md, acceptance.json,
        <project_root>/outputs/RESULTS.md (Step 1 block; not in smoke mode).
Acceptance (CLAUDE.md Step 1): RadGraph F1 of Model A > zero-shot MedGemma 1.5 with a significant paired test,
on >= eval.min_test_studies real test studies per dataset (primary target = FINDINGS).
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

RADGRAPH = "radgraph_partial"
LOW_RADGRAPH = 0.24   # CLAUDE.md: "If RadGraph F1 < ~24: add lateral view + indication + more data before RL"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step1.yaml")
    ap.add_argument("--run-name", default="a1")
    ap.add_argument("--baseline-run", default="zeroshot", help="Step 0 run name holding the zero-shot predictions")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-green", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="score an incomplete Model A file (marked PARTIAL; never for final numbers)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    sc = cfg["scoring"]
    suffix = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step1", args.run_name + suffix)
    bdir = run_dir(cfg, "step0", args.baseline_run + suffix)
    eval_ids = json.loads((data_dir(cfg) / f"step0_eval_ids{suffix}.json").read_text(encoding="utf-8"))
    mcfg = metrics_config(sc, args.smoke, args.no_green)
    write_run_info(rdir, cfg, args, {"step": "step1_score", "metrics": mcfg, "baseline_dir": str(bdir)})

    sys_a, base = cfg["model_a"]["system"], cfg["model_a"]["base"]
    systems = {  # name -> (folder, predictions)
        sys_a: (rdir / sys_a, collect(rdir / sys_a, eval_ids, args.allow_partial)),
        base: (bdir / base, collect(bdir / base, eval_ids, False)),
    }
    if (bdir / "medgemma-1").exists():
        systems["medgemma-1"] = (bdir / "medgemma-1", collect(bdir / "medgemma-1", eval_ids, False))
    if not systems[sys_a][1]:
        raise SystemExit(f"No Model A predictions under {rdir / sys_a} (run scripts/step1_generate.py).")
    if not systems[base][1]:
        raise SystemExit(f"No complete zero-shot {base} predictions under {bdir / base} (finish Step 0 first).")

    scorer = LazyScorer(mcfg)
    results, per = {}, {}
    entries = []
    for name, (folder, preds) in systems.items():
        for ds, rows in preds.items():
            res, ps = score_system(folder / ds, name, ds, rows, len(eval_ids[ds]), mcfg, sc["targets"], scorer)
            results[(name, ds)], per[(name, ds)] = res, ps
            label = name if name == sys_a else f"zero-shot {name}"
            for t in sc["targets"]:
                entries.append({"system": label + (" (PARTIAL)" if res["partial"] else ""), "dataset": ds,
                                "target": t, "n": res["n"], "dtype": ",".join(res["dtype"]),
                                "metrics": res["targets"][t], "parse_ok_rate": res["parse_ok_rate"]})

    sig, accept = {}, {}
    alpha = float(sc["significance_level"])
    min_n = int(cfg["eval"]["min_test_studies"])
    for ds in sorted(systems[sys_a][1]):
        if ds not in systems[base][1]:
            accept[ds] = {"accepted": False, "reason": f"no zero-shot {base} predictions for {ds}"}
            continue
        print(f"\n[{ds}] paired test {base} (baseline) vs {sys_a} (findings)")
        s = paired(base, systems[base][1][ds], per[(base, ds)], sys_a, systems[sys_a][1][ds], per[(sys_a, ds)],
                   1000 if args.smoke else int(sc["significance_n_samples"]), alpha, seed)
        if s is None:
            accept[ds] = {"accepted": False, "reason": "Model A and baseline cover different studies"}
            continue
        sig[ds] = s
        rg_a = results[(sys_a, ds)]["targets"]["findings"].get(RADGRAPH)
        rg_b = results[(base, ds)]["targets"]["findings"].get(RADGRAPH)
        p = pvalues(s, sys_a).get(RADGRAPH)
        n = results[(sys_a, ds)]["n"]
        ok = (rg_a is not None and rg_b is not None and p is not None and rg_a > rg_b and p < alpha
              and n >= min_n and not results[(sys_a, ds)]["partial"])
        why = [m for c, m in [(rg_a is None or rg_b is None or rg_a <= rg_b, "RadGraph F1 not higher"),
                              (p is None or p >= alpha, f"not significant (alpha {alpha})"),
                              (n < min_n, f"n < {min_n} studies"),
                              (results[(sys_a, ds)]["partial"], "partial predictions")] if c]
        accept[ds] = {"accepted": bool(ok), "radgraph_model_a": rg_a, "radgraph_zeroshot": rg_b, "p_value": p,
                      "n": n, "why_not": why, "low_radgraph_rule": rg_a is not None and rg_a < LOW_RADGRAPH}
    (rdir / "significance.json").write_text(json.dumps(sig, indent=2, default=str), encoding="utf-8")
    (rdir / "acceptance.json").write_text(json.dumps(accept, indent=2), encoding="utf-8")

    table = results_table(sorted(entries, key=lambda e: (e["dataset"], e["target"], e["system"])))
    lines = []
    for ds, a in accept.items():
        if "p_value" not in a:
            lines.append(f"- {ds}: not evaluated - {a['reason']}")
            continue
        verdict = "PASS" if a["accepted"] else "NOT YET"
        pct = lambda v: "-" if v is None else f"{100 * v:.2f}"  # noqa: E731
        pv = "-" if a["p_value"] is None else f"{a['p_value']:.4f}"
        lines.append(f"- {ds}: RadGraph F1 (findings) {pct(a['radgraph_zeroshot'])} -> "
                     f"{pct(a['radgraph_model_a'])}, paired p={pv}, n={a['n']} -> **{verdict}**"
                     + (f" ({'; '.join(a['why_not'])})" if a["why_not"] else "")
                     + ("  (below ~24: add lateral view + indication + more data before RL)"
                        if a["low_radgraph_rule"] else ""))
    sig_lines = [f"- {ds}: {sys_a} vs zero-shot {base} (findings, paired AR, {s['n_samples']} trials): "
                 + ", ".join(f"{k} p={v:.4f}" for k, v in pvalues(s, sys_a).items()) for ds, s in sig.items()]
    body = "\n".join([
        f"## Step 1 — Model A (QLoRA MedGemma 1.5){' (SMOKE: not results)' if args.smoke else ''}",
        f"Run: `{rdir}`; baseline: `{bdir}`. Scores x100 except RadCliQ-v1. Primary target = FINDINGS.",
        "", table, "",
        "Acceptance (RadGraph F1 beats zero-shot MedGemma 1.5, significant paired test):", *lines, "",
        "Paired significance (RadGraph/RaTEScore/GREEN only; CheXbert/BLEU are corpus-level):" if sig_lines else "",
        *sig_lines,
    ])
    (rdir / "results_table.md").write_text(body + "\n", encoding="utf-8")
    if not args.smoke:
        update_results_md(project_root(cfg) / "outputs" / "RESULTS.md", body, "STEP1")
    print("\n" + body)


if __name__ == "__main__":
    main()
