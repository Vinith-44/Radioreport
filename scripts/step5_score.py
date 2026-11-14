"""Step 5c: full RadEval comparison Model B vs Model A (same test studies, paired) + the winner decision.

  python scripts/step5_score.py --run-name b2 [--smoke]

Arms: Model A = compare.model_a (Step 1 predictions), Model B = outputs/runs/step5/<run>/model-b. Winner rule
(configs/step5.yaml compare, fixed before scoring): higher mean RadGraph F1 (findings) over datasets; within
tie_points -> higher mean CheXbert-14 micro F1. Writes winner.json (read by Steps 6-10), results_table.md, and
the STEP5 block of outputs/RESULTS.md (not in smoke mode).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common import data_dir, load_config, project_root, run_dir, set_seed, write_run_info  # noqa: E402
from src.eval.radeval_wrapper import results_table  # noqa: E402
from src.eval.scoring import (LazyScorer, collect, metrics_config, paired, pvalues,  # noqa: E402
                              score_system, update_results_md)


def pick_winner(a: dict, b: dict, tie: float) -> tuple[str, str]:
    """a/b: {dataset: findings metrics}. Returns (winner key 'model-a'|'model-b', reason)."""
    ds = sorted(set(a) & set(b))
    if not ds:
        raise ValueError("no dataset scored for both models")
    rg = lambda m: float(np.mean([m[d].get("radgraph_partial", np.nan) for d in ds]))  # noqa: E731
    cx = lambda m: float(np.mean([m[d].get("f1chexbert_all_micro_f1", np.nan) for d in ds]))  # noqa: E731
    ra, rb = rg(a), rg(b)
    if abs(ra - rb) >= tie:
        w = "model-a" if ra > rb else "model-b"
        return w, f"mean RadGraph F1 {100 * ra:.2f} (A) vs {100 * rb:.2f} (B) over {ds}"
    ca, cb = cx(a), cx(b)
    w = "model-a" if ca >= cb else "model-b"
    return w, f"RadGraph tie ({100 * ra:.2f} vs {100 * rb:.2f}); CheXbert-14 {100 * ca:.2f} (A) vs {100 * cb:.2f} (B)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step5.yaml")
    ap.add_argument("--run-name", default="b2")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-green", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    sc, cmp_ = cfg["scoring"], cfg["compare"]
    sfx = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step5", args.run_name + sfx)
    eval_ids = json.loads((data_dir(cfg) / f"step0_eval_ids{sfx}.json").read_text(encoding="utf-8"))
    mcfg = metrics_config(sc, args.smoke, args.no_green)
    write_run_info(rdir, cfg, args, {"step": "step5_score", "metrics": mcfg})
    ma = cmp_["model_a"]
    a_dir = run_dir(cfg, ma["step"], ma["run"] + sfx) / ma["system"]
    b_name = cfg["model_b_eval"]["system"]
    arms = {"model-a": (a_dir, collect(a_dir, eval_ids, False)), "model-b": (rdir / b_name, collect(rdir / b_name, eval_ids, False))}
    for k, (d, p) in arms.items():
        if not p:
            raise SystemExit(f"no complete {k} predictions under {d}")

    scorer = LazyScorer(mcfg)
    res, per, entries = {}, {}, []
    for key, (folder, preds) in arms.items():
        for ds, rows in preds.items():
            r, p = score_system(folder / ds, key, ds, rows, len(eval_ids[ds]), mcfg, sc["targets"], scorer)
            res[(key, ds)], per[(key, ds)] = r, p
            for t in sc["targets"]:
                entries.append({"system": key, "dataset": ds, "target": t, "n": r["n"], "dtype": ",".join(r["dtype"]),
                                "metrics": r["targets"][t], "parse_ok_rate": r["parse_ok_rate"]})
    sig = {}
    for ds in sorted(set(arms["model-a"][1]) & set(arms["model-b"][1])):
        print(f"\n[{ds}] paired test model-a vs model-b (findings)")
        s = paired("model-a", arms["model-a"][1][ds], per[("model-a", ds)], "model-b", arms["model-b"][1][ds],
                   per[("model-b", ds)], 1000 if args.smoke else int(sc["significance_n_samples"]),
                   float(sc["significance_level"]), seed)
        if s:
            sig[ds] = s
    (rdir / "significance.json").write_text(json.dumps(sig, indent=2, default=str), encoding="utf-8")
    fa = {ds: res[("model-a", ds)]["targets"]["findings"] for ds in arms["model-a"][1]}
    fb = {ds: res[("model-b", ds)]["targets"]["findings"] for ds in arms["model-b"][1]}
    winner, why = pick_winner(fa, fb, float(cmp_["tie_points"]))
    spec = {"model-a": {"type": "a", **ma}, "model-b": {"type": "b", "step": "step5", "run": args.run_name,
                                                         "system": b_name}}
    out = {"winner": winner, "reason": why, "winner_spec": spec[winner], "ablation_spec": spec[
        "model-b" if winner == "model-a" else "model-a"], "smoke": args.smoke}
    (rdir / "winner.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    sig_lines = [f"- {ds}: model-b vs model-a: " + ", ".join(f"{m} p={v:.4f}" for m, v in pvalues(s, "model-b").items())
                 for ds, s in sig.items()]
    body = "\n".join([
        f"## Step 5 — Model B (joint tuning) vs Model A{' (SMOKE: not results)' if args.smoke else ''}",
        f"Model A: `{a_dir}`; Model B: `{rdir / b_name}`. Scores x100 except RadCliQ-v1. Target = FINDINGS.", "",
        results_table(sorted(entries, key=lambda e: (e["dataset"], e["target"], e["system"]))), "",
        f"**Winner (pre-registered rule): {winner}** - {why}. It carries Steps 6-10; the other is an ablation.", "",
        "Paired significance (findings):" if sig_lines else "", *sig_lines])
    (rdir / "results_table.md").write_text(body + "\n", encoding="utf-8")
    if not args.smoke:
        update_results_md(project_root(cfg) / "outputs" / "RESULTS.md", body, "STEP5")
    print("\n" + body)


if __name__ == "__main__":
    main()
