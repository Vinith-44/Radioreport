"""Step 3d: clinical metrics before vs after dropping FLAGGED sentences (paired) + the verifier's catch rates.

  python scripts/step3_score.py --run-name v1 [--smoke]

Arms (same studies, verify_ids.json): outputs/runs/step3/<run>/source (the verified system's reports, exact copy)
vs <source>+verified-drop. Scores of the source arm are copied from its original folder when the digest matches
(no re-scoring). Step 3 acceptance (CLAUDE.md): report (1) the share of clinically wrong sentences that land in
FLAGGED/UNCERTAIN and (2) clinical metrics after dropping FLAGGED sentences -> outputs/RESULTS.md (STEP3 block).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config, project_root, run_dir, set_seed, write_run_info  # noqa: E402
from src.eval.radeval_wrapper import results_table  # noqa: E402
from src.eval.scoring import (LazyScorer, collect, digest, metrics_config, paired, pvalues,  # noqa: E402
                              score_system, update_results_md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step3.yaml")
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-green", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    sc, vc = cfg["scoring"], cfg["verify"]
    sfx = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step3", args.run_name + sfx)
    ids = json.loads((rdir / "verify_ids.json").read_text(encoding="utf-8"))
    mcfg = metrics_config(sc, args.smoke, args.no_green)
    write_run_info(rdir, cfg, args, {"step": "step3_score", "metrics": mcfg})
    src = vc["source"]
    orig_dir = run_dir(cfg, src["step"], src["run"] + sfx) / src["system"]
    drop_name = f"{src['system']}+{vc['drop_system']}"

    arms = {"source": collect(rdir / "source", ids, False), drop_name: collect(rdir / drop_name, ids, False)}
    # reuse the source system's existing scores when they cover exactly these rows
    for ds, rows in arms["source"].items():
        om = orig_dir / ds / "metrics.json"
        dst = rdir / "source" / ds
        if om.exists() and json.loads(om.read_text(encoding="utf-8")).get("digest") == digest(rows, mcfg) \
                and not (dst / "metrics.json").exists():
            shutil.copy(om, dst / "metrics.json")
            shutil.copy(orig_dir / ds / "per_sample.jsonl", dst / "per_sample.jsonl")

    scorer = LazyScorer(mcfg)
    res, per, entries = {}, {}, []
    for name, preds in arms.items():
        label = src["system"] if name == "source" else drop_name
        for ds, rows in preds.items():
            r, p = score_system(rdir / name / ds, label, ds, rows, len(ids[ds]), mcfg, sc["targets"], scorer)
            res[(name, ds)], per[(name, ds)] = r, p
            for t in sc["targets"]:
                entries.append({"system": label, "dataset": ds, "target": t, "n": r["n"], "dtype": ",".join(r["dtype"]),
                                "metrics": r["targets"][t], "parse_ok_rate": r["parse_ok_rate"]})

    sig = {}
    for ds in sorted(arms[drop_name]):
        if ds not in arms["source"]:
            continue
        print(f"\n[{ds}] paired test {src['system']} vs {drop_name} (findings)")
        s = paired(src["system"], arms["source"][ds], per[("source", ds)], drop_name, arms[drop_name][ds],
                   per[(drop_name, ds)], 1000 if args.smoke else int(sc["significance_n_samples"]),
                   float(sc["significance_level"]), seed)
        if s:
            sig[ds] = s
    (rdir / "significance.json").write_text(json.dumps(sig, indent=2, default=str), encoding="utf-8")

    ev = json.loads((rdir / "verify_eval.json").read_text(encoding="utf-8"))
    n_drop = {ds: sum(r.get("n_flagged_dropped", 0) for r in rows) for ds, rows in arms[drop_name].items()}
    lines = []
    for ds, e in sorted(ev["per_dataset"].items()):
        pct = lambda k: "-" if e[k] is None else f"{100 * e[k]:.1f}%"  # noqa: E731
        lines.append(f"- {ds}: {e['sentences']} sentences {e['verdicts']}; judged vs reference: {e['wrong']} wrong / "
                     f"{e['correct']} correct. **Wrong sentences caught (FLAGGED or UNCERTAIN): "
                     f"{pct('wrong_caught_rate (FLAGGED or UNCERTAIN)')}** (FLAGGED alone "
                     f"{pct('wrong_flagged_rate (FLAGGED)')}); false alarms (correct but FLAGGED): "
                     f"{pct('false_alarm_rate (correct but FLAGGED)')}; FLAGGED sentences dropped: {n_drop.get(ds, 0)}")
    sig_lines = [f"- {ds}: " + ", ".join(f"{m} p={v:.4f}" for m, v in pvalues(s, drop_name).items())
                 for ds, s in sig.items()]
    auc = ev["probe_test_auroc_vs_reference_chexbert (evaluation only)"]
    body = "\n".join([
        f"## Step 3 — verify-then-report{' (SMOKE: not results)' if args.smoke else ''}",
        f"Run: `{rdir}`; verified system: `{orig_dir}`. Rules: `src/verify/verdict.py`. Target = FINDINGS.",
        "", "Verifier vs reference report (sentence level):", *lines, "",
        "Clinical metrics before / after dropping FLAGGED sentences (same studies):", "",
        results_table(sorted(entries, key=lambda e: (e["dataset"], e["target"], e["system"]))), "",
        f"Paired significance, {drop_name} vs {src['system']} (findings):" if sig_lines else "", *sig_lines, "",
        "Probe test AUROC vs reference CheXbert labels: " + ", ".join(f"{k} {v}" for k, v in auc.items() if v is not None),
    ])
    (rdir / "results_table.md").write_text(body + "\n", encoding="utf-8")
    if not args.smoke:
        update_results_md(project_root(cfg) / "outputs" / "RESULTS.md", body, "STEP3")
    print("\n" + body)


if __name__ == "__main__":
    main()
