"""Step 9b: GRPO model vs its SFT source on the same test studies (generation + paired tests + acceptance).

  python scripts/step9_score.py --run-name g1 [--smoke]

Generates the GRPO model's test predictions (scripts/generate_system.py) if missing, scores both arms with RadEval,
and runs paired approximate-randomisation tests on RadGraph / RaTEScore / GREEN (mean of per-sample scores) AND on
CheXbert micro-F1 (14 labels), which is corpus-level but exactly recomputable from cached per-study label vectors.
Acceptance (configs/step9.yaml): significant gain in CheXbert F1 or GREEN, RadGraph point estimate not lower.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from src.common import data_dir, load_config, project_root, run_dir, set_seed, write_run_info  # noqa: E402
from src.eval.radeval_wrapper import SIG_METRICS, cached_metric, results_table, target_texts  # noqa: E402
from src.eval.scoring import LazyScorer, collect, metrics_config, score_system, update_results_md  # noqa: E402

CX = "f1chexbert_all_micro_f1"


def cached_micro_f1(ref_lab: np.ndarray, hyp_lab: dict[str, np.ndarray], hyps: dict[str, list[str]]):
    """compare_systems metric callable: micro-F1 of the label vectors cached per (index, hypothesis)."""
    cache = {}
    for s, hs in hyps.items():
        for i, h in enumerate(hs):
            cache[(i, h)] = hyp_lab[s][i].astype(bool)
    R = ref_lab.astype(bool)

    def fn(hs: list[str], refs: list[str]) -> float:
        H = np.stack([cache[(i, h)] for i, h in enumerate(hs)])
        tp = float((H & R).sum())
        return 2 * tp / max(1.0, float(H.sum() + R.sum()))
    return fn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step9.yaml")
    ap.add_argument("--run-name", default="g1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-green", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    sc = cfg["scoring"]
    sfx = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step9", args.run_name + sfx)
    meta = json.loads((rdir / "grpo_meta.json").read_text(encoding="utf-8"))
    src = meta["source"]
    from src.models.systems import spec_dir

    sft_dir = spec_dir(cfg, src, args.smoke) / src["system"]
    gname = src["system"] + cfg["grpo_eval"]["system_suffix"]
    from generate_system import generate_for_spec

    generate_for_spec(cfg, {"type": src["type"], "step": "step9", "run": args.run_name, "system": gname},
                      rdir / gname, args.smoke)
    eval_ids = json.loads((data_dir(cfg) / f"step0_eval_ids{sfx}.json").read_text(encoding="utf-8"))
    mcfg = metrics_config(sc, args.smoke, args.no_green)
    write_run_info(rdir, cfg, args, {"step": "step9_score", "metrics": mcfg})
    arms = {"sft": (sft_dir, collect(sft_dir, eval_ids, False)), "grpo": (rdir / gname, collect(rdir / gname, eval_ids, False))}
    scorer = LazyScorer(mcfg)
    res, per, entries = {}, {}, []
    for k, (folder, preds) in arms.items():
        for ds, rows in preds.items():
            r, p = score_system(folder / ds, k, ds, rows, len(eval_ids[ds]), mcfg, sc["targets"], scorer)
            res[(k, ds)], per[(k, ds)] = r, p
            for t in sc["targets"]:
                entries.append({"system": f"{src['system']} ({k.upper()})", "dataset": ds, "target": t, "n": r["n"],
                                "dtype": ",".join(r["dtype"]), "metrics": r["targets"][t],
                                "parse_ok_rate": r["parse_ok_rate"]})

    from radeval import compare_systems

    from src.retrieval.rerank import chexbert_labels

    alpha = float(sc["significance_level"])
    accept, sig = {}, {}
    for ds in sorted(set(arms["sft"][1]) & set(arms["grpo"][1])):
        a, b = arms["sft"][1][ds], arms["grpo"][1][ds]
        assert [r["study_id"] for r in a] == [r["study_id"] for r in b]
        refs, ha = target_texts(a, "findings")
        _, hb = target_texts(b, "findings")
        lab = chexbert_labels(refs + ha + hb)
        n = len(refs)
        systems = {"sft": ha, "grpo": hb}
        metrics = {m: cached_metric({s: per[(s, ds)]["findings"][m] for s in systems}, systems)
                   for m in SIG_METRICS if all(m in per[(s, ds)]["findings"] for s in systems)}
        metrics[CX] = cached_micro_f1(lab[:n], {"sft": lab[n:2 * n], "grpo": lab[2 * n:]}, systems)
        _, scores = compare_systems(systems=systems, metrics=metrics, references=refs,
                                    n_samples=1000 if args.smoke else int(sc["significance_n_samples"]),
                                    significance_level=alpha, seed=seed, print_results=True)
        pv = {k.removesuffix("_pvalue"): float(v) for k, v in scores.get("grpo", {}).items() if k.endswith("_pvalue")}
        fa, fb = res[("sft", ds)]["targets"]["findings"], res[("grpo", ds)]["targets"]["findings"]
        cx_a, cx_b = metrics[CX](ha, refs), metrics[CX](hb, refs)
        gain_cx = cx_b > cx_a and pv.get(CX, 1) < alpha
        gain_green = "green" in fa and fb.get("green", -1) > fa["green"] and pv.get("green", 1) < alpha
        rg_ok = fb.get("radgraph_partial", -1) >= fa.get("radgraph_partial", 0)
        accept[ds] = {"accepted": bool((gain_cx or gain_green) and rg_ok), "chexbert_micro_f1": [cx_a, cx_b],
                      "green": [fa.get("green"), fb.get("green")], "radgraph": [fa.get("radgraph_partial"),
                                                                              fb.get("radgraph_partial")],
                      "p_values": pv}
        sig[ds] = pv
    (rdir / "acceptance.json").write_text(json.dumps(accept, indent=2), encoding="utf-8")
    fmt = lambda v: "-" if v is None else f"{100 * v:.2f}"  # noqa: E731
    lines = [f"- {ds}: CheXbert-14 micro-F1 {fmt(a['chexbert_micro_f1'][0])} -> {fmt(a['chexbert_micro_f1'][1])} "
             f"(p={a['p_values'].get(CX, float('nan')):.4f}); GREEN {fmt(a['green'][0])} -> {fmt(a['green'][1])}; "
             f"RadGraph {fmt(a['radgraph'][0])} -> {fmt(a['radgraph'][1])} -> **{'PASS' if a['accepted'] else 'NOT YET'}**"
             for ds, a in accept.items()]
    body = "\n".join([
        f"## Step 9 — GRPO with multiple rewards{' (SMOKE: not results)' if args.smoke else ''}",
        f"SFT: `{sft_dir}`; GRPO: `{rdir / gname}` ({meta['steps']} steps). Target = FINDINGS.", "",
        results_table(sorted(entries, key=lambda e: (e["dataset"], e["target"], e["system"]))), "",
        "Acceptance (significant gain in CheXbert-14 F1 or GREEN, RadGraph not lower):", *lines])
    (rdir / "results_table.md").write_text(body + "\n", encoding="utf-8")
    if not args.smoke:
        update_results_md(project_root(cfg) / "outputs" / "RESULTS.md", body, "STEP9")
    print("\n" + body)


if __name__ == "__main__":
    main()
