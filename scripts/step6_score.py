"""Step 6c: the clean -> photo score drop, with vs without robustness training (CLAUDE.md Step 6 acceptance table).

  python scripts/step6_score.py --run-name e1 [--smoke]

Report test sets (full RadEval, Step 0 test subset): baseline/robust x clean/photo, paired photo-vs-clean tests.
CheXphoto (real phone photos, labels only): CheXbert labels of the generated report vs the CheXpert labels
(positive/uncertain = 1), micro-F1 over the 14 labels and the 5 competition labels, clean vs photo of the SAME
studies. Writes results_table.md, drop.json and the STEP6 block of outputs/RESULTS.md (not in smoke mode).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common import data_dir, load_config, project_root, read_jsonl, run_dir, set_seed, write_run_info  # noqa: E402
from src.eval.radeval_wrapper import TABLE_COLUMNS, results_table  # noqa: E402
from src.eval.scoring import LazyScorer, collect, metrics_config, paired, pvalues, score_system, update_results_md  # noqa: E402

TOP5 = ["Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion"]


def micro_f1(y: np.ndarray, p: np.ndarray) -> float:
    tp = float((y & p).sum())
    return 2 * tp / max(1.0, float(y.sum() + p.sum()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step6.yaml")
    ap.add_argument("--run-name", default="e1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-green", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    sc = cfg["scoring"]
    sfx = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step6", args.run_name + sfx)
    eval_ids = json.loads((data_dir(cfg) / f"step0_eval_ids{sfx}.json").read_text(encoding="utf-8"))
    mcfg = metrics_config(sc, args.smoke, args.no_green)
    write_run_info(rdir, cfg, args, {"step": "step6_score", "metrics": mcfg})

    scorer = LazyScorer(mcfg)
    res, per, preds, entries = {}, {}, {}, []
    for arm in ["baseline-clean", "baseline-photo", "robust-clean", "robust-photo"]:
        preds[arm] = collect(rdir / arm, eval_ids, False)
        for ds, rows in preds[arm].items():
            r, p = score_system(rdir / arm / ds, arm, ds, rows, len(eval_ids[ds]), mcfg, sc["targets"], scorer)
            res[(arm, ds)], per[(arm, ds)] = r, p
            for t in sc["targets"]:
                entries.append({"system": arm, "dataset": ds, "target": t, "n": r["n"], "dtype": ",".join(r["dtype"]),
                                "metrics": r["targets"][t], "parse_ok_rate": r["parse_ok_rate"]})
    drop, sig = {}, {}
    for sysn in ("baseline", "robust"):
        for ds in sorted(set(preds.get(f"{sysn}-clean", {})) & set(preds.get(f"{sysn}-photo", {}))):
            c = res[(f"{sysn}-clean", ds)]["targets"]["findings"]
            ph = res[(f"{sysn}-photo", ds)]["targets"]["findings"]
            drop[f"{sysn}/{ds}"] = {k: round(ph[k] - c[k], 4) for _, k, _ in TABLE_COLUMNS if k in c and k in ph}
            s = paired(f"{sysn}-clean", preds[f"{sysn}-clean"][ds], per[(f"{sysn}-clean", ds)], f"{sysn}-photo",
                       preds[f"{sysn}-photo"][ds], per[(f"{sysn}-photo", ds)],
                       1000 if args.smoke else int(sc["significance_n_samples"]), float(sc["significance_level"]), seed)
            if s:
                sig[f"{sysn}/{ds}"] = pvalues(s, f"{sysn}-photo")

    # CheXphoto label agreement
    cxp = {}
    from src.retrieval.rerank import CHEXBERT_LABELS

    rows_by = {a: read_jsonl(rdir / a / "chexphoto" / "predictions.jsonl")
               for a in ["baseline-chexphoto_clean", "baseline-chexphoto_photo", "robust-chexphoto_clean",
                         "robust-chexphoto_photo"] if (rdir / a / "chexphoto" / "predictions.jsonl").exists()}
    if rows_by:
        from src.data.chexphoto import label_vector
        from src.retrieval.rerank import chexbert_labels

        top = [CHEXBERT_LABELS.index(n) for n in TOP5]
        for arm, rows in rows_by.items():
            y = np.asarray([label_vector(r["labels"], CHEXBERT_LABELS) for r in rows], dtype=bool)
            p = chexbert_labels([r["hypothesis"] for r in rows]).astype(bool)
            cxp[arm] = {"n": len(rows), "micro_f1_14": round(micro_f1(y, p), 4),
                        "micro_f1_5": round(micro_f1(y[:, top], p[:, top]), 4)}
    (rdir / "drop.json").write_text(json.dumps({"report_sets_photo_minus_clean": drop, "paired_p": sig,
                                                "chexphoto_label_f1": cxp}, indent=2), encoding="utf-8")
    hdr = "| system | dataset | " + " | ".join(h for h, k, _ in TABLE_COLUMNS) + " |"
    dl = [hdr, "|" + "---|" * (len(TABLE_COLUMNS) + 2)]
    for k, d in drop.items():
        sysn, ds = k.split("/")
        dl.append(f"| {sysn} | {ds} | " + " | ".join("-" if c not in d else (f"{100 * d[c]:+.2f}" if s != 1 else
                                                                             f"{d[c]:+.4f}") for _, c, s in TABLE_COLUMNS) + " |")
    cl = ["| arm | n | CheXbert-label micro-F1 (14) | micro-F1 (5) |", "|---|---|---|---|"] + \
        [f"| {a} | {v['n']} | {100 * v['micro_f1_14']:.2f} | {100 * v['micro_f1_5']:.2f} |" for a, v in cxp.items()]
    body = "\n".join([
        f"## Step 6 — robustness to phone photos / WhatsApp{' (SMOKE: not results)' if args.smoke else ''}",
        f"Run: `{rdir}`. baseline = Step 5 winner; robust = same recipe + 40% photo-augmented training.", "",
        "Score change photo - clean (findings; negative = drop):", *dl, "",
        "Paired photo vs clean p-values: " + "; ".join(f"{k}: " + ", ".join(f"{m} p={v:.4f}" for m, v in p_.items())
                                                       for k, p_ in sig.items()), "",
        "CheXphoto (real phone photos of CheXpert images; labels only, training patients excluded; UNOFFICIAL copy):",
        *(cl if cxp else ["(not evaluated)"]), "",
        results_table(sorted(entries, key=lambda e: (e["dataset"], e["target"], e["system"])))])
    (rdir / "results_table.md").write_text(body + "\n", encoding="utf-8")
    if not args.smoke:
        update_results_md(project_root(cfg) / "outputs" / "RESULTS.md", body, "STEP6")
    print("\n" + body)


if __name__ == "__main__":
    main()
