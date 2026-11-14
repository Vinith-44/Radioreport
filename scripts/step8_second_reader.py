"""Step 8: second-reader mode, evaluated by deleting one positive finding from real radiologist reports.

  python scripts/step8_second_reader.py --run-name s1 [--smoke]

Inputs: the Step 2 probe + val/test image embeddings (second_reader.step2_run), the VAL reports (threshold
calibration only) and the TEST reference reports of the fixed Step 0 subset (the "radiologist's report").
For every test report: (a) flags on the UNMODIFIED report (false alarms - or genuine misses, indistinguishable
without a second radiologist), (b) one positive condition deleted -> is it flagged? Precision = flags that are the
deleted condition / all flags on the modified report. Writes second_reader.jsonl, metrics.json, the STEP8 block.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import data_dir, load_config, project_root, read_jsonl, run_dir, set_seed, write_run_info  # noqa: E402
from src.data.text_clean import format_report  # noqa: E402
from src.eval.scoring import update_results_md  # noqa: E402
from src.models.probe import load_probe, predict  # noqa: E402
from src.retrieval.store import ids_sha, load_split  # noqa: E402
from src.verify.claims import split_sentences  # noqa: E402
from src.verify.second_reader import CONDITIONS, calibrate, flag, mentioned, simulate_deletion  # noqa: E402


def sentence_classes(texts: dict[str, str], cache: Path) -> dict[str, tuple[list[str], np.ndarray]]:
    """study -> (sentences, raw CheXbert classes); cached by the exact texts."""
    key = ids_sha(pd.DataFrame({"study_id": [f"{k}\x00{v}" for k, v in sorted(texts.items())]}))
    if cache.exists():
        d = json.loads(cache.read_text(encoding="utf-8"))
        if d.get("key") == key:
            return {k: (v[0], np.asarray(v[1], dtype=np.int8).reshape(-1, 14)) for k, v in d["data"].items()}
    from src.verify.claims import chexbert_classes

    sents = {k: split_sentences(v) for k, v in texts.items()}
    flat = [s for k in sorted(sents) for s in sents[k]]
    cls = chexbert_classes(flat) if flat else np.zeros((0, 14), np.int8)
    out, i = {}, 0
    for k in sorted(sents):
        out[k] = (sents[k], cls[i:i + len(sents[k])])
        i += len(sents[k])
    cache.write_text(json.dumps({"key": key, "data": {k: [s, c.tolist()] for k, (s, c) in out.items()}}),
                     encoding="utf-8")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step8.yaml")
    ap.add_argument("--run-name", default="s1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    sr = cfg["second_reader"]
    sfx = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step8", args.run_name + sfx)
    s2 = run_dir(cfg, "step2", sr["step2_run"] + sfx)
    write_run_info(rdir, cfg, args, {"step": "step8_second_reader", "second_reader": sr})

    probe, pinfo = load_probe(s2 / "probe")
    va_ids, va_emb = load_split(s2, "val")
    te_ids, te_emb = load_split(s2, "test")
    p_va, _ = predict(probe, va_emb, pinfo["thresholds"])
    p_te, _ = predict(probe, te_emb, pinfo["thresholds"])
    test = pd.read_parquet(data_dir(cfg) / f"test{sfx}.parquet")
    test = test.assign(study_id=test["study_id"].astype(str)).set_index("study_id")
    te_text = {str(s): format_report(test.loc[s, "findings"], test.loc[s, "impression"]) for s in te_ids["study_id"]}
    va_text = {str(r["study_id"]): format_report(r["findings"], r["impression"]) for r in va_ids.to_dict("records")}
    if sr.get("max_studies"):
        keep = sorted(te_text)[: int(sr["max_studies"])]
        te_text = {k: te_text[k] for k in keep}
    va_sc = sentence_classes(va_text, rdir / "val_sentence_classes.json")
    te_sc = sentence_classes(te_text, rdir / "test_sentence_classes.json")

    # calibration on VAL only
    va_m = [mentioned(va_sc[str(s)][1]) for s in va_ids["study_id"]]
    tau = calibrate(p_va, va_m, float(sr["target_far"]), int(sr["min_absent"]))
    (rdir / "thresholds.json").write_text(json.dumps(tau, indent=2), encoding="utf-8")
    print("tau (val-calibrated):", {k: round(v, 3) for k, v in tau.items()}, flush=True)

    rng = random.Random(seed)
    tix = {str(s): i for i, s in enumerate(te_ids["study_id"])}
    recs = []
    for sid in sorted(te_text):
        sents, cls = te_sc[sid]
        p = p_te[tix[sid]]
        orig_flags = flag(p, mentioned(cls), tau)
        sim = simulate_deletion(sents, cls, rng)
        rec = {"study_id": sid, "dataset": test.loc[sid, "dataset"], "flags_unmodified": orig_flags}
        if sim:
            mod_flags = flag(p, mentioned(cls[sim["kept_idx"]]), tau)
            rec.update({"deleted": sim["deleted"], "removed_also": sim["removed_also"], "modified_report": sim["text"],
                        "flags_modified": mod_flags, "hit": sim["deleted"] in {f["finding"] for f in mod_flags}})
        recs.append(rec)

    if sr.get("use_boxes"):   # a flag must also be localised by MedGemma (official prompt, step3_localize)
        from step3_localize import locate

        from src.models.medgemma_zeroshot import load_model

        cand = {(r["study_id"], f["finding"]) for r in recs for k in ("flags_unmodified", "flags_modified")
                for f in r.get(k, [])}
        bp = rdir / "boxes.jsonl"
        done = {(b["study_id"], b["finding"]) for b in read_jsonl(bp)}
        todo = [{"study_id": s, "dataset": "test", "finding": f, "image_path": test.loc[s, "image_path"]}
                for s, f in sorted(cand - done)]
        if todo:
            bc = cfg["box"]
            model, proc = load_model(cfg["models"][bc["model"]], bc["dtype"])
            locate(model, proc, todo, bc["phrases"], bc, float(cfg["verify"]["side_center_band"]), bp,
                   log=lambda m: print(m, flush=True))
        ok = {(b["study_id"], b["finding"]) for b in read_jsonl(bp) if b["status"] == "ok"}
        for r in recs:
            for k in ("flags_unmodified", "flags_modified"):
                if k in r:
                    r[k] = [f for f in r[k] if (r["study_id"], f["finding"]) in ok]
            if "flags_modified" in r:
                r["hit"] = r["deleted"] in {f["finding"] for f in r["flags_modified"]}

    with open(rdir / "second_reader.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    sim = [r for r in recs if "deleted" in r]
    n_flags_mod = sum(len(r["flags_modified"]) for r in sim)
    hits = sum(r["hit"] for r in sim)
    per = {}
    for c in CONDITIONS:
        rs = [r for r in sim if r["deleted"] == c]
        if rs:
            per[c] = {"n": len(rs), "recall": round(sum(r["hit"] for r in rs) / len(rs), 4)}
    m = {"n_reports": len(recs), "n_simulated": len(sim), "recall": round(hits / max(1, len(sim)), 4),
         "precision": round(hits / max(1, n_flags_mod), 4),
         "flags_per_unmodified_report": round(sum(len(r["flags_unmodified"]) for r in recs) / max(1, len(recs)), 4),
         "unmodified_reports_with_any_flag": round(sum(bool(r["flags_unmodified"]) for r in recs) / max(1, len(recs)), 4),
         "most_flagged_unmodified": Counter(f["finding"] for r in recs for f in r["flags_unmodified"]).most_common(5),
         "per_deleted_condition": per, "target_far_val": sr["target_far"], "use_boxes": bool(sr.get("use_boxes"))}
    (rdir / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    key = [f"| {c} | {per[c]['n']} | {100 * per[c]['recall']:.1f}% |" for c in sr["key_findings"] if c in per]
    allc = [f"| {c} | {v['n']} | {100 * v['recall']:.1f}% |" for c, v in per.items()]
    body = "\n".join([
        f"## Step 8 — second-reader mode{' (SMOKE: not results)' if args.smoke else ''}",
        f"Run: `{rdir}`. Probe thresholds calibrated on VAL (<= {100 * sr['target_far']:.0f}% flags per condition on "
        f"val studies without it){'; flags also need a MedGemma box' if sr.get('use_boxes') else ''}.", "",
        f"Simulated misses (one positive condition deleted from {len(sim)} real test reports): "
        f"**recall {100 * m['recall']:.1f}%, precision {100 * m['precision']:.1f}%**.",
        f"Unmodified reports: {m['flags_per_unmodified_report']:.3f} flags per report "
        f"({100 * m['unmodified_reports_with_any_flag']:.1f}% of reports get >= 1 flag).", "",
        "Key findings:", "| deleted condition | n | recall |", "|---|---|---|", *key, "",
        "All conditions:", "| deleted condition | n | recall |", "|---|---|---|", *allc])
    (rdir / "results_table.md").write_text(body + "\n", encoding="utf-8")
    if not args.smoke:
        update_results_md(project_root(cfg) / "outputs" / "RESULTS.md", body, "STEP8")
    print("\n" + body)


if __name__ == "__main__":
    main()
