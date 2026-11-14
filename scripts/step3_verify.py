"""Step 3c: sentence verdicts (VERIFIED / UNCERTAIN / FLAGGED) + evaluation of the verifier against the reference.

  python scripts/step3_verify.py --run-name v1 [--smoke]

Evidence: Step 2 probe on the test image (verify.step2_run), MedGemma boxes (boxes.jsonl), and as context the
Step 2 retrieved train cases whose CheXbert label agrees with the claim ("supporting_cases"). Rules: see
src/verify/verdict.py. Writes outputs/runs/step3/<run>/
  verify.jsonl                         one JSON per report: sentences, claims, verdicts, reasons, boxes, cases
  <source>+verified-drop/<ds>/predictions.jsonl   the same reports with FLAGGED sentences removed
  verify_eval.json                     how many clinically WRONG sentences (vs reference) were caught, false alarms
The reference report is used ONLY for verify_eval.json, never to decide a verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.common import load_config, read_jsonl, run_dir, set_seed, write_run_info  # noqa: E402
from src.data.text_clean import format_report  # noqa: E402
from src.models.probe import auroc, load_probe, predict  # noqa: E402
from src.retrieval.rerank import CHEXBERT_LABELS  # noqa: E402
from src.retrieval.store import load_split  # noqa: E402
from src.verify.verdict import (claim_correct, claim_verdict, probe_state, sentence_correct,  # noqa: E402
                                sentence_verdict)

LIDX = {n: i for i, n in enumerate(CHEXBERT_LABELS)}


def summarize(sents: list[dict]) -> dict:
    v = Counter(s["verdict"] for s in sents)
    wrong = [s for s in sents if s["correct"] is False]
    right = [s for s in sents if s["correct"] is True]
    caught = sum(s["verdict"] in ("FLAGGED", "UNCERTAIN") for s in wrong)
    flagged_w = sum(s["verdict"] == "FLAGGED" for s in wrong)
    fa = sum(s["verdict"] == "FLAGGED" for s in right)
    fa_u = sum(s["verdict"] in ("FLAGGED", "UNCERTAIN") for s in right)
    rate = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {"sentences": len(sents), "verdicts": dict(v), "judged": len(wrong) + len(right),
            "wrong": len(wrong), "correct": len(right),
            "wrong_caught_rate (FLAGGED or UNCERTAIN)": rate(caught, len(wrong)),
            "wrong_flagged_rate (FLAGGED)": rate(flagged_w, len(wrong)),
            "false_alarm_rate (correct but FLAGGED)": rate(fa, len(right)),
            "correct_not_verified_rate (correct but FLAGGED or UNCERTAIN)": rate(fa_u, len(right)),
            "precision_of_FLAGGED (share of FLAGGED that are wrong)": rate(flagged_w, flagged_w + fa)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step3.yaml")
    ap.add_argument("--run-name", default="v1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    vc = cfg["verify"]
    sfx = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step3", args.run_name + sfx)
    s2 = run_dir(cfg, "step2", vc["step2_run"] + sfx)
    write_run_info(rdir, cfg, args, {"step": "step3_verify", "step2_run": str(s2), "verify": vc})
    m = float(vc["strong_margin"])

    # probe on the test images
    te_ids, te_emb = load_split(s2, "test")
    probe, pinfo = load_probe(s2 / "probe")
    p_all, _ = predict(probe, te_emb, pinfo["thresholds"])
    thr = np.asarray(pinfo["thresholds"])
    prob = {str(s): p_all[i] for i, s in enumerate(te_ids["study_id"])}
    # supporting cases: Step 2 retrieved train cases + their CheXbert labels
    retr = {r["study_id"]: [c["study_id"] for c in r["chosen"]] for r in read_jsonl(s2 / "retrieval.jsonl")} \
        if (s2 / "retrieval.jsonl").exists() else {}
    tr_lab = {}
    if retr:
        tr_ids, _, lab = load_split(s2, "train", labels=True)
        tr_lab = {str(s): lab[i] for i, s in enumerate(tr_ids["study_id"])}
    boxes = {(r["study_id"], r["finding"]): r for r in read_jsonl(rdir / "boxes.jsonl")}

    src_rows = {}
    for p in sorted((rdir / "source").glob("*/predictions.jsonl")):
        for r in read_jsonl(p):
            src_rows[r["study_id"]] = r
    sys_out = f"{vc['source']['system']}+{vc['drop_system']}"
    all_sents: dict[str, list] = {}
    n_missing_box = 0
    out_files = {}
    probe_eval = {"y": [], "p": []}
    with open(rdir / "verify.jsonl", "w", encoding="utf-8") as fv:
        for rep in read_jsonl(rdir / "claims.jsonl"):
            sid, ds = rep["study_id"], rep["dataset"]
            if sid not in prob:
                raise SystemExit(f"no Step 2 test embedding for study {sid} (rerun step2_embed with the same test subset)")
            p = prob[sid]
            ref = rep["ref_classes"]
            probe_eval["y"].append([1 if ref[n] in (1, 3) else 0 for n in CHEXBERT_LABELS])
            probe_eval["p"].append(p)
            sents = []
            for s in rep["sentences"]:
                cl_out, verdicts, judg = [], [], []
                for c in s["claims"]:
                    i = LIDX[c["finding"]]
                    ps = probe_state(p[i], thr[i], m)
                    bx = boxes.get((sid, c["finding"])) if c["polarity"] == "positive" else None
                    if c["polarity"] == "positive" and c["finding"] != "No Finding" and bx is None:
                        n_missing_box += 1
                    v, why = claim_verdict(c, ps, bx)
                    want = 1 if c["polarity"] == "positive" else 0 if c["polarity"] == "negative" else None
                    support = [cid for cid in retr.get(sid, []) if want is not None and cid in tr_lab
                               and int(tr_lab[cid][i]) == want]
                    ok = claim_correct(c, ref[c["finding"]], set(rep["ref_sides"].get(c["finding"], [])))
                    verdicts.append(v)
                    judg.append(ok)
                    cl_out.append({**c, "verdict": v, "reason": why, "probe": ps,
                                   "box": None if bx is None else {k: bx[k] for k in ("status", "boxes", "sides", "zones")},
                                   "supporting_cases": support, "correct_vs_reference": ok})
                sv = sentence_verdict(verdicts)
                sents.append({"section": s["section"], "text": s["text"], "verdict": sv, "claims": cl_out,
                              "correct": sentence_correct(judg)})
            all_sents.setdefault(ds, []).extend(sents)
            fv.write(json.dumps({"study_id": sid, "dataset": ds, "sentences": sents,
                                 "counts": dict(Counter(x["verdict"] for x in sents))}, default=str) + "\n")

            # the same report without FLAGGED sentences (a section without flags is kept byte-identical)
            r = dict(src_rows[sid])
            n_drop = 0
            for sec in vc["sections"]:
                ss = [x for x in sents if x["section"] == sec]
                if any(x["verdict"] == "FLAGGED" for x in ss):
                    n_drop += sum(x["verdict"] == "FLAGGED" for x in ss)
                    r[f"hyp_{sec}"] = " ".join(x["text"] for x in ss if x["verdict"] != "FLAGGED")
            r["hypothesis"] = format_report(r["hyp_findings"], r["hyp_impression"]) if n_drop else r["hypothesis"]
            r["model"], r["n_flagged_dropped"] = sys_out, n_drop
            if ds not in out_files:
                d = rdir / sys_out / ds
                d.mkdir(parents=True, exist_ok=True)
                out_files[ds] = open(d / "predictions.jsonl", "w", encoding="utf-8")
            out_files[ds].write(json.dumps(r) + "\n")
    for f in out_files.values():
        f.close()

    y, pp = np.asarray(probe_eval["y"]), np.asarray(probe_eval["p"])
    probe_auc = {n: (None if np.isnan(a) else round(a, 4))
                 for n, a in ((n, auroc(y[:, i], pp[:, i])) for i, n in enumerate(CHEXBERT_LABELS))}
    res = {"per_dataset": {ds: summarize(s) for ds, s in all_sents.items()},
           "all": summarize([x for s in all_sents.values() for x in s]),
           "positive_claims_without_box_query": n_missing_box,
           "probe_test_auroc_vs_reference_chexbert (evaluation only)": probe_auc,
           "rules": "src/verify/verdict.py", "strong_margin": m}
    (rdir / "verify_eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))
    if n_missing_box:
        print(f"WARNING: {n_missing_box} positive claims had no box result (run step3_localize.py to completion)")


if __name__ == "__main__":
    main()
