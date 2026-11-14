"""Step 3a: split the generated reports into sentences and extract each sentence's claims (CheXbert raw classes +
side/zone); also label the REFERENCE reports (used only to evaluate the verdicts, never by the verifier).

  python scripts/step3_claims.py --run-name v1 [--smoke]

Reads the source system's predictions (configs/step3.yaml verify.source). Writes outputs/runs/step3/<run>/
claims.jsonl, verify_ids.json (the verified study ids per dataset), and source/<dataset>/predictions.jsonl (an
exact copy of the source rows for those ids, so the before/after comparison lives in one folder).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import data_dir, load_config, read_jsonl, run_dir, set_seed, write_run_info  # noqa: E402
from src.retrieval.rerank import CHEXBERT_LABELS  # noqa: E402
from src.verify.claims import chexbert_classes, claims_of, finding_sides, split_sentences  # noqa: E402


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
    src = vc["source"]
    sdir = run_dir(cfg, src["step"], src["run"] + sfx) / src["system"]
    rdir = run_dir(cfg, "step3", args.run_name + sfx)
    eval_ids = json.loads((data_dir(cfg) / f"step0_eval_ids{sfx}.json").read_text(encoding="utf-8"))
    write_run_info(rdir, cfg, args, {"step": "step3_claims", "source_dir": str(sdir)})

    cap = vc.get("max_studies_per_dataset")
    rows, ids = [], {}
    for ds, want in eval_ids.items():
        p = sdir / ds / "predictions.jsonl"
        if not p.exists():
            print(f"[{ds}] no source predictions at {p} - skipped")
            continue
        have = {r["study_id"]: r for r in read_jsonl(p)}
        missing = [i for i in want if i not in have]
        if missing:
            raise SystemExit(f"[{ds}] source predictions incomplete ({len(missing)} missing) - finish that step first")
        sel = want[: int(cap)] if cap else want
        ids[ds] = sel
        out = rdir / "source" / ds
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "predictions.jsonl", "w", encoding="utf-8") as f:
            for i in sel:
                f.write(json.dumps(have[i]) + "\n")
        rows += [have[i] for i in sel]
    if not rows:
        raise SystemExit(f"No source predictions under {sdir}")
    (rdir / "verify_ids.json").write_text(json.dumps(ids), encoding="utf-8")

    # Every text that needs CheXbert, in one batch-friendly list.
    hyp_s, ref_s, ref_full = [], [], []
    for r in rows:
        hyp_s.append([(sec, s) for sec in vc["sections"] for s in split_sentences(r.get(f"hyp_{sec}") or "")])
        ref_s.append(split_sentences(f"{r['reference_findings']} {r['reference_impression']}".strip()))
        ref_full.append(f"{r['reference_findings']} {r['reference_impression']}".strip())
    flat = [s for ss in hyp_s for _, s in ss] + [s for ss in ref_s for s in ss] + ref_full
    print(f"{len(rows)} reports: {sum(map(len, hyp_s))} generated sentences, {sum(map(len, ref_s))} reference "
          f"sentences -> CheXbert", flush=True)
    from radeval.metrics.f1chexbert import F1CheXbert

    import torch

    scorer = F1CheXbert(device="cuda" if torch.cuda.is_available() else "cpu", batch_size=64)
    cls = chexbert_classes(flat, scorer)
    k = 0
    with open(rdir / "claims.jsonl", "w", encoding="utf-8") as f:
        hyp_cls = []
        for ss in hyp_s:
            hyp_cls.append(cls[k:k + len(ss)])
            k += len(ss)
        ref_cls = []
        for ss in ref_s:
            ref_cls.append(cls[k:k + len(ss)])
            k += len(ss)
        full_cls = cls[k:k + len(ref_full)]
        for r, ss, hc, rs, rc, fc in zip(rows, hyp_s, hyp_cls, ref_s, ref_cls, full_cls):
            sides = finding_sides(rs, rc)
            f.write(json.dumps({
                "study_id": r["study_id"], "dataset": r["dataset"],
                "sentences": [{"section": sec, "text": s, "claims": claims_of(s, c)} for (sec, s), c in zip(ss, hc)],
                "ref_classes": {n: int(c) for n, c in zip(CHEXBERT_LABELS, fc)},
                "ref_sides": {n: sorted(v) for n, v in sides.items()},
            }) + "\n")
    n_claims = sum(len(claims_of(s, c)) for ss, hc in zip(hyp_s, hyp_cls) for (_, s), c in zip(ss, hc))
    print(f"wrote {rdir / 'claims.jsonl'}: {n_claims} claims", flush=True)


if __name__ == "__main__":
    main()
