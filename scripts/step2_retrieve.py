"""Step 2d: FAISS index over TRAIN images + fact-aware retrieval for every test query.

  python scripts/step2_retrieve.py --run-name r1 [--smoke] [--mode rerank|similarity]

For each test study of the fixed Step 0 subset: top_k most similar TRAIN images (cosine, MedSigLIP) ->
re-rank by agreement between each candidate's CheXbert report labels and the Probe's prediction for the query
image -> the n_refs best (distinct) reports. Writes <run>/retrieval[_similarity].jsonl
(study_id -> references block + what was chosen and why) and <run>/index/train.faiss.
Leakage guards: the index holds train studies only; a candidate of the query's own patient is never used;
the script asserts that no chosen study's patient appears in the test split.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import data_dir, load_config, run_dir, set_seed, write_run_info  # noqa: E402
from src.models.probe import load_probe, predict  # noqa: E402
from src.retrieval.index import FlatIP  # noqa: E402
from src.retrieval.rerank import CHEXBERT_LABELS, references_block, rerank  # noqa: E402
from src.retrieval.store import load_split, report_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step2.yaml")
    ap.add_argument("--run-name", default="r1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", choices=["rerank", "similarity"], default="rerank",
                    help="similarity = plain top-n by image similarity (extra ablation)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["seed"]))
    rc = cfg["retrieval"]
    suffix = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step2", args.run_name + suffix)
    write_run_info(rdir, cfg, args, {"step": "step2_retrieve", "retrieval": rc, "mode": args.mode})

    tr_ids, tr_emb, tr_lab = load_split(rdir, "train", labels=True)
    te_ids, te_emb = load_split(rdir, "test")
    test_patients = set(pd.read_parquet(data_dir(cfg) / f"test{suffix}.parquet", columns=["patient_id"])["patient_id"])
    leak = set(tr_ids["patient_id"]) & test_patients
    assert not leak, f"PATIENT LEAKAGE: {len(leak)} index (train) patients are in the test split"

    index = FlatIP(tr_emb)
    index.save(rdir / "index" / "train.faiss")
    print(f"index: {index.n} train images, dim {index.d}, backend {index.backend}", flush=True)
    probe, pinfo = load_probe(rdir / "probe")
    _, te_bin = predict(probe, te_emb, pinfo["thresholds"])

    k, n_refs = int(rc["top_k"]), int(rc["n_refs"])
    sims, idxs = index.search(te_emb, k)
    texts = [report_text(r, rc["report_text"]) for r in tr_ids.to_dict("records")]
    pats = tr_ids["patient_id"].tolist()
    sids = tr_ids["study_id"].astype(str).tolist()
    out_p = rdir / ("retrieval.jsonl" if args.mode == "rerank" else "retrieval_similarity.jsonl")
    stats = {"agree_chosen": [], "agree_top_sim": [], "sim_chosen": []}
    with open(out_p, "w", encoding="utf-8") as f:
        for q, row in enumerate(te_ids.to_dict("records")):
            ci, cs = idxs[q], sims[q]
            c_lab = tr_lab[ci]
            chosen = rerank(ci, cs, c_lab, None if args.mode == "similarity" else te_bin[q],
                            [pats[i] for i in ci], [texts[i] for i in ci], row["patient_id"], n_refs,
                            bool(rc["dedupe"]))
            widened = False
            if len(chosen) < n_refs:  # e.g. many word-identical normal reports in the top k: look further
                s2, i2 = index.search(te_emb[q], k * 5)
                ci, cs = i2[0], s2[0]
                c_lab = tr_lab[ci]
                chosen = rerank(ci, cs, c_lab, None if args.mode == "similarity" else te_bin[q],
                                [pats[i] for i in ci], [texts[i] for i in ci], row["patient_id"], n_refs,
                                bool(rc["dedupe"]))
                widened = True
                stats["n_widened"] = stats.get("n_widened", 0) + 1
            if len(chosen) < n_refs:
                raise SystemExit(f"study {row['study_id']}: only {len(chosen)} usable candidates in top {k * 5}")
            for c in chosen:
                assert pats[c["index"]] not in test_patients and pats[c["index"]] != row["patient_id"]
            stats["agree_chosen"] += [float((tr_lab[c["index"]] == te_bin[q]).mean()) for c in chosen]
            stats["agree_top_sim"] += [float((tr_lab[i] == te_bin[q]).mean()) for i in ci[:n_refs]]
            stats["sim_chosen"] += [c["sim"] for c in chosen]
            f.write(json.dumps({
                "study_id": str(row["study_id"]), "dataset": row["dataset"],
                "references": references_block([texts[c["index"]] for c in chosen], int(rc["max_ref_chars"])),
                "chosen": [{"study_id": sids[c["index"]], "patient_id": pats[c["index"]], "sim": round(c["sim"], 4),
                            "agreement": round(c["agreement"], 4), "rank_in_search": c["rank_in_search"]}
                           for c in chosen],
                "probe_positive": [n for n, b in zip(CHEXBERT_LABELS, te_bin[q]) if b],
                "mode": args.mode, "widened_search": widened,
            }) + "\n")
    n_wide = stats.pop("n_widened", 0)
    summary = {k2: round(float(np.mean(v)), 4) for k2, v in stats.items()}
    summary["n_widened_search"] = n_wide
    summary.update({"n_queries": len(te_ids), "top_k": k, "n_refs": n_refs, "mode": args.mode,
                    "index_size": index.n, "file": str(out_p)})
    (rdir / f"retrieval_summary_{args.mode}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    ex = [json.loads(line) for line in open(out_p, encoding="utf-8")][:2]
    for e in ex:
        print(f"\n--- test study {e['study_id']} (probe: {e['probe_positive']})\n{e['references']}")


if __name__ == "__main__":
    main()
