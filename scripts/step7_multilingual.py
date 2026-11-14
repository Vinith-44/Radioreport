"""Step 7: plain-language summaries in Indian languages + back-translation faithfulness check. Resumable.

  python scripts/step7_multilingual.py --run-name m1 [--smoke]

Per report: English report -> simple English summary (MedGemma 1.5, text only) -> translation (IndicTrans2, or the
LLM) into each language -> back-translation to English -> CheXbert labels (RadEval, rrg) of the back-translation
must equal those of the English report; otherwise the summary is FLAGGED. The labels of the English summary are
checked too, to tell whether a lost / added finding came from simplification or from translation.
Stages are cached in outputs/runs/step7/<run>/{summaries,translations}.jsonl; one model is loaded at a time.
Writes faithfulness.json (per-language agreement), flagged_<language>.jsonl (for review), and the STEP7 block.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (append_jsonl, load_config, project_root, read_jsonl, run_dir, set_seed,  # noqa: E402
                        write_run_info)
from src.data.text_clean import format_report  # noqa: E402
from src.multilingual.translate import (LANGS, SIMPLIFY_PROMPT, LLMTranslator, label_agreement,  # noqa: E402
                                        text_generate, translate_text)
from src.eval.scoring import update_results_md  # noqa: E402


def free():
    """Call AFTER dropping every reference to the model (x = None): then the GPU memory is really released."""
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step7.yaml")
    ap.add_argument("--run-name", default="m1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--backend", choices=["indictrans2", "llm"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    mc = cfg["multilingual"]
    backend = args.backend or mc["backend"]
    sfx = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step7", args.run_name + sfx)
    src = mc.get("source")
    if not src:
        w = run_dir(cfg, "step5", cfg["robust"]["winner_run"] + sfx) / "winner.json"
        if not w.exists():
            raise SystemExit(f"no source set and no {w}")
        src = json.loads(w.read_text(encoding="utf-8"))["winner_spec"]
    sdir = run_dir(cfg, src["step"], src["run"] + sfx) / src["system"]
    rows = [r for p in sorted(sdir.glob("*/predictions.jsonl")) for r in read_jsonl(p)]
    if not rows:
        raise SystemExit(f"no predictions under {sdir}")
    n = 4 if args.smoke else int(mc["n_reports"])
    rows = sorted(random.Random(seed).sample(rows, min(n, len(rows))), key=lambda r: r["study_id"])
    langs = mc["languages"][:1] if args.smoke else mc["languages"]
    bad = [l_ for l_ in langs if l_ not in LANGS]
    if bad:
        raise SystemExit(f"unknown languages {bad}; known: {list(LANGS)}")
    write_run_info(rdir, cfg, args, {"step": "step7", "source": str(sdir), "backend": backend, "languages": langs})
    report = {r["study_id"]: format_report(r["hyp_findings"], r["hyp_impression"]) for r in rows}

    # 1) English plain-language summaries
    sp = rdir / "summaries.jsonl"
    have = {r["study_id"]: r for r in read_jsonl(sp)}
    todo = [s for s in report if s not in have]
    llm = proc = None
    from src.models.medgemma_zeroshot import load_model

    done_t = {(r["study_id"], r["language"]) for r in read_jsonl(rdir / f"translations_{backend}.jsonl")}
    jobs_left = any((s, l_) not in done_t for l_ in langs for s in report)
    if todo or (backend == "llm" and jobs_left):
        llm, proc = load_model(cfg["models"][mc["summarizer"]], mc["summarizer_dtype"])
    for i in range(0, len(todo), int(mc["batch_size"])):
        ch = todo[i:i + int(mc["batch_size"])]
        outs = text_generate(llm, proc, [SIMPLIFY_PROMPT.format(report=report[s]) for s in ch], int(mc["max_new_tokens"]))
        append_jsonl(sp, [{"study_id": s, "summary_en": o} for s, o in zip(ch, outs)])
        print(f"summaries {len(have) + i + len(ch)}/{len(report)}", flush=True)
    summ = {r["study_id"]: r["summary_en"] for r in read_jsonl(sp)}

    # 2) translation + back-translation
    tp = rdir / f"translations_{backend}.jsonl"
    have_t = {(r["study_id"], r["language"]) for r in read_jsonl(tp)}
    jobs = [(s, l_) for l_ in langs for s in report if (s, l_) not in have_t]
    if jobs:
        if backend == "llm":
            fwd = back = LLMTranslator(llm, proc, int(mc["max_new_tokens"]))
        else:
            if llm is not None:
                llm = proc = None
                free()
            import torch

            from src.models.medgemma_zeroshot import hf_token
            from src.multilingual.translate import IndicTrans2

            dev = "cuda" if torch.cuda.is_available() else "cpu"
            it = mc["indictrans2"]
            fwd = IndicTrans2("en-indic", it["en_indic"], dev, hf_token(), it["dtype"])
            back = None
        tr = {}
        for s, l_ in jobs:
            tr[(s, l_)] = translate_text(fwd, summ[s], "eng_Latn", LANGS[l_])
        if back is None:
            fwd = None
            free()
            back = IndicTrans2("indic-en", it["indic_en"], dev, hf_token(), it["dtype"])
        out = []
        for (s, l_), t in tr.items():
            out.append({"study_id": s, "language": l_, "translation": t,
                        "back_translation": translate_text(back, t, LANGS[l_], "eng_Latn")})
            if len(out) % 50 == 0:
                append_jsonl(tp, out)
                print(f"translated {len(have_t) + len(out)}/{len(report) * len(langs)}", flush=True)
                out = []
        append_jsonl(tp, out)
        fwd = back = None
        free()
    llm = proc = None
    free()

    # 3) faithfulness: CheXbert labels of report vs back-translation (and vs English summary)
    from src.retrieval.rerank import chexbert_labels

    trs = [r for r in read_jsonl(tp) if r["study_id"] in report and r["language"] in langs]
    ids = sorted(report)
    lab = chexbert_labels([report[s] for s in ids] + [summ[s] for s in ids] + [r["back_translation"] for r in trs])
    L_rep = {s: lab[i].tolist() for i, s in enumerate(ids)}
    L_sum = {s: lab[len(ids) + i].tolist() for i, s in enumerate(ids)}
    res = {"backend": backend, "n_reports": len(ids), "summary_en": {}, "languages": {}}
    agr_sum = [label_agreement(L_rep[s], L_sum[s]) for s in ids]
    res["summary_en"] = {"exact_agreement": round(sum(a["exact"] for a in agr_sum) / len(ids), 4),
                         "per_label_agreement": round(sum(a["per_label_agree"] for a in agr_sum) / len(ids), 4)}
    for l_ in langs:
        sel = [(k, r) for k, r in enumerate(trs) if r["language"] == l_]
        flagged, ex = [], 0
        for k, r in sel:
            a = label_agreement(L_rep[r["study_id"]], lab[2 * len(ids) + k].tolist())
            ex += a["exact"]
            if not a["exact"]:
                ss = label_agreement(L_rep[r["study_id"]], L_sum[r["study_id"]])
                flagged.append({**r, "report_en": report[r["study_id"]], "summary_en": summ[r["study_id"]],
                                "lost": a["lost"], "added": a["added"],
                                "stage": "simplification" if not ss["exact"] else "translation"})
        with open(rdir / f"flagged_{l_}_{backend}.jsonl", "w", encoding="utf-8") as f:
            for x in flagged:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        res["languages"][l_] = {"n": len(sel), "label_agreement_rate": round(ex / max(1, len(sel)), 4),
                                "flagged": len(flagged),
                                "flagged_at_translation": sum(x["stage"] == "translation" for x in flagged)}
    (rdir / f"faithfulness_{backend}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    lines = [f"| {l_} | {v['n']} | {100 * v['label_agreement_rate']:.1f}% | {v['flagged']} | {v['flagged_at_translation']} |"
             for l_, v in res["languages"].items()]
    body = "\n".join([
        f"## Step 7 — Indian-language patient summaries{' (SMOKE: not results)' if args.smoke else ''}",
        f"Source reports: `{sdir}` ({len(ids)} test reports); translation backend: {backend}. A summary is FLAGGED "
        "when the CheXbert labels of its back-translation differ from the English report's.", "",
        f"English plain-language summary vs report: exact label agreement "
        f"{100 * res['summary_en']['exact_agreement']:.1f}%", "",
        "| language | n | label agreement (exact, 14 labels) | flagged | of which lost/added at translation |",
        "|---|---|---|---|---|", *lines, "", f"Flagged examples for review: `{rdir}/flagged_<language>_{backend}.jsonl`"])
    (rdir / "results_table.md").write_text(body + "\n", encoding="utf-8")
    if not args.smoke:
        update_results_md(project_root(cfg) / "outputs" / "RESULTS.md", body, "STEP7")
    print("\n" + body)


if __name__ == "__main__":
    main()
