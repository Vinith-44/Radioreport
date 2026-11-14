"""Shared scoring pipeline for every step: collect saved predictions, score them with RadEval (cached by a
digest of the exact (study_id, hypothesis, reference) triples + metric config), paired tests, RESULTS.md blocks.
Numbers only ever come from predictions.jsonl files on disk (CLAUDE.md §3)."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.common import read_jsonl
from src.eval.radeval_wrapper import paired_test, target_texts


def metrics_config(sc: dict, smoke: bool, no_green: bool = False) -> dict:
    if smoke:
        return {"default": sc["smoke_metrics"], "per_sample": [m for m in sc["per_sample_metrics"]
                                                               if m in sc["smoke_metrics"]], "green": False}
    return {"default": sc["default_metrics"], "per_sample": sc["per_sample_metrics"],
            "green": bool(sc["green"]) and not no_green}


def digest(rows: list[dict], metrics_cfg: dict) -> str:
    h = hashlib.sha256(json.dumps(metrics_cfg, sort_keys=True).encode())
    for r in rows:
        h.update(f"{r['study_id']}\x00{r['hypothesis']}\x00{r['reference']}\n".encode("utf-8"))
    return h.hexdigest()


def collect(sys_dir: Path, eval_ids: dict, allow_partial: bool) -> dict[str, list[dict]]:
    """<sys_dir>/<dataset>/predictions.jsonl -> {dataset: rows in eval-id order}; incomplete files skipped."""
    out = {}
    for p in sorted(sys_dir.glob("*/predictions.jsonl")):
        ds = p.parent.name
        rows = {r["study_id"]: r for r in read_jsonl(p)}
        want = eval_ids.get(ds, [])
        have = [rows[i] for i in want if i in rows]
        if len(have) < len(want):
            msg = f"[{sys_dir.name}/{ds}] {len(have)}/{len(want)} predictions"
            if not allow_partial:
                print(f"SKIP {msg} - incomplete (finish generation, or --allow-partial).")
                continue
            print(f"PARTIAL {msg}")
        if have:
            out[ds] = have
    return out


class LazyScorer:
    """Loads RadEval's models only if something actually needs (re)scoring."""

    def __init__(self, mcfg: dict):
        self.mcfg, self._s = mcfg, None

    def score(self, refs, hyps):
        if self._s is None:
            from src.eval.radeval_wrapper import Scorer

            self._s = Scorer(self.mcfg["default"], self.mcfg["per_sample"], self.mcfg["green"])
        return self._s.score(refs, hyps)


def score_system(out: Path, system: str, ds: str, rows: list[dict], n_wanted: int, mcfg: dict,
                 targets: list[str], scorer: LazyScorer) -> tuple[dict, dict]:
    """Score one (system, dataset) folder; reuse metrics.json/per_sample.jsonl when the digest matches.
    Returns (result dict, per-sample scores {target: {metric: [..]}})."""
    dg = digest(rows, mcfg)
    mpath = out / "metrics.json"
    cached = json.loads(mpath.read_text(encoding="utf-8")) if mpath.exists() else None
    if cached and cached.get("digest") == dg:
        print(f"[{system}/{ds}] metrics up to date (cached).")
        per = {t: {} for t in targets}
        for r in read_jsonl(out / "per_sample.jsonl"):
            for t in targets:
                for m, v in r.get(t, {}).items():
                    per[t].setdefault(m, []).append(v)
        return cached, per
    result = {"model": system, "dataset": ds, "n": len(rows), "digest": dg, "metrics_cfg": mcfg,
              "partial": len(rows) < n_wanted,
              "dtype": sorted({r["dtype"] for r in rows}),
              "model_id": sorted({r["model_id"] for r in rows}),
              "adapter_sha256": sorted({str(r.get("adapter_sha256")) for r in rows}),
              "parse_ok_rate": sum(r["parse_ok"] for r in rows) / len(rows),
              "hit_max_tokens_rate": sum(r["hit_max_tokens"] for r in rows) / len(rows),
              "empty_hyp_findings": sum(not r["hyp_findings"].strip() for r in rows),
              "targets": {}}
    per = {}
    for t in targets:
        refs, hyps = target_texts(rows, t)
        print(f"[{system}/{ds}/{t}] scoring {len(rows)} studies ...", flush=True)
        agg, ps = scorer.score(refs, hyps)
        result["targets"][t] = agg
        per[t] = ps
    mpath.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with open(out / "per_sample.jsonl", "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"study_id": r["study_id"],
                                **{t: {m: v[i] for m, v in per[t].items()} for t in per}}) + "\n")
    return result, per


def paired(base_name: str, base_rows: list[dict], base_per: dict, cand_name: str, cand_rows: list[dict],
           cand_per: dict, n_samples: int, alpha: float, seed: int, target: str = "findings") -> dict | None:
    """Paired approximate-randomisation test (baseline first). None if the two do not cover the same studies."""
    if [r["study_id"] for r in base_rows] != [r["study_id"] for r in cand_rows]:
        return None
    if [r["reference"] for r in base_rows] != [r["reference"] for r in cand_rows]:
        raise AssertionError("same study ids but different references: the two runs used different data")
    refs, hb = target_texts(base_rows, target)
    _, hc = target_texts(cand_rows, target)
    return paired_test({base_name: hb, cand_name: hc},
                       {base_name: base_per[target], cand_name: cand_per[target]}, refs, n_samples, alpha, seed)


def pvalues(sig: dict, system: str) -> dict[str, float]:
    s = sig.get("scores", {}).get(system, {})
    return {k.removesuffix("_pvalue"): float(v) for k, v in s.items() if k.endswith("_pvalue")}


def update_results_md(path: Path, body: str, tag: str) -> None:
    begin, end = f"<!-- {tag}:BEGIN -->", f"<!-- {tag}:END -->"
    text = path.read_text(encoding="utf-8") if path.exists() else "# RESULTS\n"
    block = f"{begin}\n{body}\n{end}"
    if begin in text:
        text = re.sub(re.escape(begin) + r".*?" + re.escape(end), lambda _: block, text, flags=re.S)
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text, encoding="utf-8")
