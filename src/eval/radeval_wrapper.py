"""RadEval wrappers: scoring, per-sample caching, paired significance, result tables.

Verified against RadEval source (github.com/jbdel/RadEval, commit d412dc2):
  - RadEval(metrics=[...], per_sample=bool, detailed=bool); evaluator(refs=..., hyps=...) -> dict
  - keys: radgraph_{simple,partial,complete}, f1chexbert_5_micro_f1, f1chexbert_all_micro_f1 (+4 more),
          ratescore, radcliq_v1 (1/mean raw -> higher is better), bleu (BLEU-4), rouge1/rouge2/rougeL
  - GREEN: the adapter's underlying scorer returns (mean, std, per_sample, df) in one call.
  - compare_systems() re-calls each metric callable on every randomisation trial, so callables here
    are lookups into cached per-sample scores. That is exact only for metrics whose corpus score is the
    mean of per-sample scores (RadGraph, RaTEScore, GREEN). CheXbert micro-F1 / BLEU are corpus-level
    and are NOT significance-tested this way.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

# Display columns for the results table: (header, metric key, scale)
TABLE_COLUMNS = [
    ("RadGraph F1 (partial)", "radgraph_partial", 100),
    ("CheXbert F1-5 (micro)", "f1chexbert_5_micro_f1", 100),
    ("CheXbert F1-14 (micro)", "f1chexbert_all_micro_f1", 100),
    ("RaTEScore", "ratescore", 100),
    ("GREEN", "green", 100),
    ("RadCliQ-v1 (higher=better)", "radcliq_v1", 1),
    ("BLEU-4", "bleu", 100),
    ("ROUGE-L", "rougeL", 100),
]
SIG_METRICS = {"radgraph_partial", "ratescore", "green"}


def _hyp(s: str) -> str:
    # An empty hypothesis is scored as "." (no clinical content) - never filled with text.
    return s if s and s.strip() else "."


def target_texts(rows: list[dict], target: str) -> tuple[list[str], list[str]]:
    if target == "findings":
        refs = [r["reference_findings"] for r in rows]
        hyps = [r["hyp_findings"] for r in rows]
    elif target == "report":
        refs = [f"{r['reference_findings']} {r['reference_impression']}".strip() for r in rows]
        hyps = [f"{r['hyp_findings']} {r['hyp_impression']}".strip() for r in rows]
    else:
        raise ValueError(target)
    return refs, [_hyp(h) for h in hyps]


class Scorer:
    """Instantiates RadEval evaluators once (they load several models) and reuses them."""

    def __init__(self, default_metrics: list[str], per_sample_metrics: list[str], green: bool):
        from radeval import RadEval

        self.default = RadEval(metrics=default_metrics, show_progress=True) if default_metrics else None
        self.per_sample = RadEval(metrics=per_sample_metrics, per_sample=True, show_progress=True) \
            if per_sample_metrics else None
        self.green = RadEval(metrics=["green"], show_progress=True) if green else None

    def score(self, refs: list[str], hyps: list[str]) -> tuple[dict, dict]:
        """Returns (aggregate metrics, per-sample lists)."""
        agg, per = {}, {}
        if self.default:
            agg.update(self.default(refs=refs, hyps=hyps))
        if self.per_sample:
            per.update(self.per_sample(refs=refs, hyps=hyps))
        if self.green:
            res = self.green.green_scorer(refs, hyps)
            if not (isinstance(res, tuple) and len(res) >= 3):
                raise RuntimeError(f"Unexpected GREEN scorer output type {type(res)}; check RadEval version.")
            mean, std, samples = res[0], res[1], list(res[2])
            valid = [float(x) for x in samples if x is not None and not (isinstance(x, float) and math.isnan(x))]
            agg.update({"green": float(mean), "green_std": float(std),
                        "green_n_valid": len(valid), "green_n_total": len(samples)})
            per["green"] = [None if x is None else float(x) for x in samples]
        for k, v in list(agg.items()):
            if isinstance(v, (np.floating, np.integer)):
                agg[k] = v.item()
        return agg, per


def cached_metric(per_sample_by_system: dict[str, list], hyps_by_system: dict[str, list[str]]) -> Callable:
    """Metric callable for compare_systems: mean of cached per-sample scores (None = invalid, excluded).

    The cache is keyed by (index, hypothesis text): after a random swap, position i holds one of the two
    systems' outputs for the same study, whose score was computed against the same reference.
    """
    cache: dict[tuple[int, str], float | None] = {}
    for name, hyps in hyps_by_system.items():
        for i, (h, s) in enumerate(zip(hyps, per_sample_by_system[name])):
            prev = cache.get((i, h), s)
            if prev is not None and s is not None and abs(prev - s) > 1e-9:
                raise AssertionError(f"Non-deterministic per-sample score at index {i}")
            cache[(i, h)] = s

    def fn(hyps: list[str], refs: list[str]) -> float:
        vals = [cache[(i, h)] for i, h in enumerate(hyps)]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else float("nan")

    return fn


def paired_test(systems: dict[str, list[str]], per_sample: dict[str, dict[str, list]],
                refs: list[str], n_samples: int, alpha: float, seed: int) -> dict:
    """RadEval's paired approximate-randomisation test; first system is the baseline."""
    from radeval import compare_systems

    metric_names = [m for m in SIG_METRICS if all(m in per_sample[s] for s in systems)]
    metrics = {m: cached_metric({s: per_sample[s][m] for s in systems}, systems) for m in metric_names}
    signatures, scores = compare_systems(systems=systems, metrics=metrics, references=refs,
                                         n_samples=n_samples, significance_level=alpha, seed=seed,
                                         print_results=True)
    return {"baseline": next(iter(systems)), "n_samples": n_samples, "alpha": alpha,
            "signatures": signatures, "scores": scores}


def _fmt(v, scale) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "-"
    return f"{v * scale:.2f}" if scale != 1 else f"{v:.4f}"


def results_table(entries: list[dict]) -> str:
    """entries: {system, dataset, target, n, dtype, metrics, parse_ok_rate}. Values come only from metrics."""
    head = ["System", "Dataset", "Target", "dtype"] + [c[0] for c in TABLE_COLUMNS] + ["n studies", "format ok %"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for e in entries:
        cells = [e["system"], e["dataset"], e["target"], e["dtype"]]
        cells += [_fmt(e["metrics"].get(k), s) for _, k, s in TABLE_COLUMNS]
        cells += [str(e["n"]), f"{100 * e['parse_ok_rate']:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
