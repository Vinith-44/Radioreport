"""Step 2e: Model A WITH retrieved references on the fixed test subset (the ablation's second arm). Resumable.

  python scripts/step2_generate.py --run-name r1 [--smoke] [--mode similarity]

Exactly the Step 0/1 generation code with the Step 1 adapter (retrieval.step1_run) and the same decoding; the
only change is the prompt (generation.prompt_template_retrieval + this run's retrieval.jsonl). Writes
outputs/runs/step2/<run>[_smoke]/model-a+retrieval[-sim]/<dataset>/predictions.jsonl.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from src.common import load_config, run_dir  # noqa: E402
from step0_zeroshot_generate import main as generate  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--config", default="configs/step2.yaml")
    ap.add_argument("--run-name", default="r1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--mode", choices=["rerank", "similarity"], default="rerank")
    known, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest   # remaining flags (e.g. --batch-size) go to the Step 0 generator
    cfg = load_config(known.config)
    sfx = "_smoke" if known.smoke else ""
    adapter = run_dir(cfg, "step1", cfg["retrieval"]["step1_run"] + sfx) / "adapter"
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"No Step 1 adapter at {adapter} (finish Step 1 first).")
    rfile = run_dir(cfg, "step2", known.run_name + sfx) / (
        "retrieval.jsonl" if known.mode == "rerank" else "retrieval_similarity.jsonl")
    if not rfile.exists():
        raise SystemExit(f"No {rfile.name} (run scripts/step2_retrieve.py --mode {known.mode} first).")
    system = cfg["ablation"]["system"] + ("" if known.mode == "rerank" else "-sim")
    generate({"config": known.config, "run_name": known.run_name, "model": cfg["model_a"]["base"],
              "step": "step2", "system": system, "adapter": str(adapter), "retrieval": str(rfile),
              "smoke": known.smoke})


if __name__ == "__main__":
    main()
