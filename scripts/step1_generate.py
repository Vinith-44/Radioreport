"""Step 1b: test predictions of Model A (MedGemma 1.5 + the Step 1 LoRA adapter). Resumable.

  python scripts/step1_generate.py --run-name a1 --smoke
  python scripts/step1_generate.py --run-name a1

Exactly the Step 0 generation code (scripts/step0_zeroshot_generate.py): same fixed test subset
(step0_eval_ids.json), prompt, greedy decoding and max_new_tokens; only the adapter (and the nf4 base it was
trained on, configs/step1.yaml generation.dtype) differ. Writes
outputs/runs/step1/<run>[_smoke]/model-a/<dataset>/predictions.jsonl. Any step0 flag can be passed through.
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
    ap.add_argument("--config", default="configs/step1.yaml")
    ap.add_argument("--run-name", default="a1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--adapter")
    known, _ = ap.parse_known_args()
    cfg = load_config(known.config)
    adapter = known.adapter or str(run_dir(cfg, "step1", known.run_name + ("_smoke" if known.smoke else "")) / "adapter")
    if not (Path(adapter) / "adapter_config.json").exists():
        raise SystemExit(f"No trained adapter at {adapter} (run scripts/step1_train_qlora.py to completion first).")
    generate({"config": known.config, "run_name": known.run_name, "model": cfg["model_a"]["base"],
              "step": "step1", "system": cfg["model_a"]["system"], "adapter": adapter})


if __name__ == "__main__":
    main()
