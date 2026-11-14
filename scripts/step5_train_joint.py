"""Step 5a: Model B phase 2 - projector + LoRA on the LLM, real reports, indication + prior report in the prompt.

  python scripts/step5_train_joint.py --run-name b2 --smoke
  torchrun --nproc_per_node 2 scripts/step5_train_joint.py --run-name b2

The Step 4 trainer with configs/step5.yaml (two learning rates, LoRA, init from the Step 4 projector).
Writes outputs/runs/step5/<run>/: projector.pt, lora/ (peft adapter), checkpoint.pt, acceptance.json, ...
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv[1:1] = ["--config", "configs/step5.yaml"]
    from step4_train_projector import main

    main()
