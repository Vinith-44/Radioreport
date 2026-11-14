"""Step 10a: launch the Gradio app.

  python scripts/step10_app.py                 # local
  python scripts/step10_app.py --share         # Kaggle / Colab: public gradio.live link (keep it private!)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step10.yaml")
    ap.add_argument("--smoke", action="store_true", help="use the *_smoke runs of every step")
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    cfg = load_config(args.config)
    from src.app.pipeline import Pipeline
    from src.app.ui import build

    demo = build(Pipeline(cfg, args.smoke), cfg)
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
