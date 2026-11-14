"""One interface to generate reports with either final system type (used by Steps 6-8, whichever model won Step 5).

spec (as written by step5_score.py winner.json, or in a config):
  {"type": "a", "step": "step1", "run": "a1", "system": "model-a"}   MedGemma 1.5 + LoRA adapter (<run>/adapter)
  {"type": "b", "step": "step5", "run": "b2", "system": "model-b"}   Model B (<run>/projector.pt [+ lora/])
Both use the shared prompt template, greedy decoding and generation.max_new_tokens.
"""
from __future__ import annotations

import sys
from pathlib import Path

from src.common import run_dir


def spec_dir(cfg: dict, spec: dict, smoke: bool) -> Path:
    return run_dir(cfg, spec["step"], spec["run"] + ("_smoke" if smoke else ""))


class System:
    def __init__(self, cfg: dict, spec: dict, smoke: bool = False, device: str | None = None):
        import torch

        from src.models.medgemma_zeroshot import hf_token

        self.cfg, self.spec, self.g = cfg, spec, cfg["generation"]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        d = spec_dir(cfg, spec, smoke)
        if spec["type"] == "a":
            from src.models.medgemma_zeroshot import load_model

            adapter = d / "adapter"
            if not (adapter / "adapter_config.json").exists():
                raise SystemExit(f"no Model A adapter at {adapter}")
            self.model, self.processor = load_model(cfg["models"][cfg["model_a"]["base"]], self.g["dtype"], str(adapter))
            self.use_prior = False
        elif spec["type"] == "b":
            import json

            from src.models.model_b import load_trained

            self.model, self.processor, self.tok, meta = load_trained(d, self.device, hf_token())
            self.use_prior = bool(json.loads((d / "projector.json").read_text(encoding="utf-8")).get("use_prior"))
        else:
            raise ValueError(spec)
        self.name = f"{spec['system']}@{spec['step']}/{spec['run']}"

    def generate(self, rows: list[dict], batch_size: int = 8, max_new_tokens: int | None = None) -> list[str]:
        """rows: dicts with image_path, indication (and prior for Model B)."""
        mx = int(max_new_tokens or self.g["max_new_tokens"])
        out = []
        if self.spec["type"] == "a":
            from src.models.medgemma_zeroshot import build_prompt, generate_batch, load_image

            for i in range(0, len(rows), batch_size):
                ch = rows[i:i + batch_size]
                res = generate_batch(self.model, self.processor, [load_image(r["image_path"]) for r in ch],
                                     [build_prompt(self.g["prompt_template"], r.get("indication"),
                                                   self.g["empty_indication"]) for r in ch], mx, False)
                out += [x["text"] for x in res]
            return out
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from torch.utils.data import DataLoader

        from step4_train_projector import AlignDS, collate

        rows = [{**r, "target": "", "prior": r.get("prior", "")} for r in rows]
        dl = DataLoader(AlignDS(rows, self.processor, self.tok, self.g["prompt_template"], self.g["empty_indication"],
                                with_target=False, use_prior=self.use_prior), batch_size=batch_size, collate_fn=collate)
        for b in dl:
            out += self.model.generate(b["px"].to(self.device), b["pre"], b["post"], self.tok, mx)
        return out
