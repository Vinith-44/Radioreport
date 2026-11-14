"""Step 9 policies: the same three operations for both model types.

  sample(rows, G)          -> per prompt, G sampled completions (token ids, ending with the stop token unless cut)
  logprobs(rows, comps)    -> per completion, log pi(token) of each completion token (WITH grad, LoRA on)
  ref_logprobs(rows, comps)-> the same under the SFT model (peft disable_adapter(), no grad)
Only the LoRA weights train ("GRPO on the best model's LoRA"); Model B's projector and encoder stay frozen.
"""
from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import torch

from src.rl.grpo import token_logprobs


def _cut(ids: list[int], stops: set[int], pad: int | None) -> tuple[list[int], bool]:
    """Completion up to and including the first stop token; truncated=True if there is none."""
    out = []
    for t in ids:
        if pad is not None and t == pad and not out:
            continue
        out.append(t)
        if t in stops:
            return out, False
    return out, True


class PolicyB:
    def __init__(self, run: Path, device: str, token: str | None):
        from peft import PeftModel

        from src.models.model_b import Projector, end_of_turn_id, load_components

        import json

        meta = json.loads((run / "projector.json").read_text(encoding="utf-8"))
        vision, self.proc, llm, self.tok = load_components(meta["encoder"], meta["llm"], meta["dtypes"][0],
                                                           meta["dtypes"][1], device, token)
        if not (run / "lora" / "adapter_config.json").exists():
            raise SystemExit(f"Model B at {run} has no LoRA (GRPO trains the LoRA of a Step 5 model)")
        self.llm = PeftModel.from_pretrained(llm, str(run / "lora"), is_trainable=True)
        self.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.proj = Projector(meta["d_vis"], meta["d_llm"], int(meta["merge"]), meta.get("projector_hidden")).to(device)
        self.proj.load_state_dict(torch.load(run / "projector.pt", map_location="cpu"))
        for p in list(self.proj.parameters()) + list(vision.parameters()):
            p.requires_grad = False
        from src.models.model_b import ModelB

        self.model = ModelB(vision, self.llm, self.proj, torch.float16 if meta["dtypes"][0] == "fp16" else torch.float32)
        self.device, self.meta = device, meta
        self.stops = {end_of_turn_id(self.tok)} | ({self.tok.eos_token_id} if self.tok.eos_token_id is not None else set())
        self.use_prior = bool(meta.get("use_prior"))

    def trainable(self):
        return [p for p in self.llm.parameters() if p.requires_grad]

    def _inputs(self, rows, g):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
        from step4_train_projector import AlignDS, collate

        ds = AlignDS([{**r, "target": "", "prior": r.get("prior", "")} for r in rows], self.proc, self.tok,
                     g["prompt_template"], g["empty_indication"], with_target=False, use_prior=self.use_prior)
        return collate([ds[i] for i in range(len(ds))])

    @torch.no_grad()
    def sample(self, rows, g: dict, G: int, max_new: int, temperature: float):
        from src.models.model_b import assemble

        b = self._inputs(rows, g)
        img = self.model.image_tokens(b["px"].to(self.device)).to(self.llm.dtype)
        batch = assemble(self.llm.get_input_embeddings(), img, b["pre"], b["post"], None, pad_left=True)
        emb = batch["inputs_embeds"].repeat_interleave(G, 0)
        att = batch["attention_mask"].repeat_interleave(G, 0)
        out = self.llm.generate(inputs_embeds=emb, attention_mask=att, do_sample=True, temperature=temperature,
                                top_p=1.0, max_new_tokens=max_new, eos_token_id=sorted(self.stops),
                                pad_token_id=self.tok.pad_token_id)
        comps = [_cut(r.tolist(), self.stops, self.tok.pad_token_id) for r in out]
        texts = [self.tok.decode(c, skip_special_tokens=True).strip() for c, _ in comps]
        return [c for c, _ in comps], [t for _, t in comps], texts, b

    def _lp(self, b, comps, G):
        from src.models.model_b import assemble

        with torch.no_grad():
            img = self.model.image_tokens(b["px"].to(self.device)).to(self.llm.dtype)
        img = img.repeat_interleave(G, 0)
        pre = [p for p in b["pre"] for _ in range(G)]
        post = [p for p in b["post"] for _ in range(G)]
        batch = assemble(self.llm.get_input_embeddings(), img, pre, post, [c if c else [self.tok.pad_token_id]
                                                                           for c in comps], pad_left=False)
        logits = self.llm(inputs_embeds=batch["inputs_embeds"], attention_mask=batch["attention_mask"]).logits
        lab = batch["labels"]
        out = []
        for k, c in enumerate(comps):
            pos = (lab[k, 1:] != -100).nonzero().squeeze(-1)
            lp = token_logprobs(logits[k, :-1][pos], lab[k, 1:][pos]) if c else logits.new_zeros(0)
            out.append(lp)
        return out

    def logprobs(self, b, comps, G):
        return self._lp(b, comps, G)

    @torch.no_grad()
    def ref_logprobs(self, b, comps, G):
        with self.llm.disable_adapter():
            return self._lp(b, comps, G)

    def save(self, out: Path, source_run: Path):
        import shutil

        out.mkdir(parents=True, exist_ok=True)
        self.llm.save_pretrained(str(out / "lora"))
        for f in ("projector.pt", "projector.json"):
            shutil.copy(source_run / f, out / f)


class PolicyA:
    """MedGemma 1.5 (nf4 base, fp32 compute - fp16 overflows on T4) + the Step 1 LoRA, trainable."""

    def __init__(self, cfg: dict, adapter: Path, device: str, token: str | None):
        import transformers
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

        mid = cfg["models"][cfg["model_a"]["base"]]
        dkey = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        base = AutoModelForImageTextToText.from_pretrained(
            mid, device_map="auto", token=token, **{dkey: torch.float32},
            quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                   bnb_4bit_compute_dtype=torch.float32))
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        self.model = PeftModel.from_pretrained(base, str(adapter), is_trainable=True)
        self.processor = AutoProcessor.from_pretrained(mid, token=token)
        self.processor.tokenizer.padding_side = "left"
        self.tok = self.processor.tokenizer
        self.stops = {self.tok.convert_tokens_to_ids("<end_of_turn>"), self.tok.eos_token_id}
        self.device = device

    def trainable(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _inputs(self, rows, g):
        from src.models.medgemma_zeroshot import build_prompt, load_image

        msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": build_prompt(
            g["prompt_template"], r.get("indication"), g["empty_indication"])}]}] for r in rows]
        texts = [self.processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
        return self.processor(text=texts, images=[[load_image(r["image_path"])] for r in rows], padding=True,
                              return_tensors="pt", add_special_tokens=False).to(self.model.device)

    @torch.no_grad()
    def sample(self, rows, g: dict, G: int, max_new: int, temperature: float):
        b = self._inputs(rows, g)
        out = self.model.generate(**b, do_sample=True, temperature=temperature, top_p=1.0, max_new_tokens=max_new,
                                  num_return_sequences=G, eos_token_id=sorted(self.stops),
                                  pad_token_id=self.tok.pad_token_id)
        gen = out[:, b["input_ids"].shape[1]:]
        comps = [_cut(r.tolist(), self.stops, self.tok.pad_token_id) for r in gen]
        texts = [self.tok.decode(c, skip_special_tokens=True).strip() for c, _ in comps]
        return [c for c, _ in comps], [t for _, t in comps], texts, b

    def _lp(self, b, comps, G):
        P = b["input_ids"].shape[1]
        L = max(1, max(len(c) for c in comps))
        pad = self.tok.pad_token_id
        ids = b["input_ids"].repeat_interleave(G, 0)
        att = b["attention_mask"].repeat_interleave(G, 0)
        comp = torch.full((len(comps), L), pad, dtype=ids.dtype, device=ids.device)
        cm = torch.zeros((len(comps), L), dtype=att.dtype, device=ids.device)
        for k, c in enumerate(comps):
            if c:
                comp[k, :len(c)] = torch.tensor(c, device=ids.device)
                cm[k, :len(c)] = 1
        kw = {"input_ids": torch.cat([ids, comp], 1), "attention_mask": torch.cat([att, cm], 1),
              "pixel_values": b["pixel_values"].repeat_interleave(G, 0)}
        if "token_type_ids" in b:   # Gemma 3: 1 on image tokens; completion tokens are text (0)
            kw["token_type_ids"] = torch.cat([b["token_type_ids"].repeat_interleave(G, 0), torch.zeros_like(comp)], 1)
        logits = self.model(**kw).logits
        return [token_logprobs(logits[k, P - 1:P - 1 + len(c)], comp[k, :len(c)]) if c else logits.new_zeros(0)
                for k, c in enumerate(comps)]

    def logprobs(self, b, comps, G):
        return self._lp(b, comps, G)

    @torch.no_grad()
    def ref_logprobs(self, b, comps, G):
        ctx = self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else contextlib.nullcontext()
        with ctx:
            return self._lp(b, comps, G)

    def save(self, out: Path, source_run: Path):
        out.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(out / "adapter"))
