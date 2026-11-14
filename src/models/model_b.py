"""Model B = frozen medical image encoder + custom projector + Gemma 3 text LLM (CLAUDE.md §5).

  encoder   : google/medsiglip-448 vision tower, frozen; patch tokens = vision_model(...).last_hidden_state
              (SigLIP has no CLS token; the grid side is read from the sequence length, it must be even)
  projector : 2x2 patch merge (concatenate each 2x2 neighbourhood -> 4*d_vis) -> Linear -> GELU -> Linear(d_llm)
  LLM       : google/gemma-3-1b-it (Gemma3ForCausalLM), frozen in Step 4, LoRA in Step 5

Verified against transformers 5.17 source: Gemma3TextScaledWordEmbedding multiplies token embeddings by
sqrt(hidden) INSIDE the embedding layer, and a forward with inputs_embeds skips that layer. So text embeddings are
taken from llm.get_input_embeddings() (already scaled) and the projector's outputs are inserted unscaled (it
learns the right scale). generate(inputs_embeds=...) starts from an empty input_ids, i.e. returns new tokens only.

Prompt = the Gemma chat template rendered with an image placeholder that is then replaced by the image tokens:
  <bos><start_of_turn>user\n [IMAGE TOKENS] \n{prompt}<end_of_turn>\n<start_of_turn>model\n  ->  {report}<end_of_turn>
"""
from __future__ import annotations

import torch
import torch.nn as nn

IMG = "<IMAGE_PLACEHOLDER>"


class Projector(nn.Module):
    def __init__(self, d_vis: int, d_llm: int, merge: int = 2, hidden: int | None = None):
        super().__init__()
        self.merge, self.d_vis = merge, d_vis
        d_in = d_vis * merge * merge
        self.fc1 = nn.Linear(d_in, hidden or d_llm)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden or d_llm, d_llm)

    def merge_patches(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        g = int(round(n ** 0.5))
        m = self.merge
        if g * g != n or g % m:
            raise ValueError(f"patch grid {n} is not an even square for a {m}x{m} merge")
        x = x.view(b, g // m, m, g // m, m, d).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(b, (g // m) ** 2, m * m * d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(self.merge_patches(x))))


def prompt_pieces(tokenizer, prompt: str) -> tuple[list[int], list[int]]:
    """Token ids before and after the image tokens, from the model's own chat template."""
    text = tokenizer.apply_chat_template([{"role": "user", "content": f"{IMG}\n{prompt}"}],
                                         add_generation_prompt=True, tokenize=False)
    if text.count(IMG) != 1:
        raise AssertionError("image placeholder not preserved by the chat template")
    pre, post = text.split(IMG)
    ids = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    pre_ids, post_ids = ids(pre), ids(post)
    if tokenizer.bos_token_id is not None and pre_ids.count(tokenizer.bos_token_id) != 1:
        raise AssertionError(f"expected exactly one <bos> in the prompt prefix, got {pre_ids}")
    return pre_ids, post_ids


def end_of_turn_id(tokenizer) -> int:
    i = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if i is None or i == tokenizer.unk_token_id:
        raise AssertionError("tokenizer has no <end_of_turn> token")
    return int(i)


def target_ids(tokenizer, report: str) -> list[int]:
    return tokenizer(report, add_special_tokens=False)["input_ids"] + [end_of_turn_id(tokenizer)]


def assemble(embed, img_tok: torch.Tensor, pre: list[list[int]], post: list[list[int]],
             tgt: list[list[int]] | None, pad_left: bool) -> dict:
    """Build inputs_embeds / attention_mask (/ labels) for a batch.
    embed: the LLM's (scaled) input embedding layer; img_tok: (B, T_img, d) projector output.
    Training: right padding, labels only on the target. Generation (tgt=None): left padding."""
    dev, dt = img_tok.device, img_tok.dtype
    seqs, labs = [], []
    for b in range(img_tok.shape[0]):
        e = [embed(torch.tensor(pre[b], device=dev)).to(dt), img_tok[b],
             embed(torch.tensor(post[b], device=dev)).to(dt)]
        n_prompt = len(pre[b]) + img_tok.shape[1] + len(post[b])
        lab = [-100] * n_prompt
        if tgt is not None:
            e.append(embed(torch.tensor(tgt[b], device=dev)).to(dt))
            lab += tgt[b]
        seqs.append(torch.cat(e, 0))
        labs.append(lab)
    L = max(s.shape[0] for s in seqs)
    d = seqs[0].shape[1]
    emb = torch.zeros(len(seqs), L, d, device=dev, dtype=dt)
    att = torch.zeros(len(seqs), L, device=dev, dtype=torch.long)
    lab = torch.full((len(seqs), L), -100, device=dev, dtype=torch.long)
    for b, s in enumerate(seqs):
        n = s.shape[0]
        sl = slice(L - n, L) if pad_left else slice(0, n)
        emb[b, sl], att[b, sl] = s, 1
        lab[b, sl] = torch.tensor(labs[b], device=dev)
    out = {"inputs_embeds": emb, "attention_mask": att}
    if tgt is not None:
        out["labels"] = lab
    return out


class ModelB(nn.Module):
    def __init__(self, vision, llm, projector: Projector, enc_dtype=torch.float16):
        super().__init__()
        self.vision, self.llm, self.projector, self.enc_dtype = vision, llm, projector, enc_dtype

    @torch.no_grad()
    def encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = self.vision(pixel_values=pixel_values.to(self.enc_dtype)).last_hidden_state
        if not torch.isfinite(h).all():
            raise FloatingPointError("non-finite encoder features (fp16 overflow?) - use encoder dtype fp32")
        return h

    def image_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.encode(pixel_values).to(next(self.projector.parameters()).dtype)   # works with DDP too
        return self.projector(feats)

    def forward(self, pixel_values, pre, post, tgt):
        img = self.image_tokens(pixel_values).to(self.llm.dtype)
        batch = assemble(self.llm.get_input_embeddings(), img, pre, post, tgt, pad_left=False)
        return self.llm(**batch)

    @torch.no_grad()
    def generate(self, pixel_values, pre, post, tokenizer, max_new_tokens: int = 200) -> list[str]:
        img = self.image_tokens(pixel_values).to(self.llm.dtype)
        batch = assemble(self.llm.get_input_embeddings(), img, pre, post, None, pad_left=True)
        eos = [end_of_turn_id(tokenizer)] + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
        out = self.llm.generate(**batch, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos,
                                pad_token_id=tokenizer.pad_token_id)
        return [t.strip() for t in tokenizer.batch_decode(out, skip_special_tokens=True)]


def load_components(enc_id: str, llm_id: str, enc_dtype: str, llm_dtype: str, device: str, token: str | None):
    from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

    dts = {"fp16": torch.float16, "fp32": torch.float32}
    siglip = AutoModel.from_pretrained(enc_id, token=token, dtype=dts[enc_dtype])
    vision = siglip.vision_model.to(device).eval()
    del siglip
    for p in vision.parameters():
        p.requires_grad = False
    proc = AutoProcessor.from_pretrained(enc_id, token=token)
    llm = AutoModelForCausalLM.from_pretrained(llm_id, token=token, dtype=dts[llm_dtype]).to(device)
    tok = AutoTokenizer.from_pretrained(llm_id, token=token)
    return vision, proc, llm, tok


def load_trained(run: "Path", device: str, token: str | None = None):
    """Rebuild a trained Model B from its run folder: projector.json (settings), projector.pt, optional lora/.
    Returns (ModelB in eval mode, image processor, tokenizer, meta)."""
    import json
    from pathlib import Path

    run = Path(run)
    meta = json.loads((run / "projector.json").read_text(encoding="utf-8"))
    vision, proc, llm, tok = load_components(meta["encoder"], meta["llm"], meta["dtypes"][0], meta["dtypes"][1],
                                             device, token)
    if (run / "lora" / "adapter_config.json").exists():
        from peft import PeftModel

        llm = PeftModel.from_pretrained(llm, str(run / "lora"))
    proj = Projector(meta["d_vis"], meta["d_llm"], int(meta["merge"]), meta.get("projector_hidden")).to(device)
    proj.load_state_dict(torch.load(run / "projector.pt", map_location="cpu"))
    model = ModelB(vision, llm.eval(), proj.eval(), torch.float16 if meta["dtypes"][0] == "fp16" else torch.float32)
    return model.eval(), proc, tok, meta
