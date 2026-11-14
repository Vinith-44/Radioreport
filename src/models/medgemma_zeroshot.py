"""Zero-shot MedGemma (1 / 1.5, 4B-it) batched report generation.

API verified against transformers 5.3 source and the google/medgemma-1.5-4b-it model card:
AutoModelForImageTextToText + AutoProcessor; chat messages with {"type": "image"} + {"type": "text"}.
Prompts are rendered with apply_chat_template(tokenize=False) and then encoded by the processor with
left padding, which is the version-stable path for batched multimodal generation.
"""
from __future__ import annotations

import os

from PIL import Image

DTYPES = ("fp32", "fp16", "nf4")


def hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:  # Kaggle Secrets (CLAUDE.md §2)
        from kaggle_secrets import UserSecretsClient

        tok = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = tok
        return tok
    except Exception:
        return None


def adapter_digest(adapter: str | None) -> str | None:
    """sha256 of the adapter weights, so one predictions file can never mix two checkpoints."""
    if not adapter:
        return None
    import hashlib
    from pathlib import Path

    files = sorted(Path(adapter).glob("adapter_model.*"))
    if not files:
        raise FileNotFoundError(f"no adapter_model.* in {adapter}")
    h = hashlib.sha256()
    with open(files[0], "rb") as f:
        for chunk in iter(lambda: f.read(8 * 2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_model(model_id: str, dtype: str, adapter: str | None = None):
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}")
    if dtype == "fp16":
        print("WARNING: Gemma 3 overflows in fp16 on T4 (empty/NaN outputs). The empty-output guard will abort.")
    kwargs: dict = {"device_map": "auto", "token": hf_token()}
    major = int(transformers.__version__.split(".")[0])
    dkey = "dtype" if major >= 5 else "torch_dtype"
    if dtype == "fp32":
        kwargs[dkey] = torch.float32
    elif dtype == "fp16":
        kwargs[dkey] = torch.float16
    else:
        from transformers import BitsAndBytesConfig

        # fp32 compute: fp16 compute would hit the same Gemma 3 overflow.
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32)
        kwargs[dkey] = torch.float32
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs).eval()
    if adapter:
        # Step 1+: a LoRA adapter on top of the same base (the processor is the base model's, unchanged).
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).eval()
    processor = AutoProcessor.from_pretrained(model_id, token=kwargs["token"])
    processor.tokenizer.padding_side = "left"
    return model, processor


def build_prompt(template: str, indication: str, empty_indication: str, references: str | None = None) -> str:
    """references: the Step 2 retrieval block; required iff the template has a {references} slot."""
    ind = (indication or "").strip() or empty_indication
    if "{references}" in template:
        if not references:
            raise ValueError("this prompt template needs retrieved references")
        return template.format(indication=ind, references=references)
    if references:
        raise ValueError("references given but the template has no {references} slot")
    return template.format(indication=ind)


def to_rgb8(im: Image.Image) -> Image.Image:
    """Any X-ray image -> 8-bit RGB.

    ReXGradient PNGs are 16-bit grayscale (mode I;16). PIL's convert("RGB") CLIPS those values at 255
    (verified: ~94% of pixels of a 12-bit ramp turn white), so high-bit-depth images are min-max rescaled.
    """
    if im.mode in ("I;16", "I;16B", "I;16L", "I", "F"):
        import numpy as np

        a = np.asarray(im, dtype=np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(a)
        im = Image.fromarray(a.round().astype(np.uint8))  # 2-D uint8 -> mode "L" (no mode=: removed in Pillow 13)
    return im.convert("RGB")


def load_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return to_rgb8(im)


def generate_batch(model, processor, images: list[Image.Image], prompts: list[str],
                   max_new_tokens: int, do_sample: bool = False) -> list[dict]:
    import torch

    msgs = [[{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": p}]}] for p in prompts]
    texts = [processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
    # The rendered chat text already starts with <bos>; add_special_tokens=False avoids a second one
    # (verified: ids then equal the model card's apply_chat_template(tokenize=True) path).
    inputs = processor(text=texts, images=[[im] for im in images], padding=True, return_tensors="pt",
                       add_special_tokens=False)
    inputs = inputs.to(model.device)
    if "pixel_values" in inputs:
        vt_dtype = next(model.parameters()).dtype
        if vt_dtype in (torch.float16, torch.float32):
            inputs["pixel_values"] = inputs["pixel_values"].to(vt_dtype)
    pad_id = processor.tokenizer.pad_token_id
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    gen = out[:, inputs["input_ids"].shape[1]:]
    decoded = processor.batch_decode(gen, skip_special_tokens=True)
    n_tok = (gen != pad_id).sum(dim=1).tolist() if pad_id is not None else [gen.shape[1]] * len(decoded)
    return [{"text": t.strip(), "n_new_tokens": int(n), "hit_max_tokens": int(n) >= max_new_tokens}
            for t, n in zip(decoded, n_tok)]
