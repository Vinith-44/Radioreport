"""Step 7: plain-language summary + English <-> Indian-language translation backends.

IndicTrans2 (ai4bharat/indictrans2-en-indic-1B and -indic-en-1B; gated, MIT) used as on its model card:
  IndicTransToolkit.processor.IndicProcessor(inference=True).preprocess_batch(sentences, src_lang, tgt_lang)
  -> AutoTokenizer/AutoModelForSeq2SeqLM(trust_remote_code=True) -> generate(num_beams=5, max_length=256,
  min_length=0, use_cache=True) -> batch_decode(skip_special_tokens=True) -> ip.postprocess_batch(.., lang=tgt).
  The card uses flash_attention_2, which the T4 does not support, so the default attention is used. The card also
  warns about transformers v5 (its custom code predates it): if loading fails, use backend "llm".
"llm" backend: the report-writing LLM itself translates (Gemma 3 / MedGemma are multilingual).
Translation is sentence by sentence (IndicTrans2 is trained on sentences; max_length 256).
"""
from __future__ import annotations

LANGS = {"hindi": "hin_Deva", "tamil": "tam_Taml", "telugu": "tel_Telu", "bengali": "ben_Beng",
         "marathi": "mar_Deva", "kannada": "kan_Knda", "malayalam": "mal_Mlym", "gujarati": "guj_Gujr"}

SIMPLIFY_PROMPT = """Rewrite the chest X-ray report below for the patient, in plain everyday English.
Rules: short sentences; explain medical words simply; keep EVERY finding, including normal ones and which side
(left/right); do not add any finding, diagnosis, cause, treatment or advice that is not in the report.
Reply with the summary only.

Report:
{report}"""

TRANSLATE_PROMPT = """Translate the following text into {language}. Keep the meaning exactly, including which side
(left/right) and every finding. Reply with the translation only.

Text:
{text}"""


def text_generate(model, processor, prompts: list[str], max_new_tokens: int = 300) -> list[str]:
    """Text-only chat generation with a Gemma-3-family model loaded by medgemma_zeroshot.load_model."""
    import torch

    msgs = [[{"role": "user", "content": [{"type": "text", "text": p}]}] for p in prompts]
    texts = [processor.apply_chat_template(m, add_generation_prompt=True, tokenize=False) for m in msgs]
    inputs = processor(text=texts, padding=True, return_tensors="pt", add_special_tokens=False).to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return [t.strip() for t in processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)]


class IndicTrans2:
    def __init__(self, direction: str, model_id: str, device: str, token: str | None, dtype: str = "fp16"):
        import torch
        from IndicTransToolkit.processor import IndicProcessor
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch, self.device, self.direction = torch, device, direction
        self.ip = IndicProcessor(inference=True)
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, token=token)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, trust_remote_code=True, token=token,
            dtype={"fp16": torch.float16, "fp32": torch.float32}[dtype]).to(device).eval()

    def __call__(self, sentences: list[str], src: str, tgt: str, batch_size: int = 16) -> list[str]:
        out = []
        for i in range(0, len(sentences), batch_size):
            batch = self.ip.preprocess_batch(sentences[i:i + batch_size], src_lang=src, tgt_lang=tgt)
            enc = self.tok(batch, truncation=True, padding="longest", return_tensors="pt",
                           return_attention_mask=True).to(self.device)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, use_cache=True, min_length=0, max_length=256, num_beams=5,
                                          num_return_sequences=1)
            dec = self.tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            out += self.ip.postprocess_batch(dec, lang=tgt)
        return out


class LLMTranslator:
    def __init__(self, model, processor, max_new_tokens: int = 200):
        self.model, self.processor, self.mx = model, processor, max_new_tokens

    def __call__(self, sentences: list[str], src: str, tgt: str, batch_size: int = 8) -> list[str]:
        name = {v: k.capitalize() for k, v in LANGS.items()}.get(tgt, "English")
        out = []
        for i in range(0, len(sentences), batch_size):
            out += text_generate(self.model, self.processor,
                                 [TRANSLATE_PROMPT.format(language=name, text=s) for s in sentences[i:i + batch_size]],
                                 self.mx)
        return out


def translate_text(fn, text: str, src: str, tgt: str) -> str:
    """Sentence-wise translation of a whole summary; sentence order is kept."""
    from src.verify.claims import split_sentences

    sents = split_sentences(text) or ([text] if text.strip() else [])
    return " ".join(fn(sents, src, tgt)) if sents else ""


def label_agreement(ref: list[int], other: list[int]) -> dict:
    """14 CheXbert rrg labels: exact match + which labels differ (missing = in the report but lost)."""
    from src.retrieval.rerank import CHEXBERT_LABELS

    diff = [n for n, a, b in zip(CHEXBERT_LABELS, ref, other) if a != b]
    return {"exact": not diff, "lost": [n for n, a, b in zip(CHEXBERT_LABELS, ref, other) if a and not b],
            "added": [n for n, a, b in zip(CHEXBERT_LABELS, ref, other) if b and not a],
            "per_label_agree": sum(a == b for a, b in zip(ref, other)) / len(ref)}
