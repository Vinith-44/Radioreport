"""Step 10: everything the app does, as plain Python (the Gradio UI in src/app/ui.py only calls this).

report(image, indication, prior) ->
  report text from the final system (src/models/systems.py spec) -> sentences -> CheXbert claims ->
  Step 2 probe on the MedSigLIP embedding of this image -> optional MedGemma boxes for positive claims ->
  Step 3 verdicts (src/verify/verdict.py) -> Step 2 fact-aware retrieval of supporting train cases.
summary(report, language)   -> Step 7 plain-language summary + translation + back-translation faithfulness flag.
second_reader(image, text)  -> Step 8 possible misses with the VAL-calibrated thresholds of a Step 8 run.
transcribe(audio, language) -> google/medasr (English, CTC, 16 kHz) or a configurable ASR model for Indic speech.
Models load lazily on first use; if the final system is Model A, the box model is that same MedGemma with the
LoRA adapter disabled (no second copy in memory).
"""
from __future__ import annotations

import contextlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from src.common import run_dir


class Pipeline:
    def __init__(self, cfg: dict, smoke: bool = False):
        self.cfg, self.ac, self.smoke = cfg, cfg["app"], smoke
        self.sfx = "_smoke" if smoke else ""
        self._sys = self._emb = self._probe = self._index = self._cx = self._loc = None
        self._tr = {}
        self._asr = {}
        self.spec = self._final_spec()

    # ------------------------------------------------------------------ loading
    def _final_spec(self) -> dict:
        if self.ac.get("system"):
            return dict(self.ac["system"])
        w = run_dir(self.cfg, "step5", self.ac["winner_run"] + self.sfx) / "winner.json"
        if not w.exists():
            raise SystemExit(f"no app.system set and no {w}")
        return json.loads(w.read_text(encoding="utf-8"))["winner_spec"]

    @property
    def system(self):
        if self._sys is None:
            from src.models.systems import System

            self._sys = System(self.cfg, self.spec, self.smoke)
        return self._sys

    def _s2(self) -> Path:
        return run_dir(self.cfg, "step2", self.ac["step2_run"] + self.sfx)

    @property
    def embedder(self):
        if self._emb is None:
            from src.models.medgemma_zeroshot import hf_token
            from src.retrieval.embed import SigLIPEmbedder

            rc = self.cfg["retrieval"]
            self._emb = SigLIPEmbedder(rc["encoder"], dtype=rc["dtype"], token=hf_token())
        return self._emb

    @property
    def probe(self):
        if self._probe is None:
            from src.models.probe import load_probe

            self._probe = load_probe(self._s2() / "probe")
        return self._probe

    @property
    def index(self):
        if self._index is None:
            from src.retrieval.index import FlatIP
            from src.retrieval.store import load_split, report_text

            ids, emb, lab = load_split(self._s2(), "train", labels=True)
            texts = [report_text(r, self.cfg["retrieval"]["report_text"]) for r in ids.to_dict("records")]
            self._index = (FlatIP(emb), ids, lab, texts)
        return self._index

    @property
    def chexbert(self):
        if self._cx is None:
            import torch
            from radeval.metrics.f1chexbert import F1CheXbert

            self._cx = F1CheXbert(device="cuda" if torch.cuda.is_available() else "cpu", batch_size=32)
        return self._cx

    def _localizer(self):
        """(model, processor, context-manager factory) for MedGemma boxes."""
        if self.spec["type"] == "a":
            m = self.system.model
            return m, self.system.processor, (m.disable_adapter if hasattr(m, "disable_adapter") else contextlib.nullcontext)
        if self._loc is None:
            from src.models.medgemma_zeroshot import load_model

            bc = self.cfg["box"]
            self._loc = load_model(self.cfg["models"][bc["model"]], self.ac["boxes"]["dtype"])
        return self._loc[0], self._loc[1], contextlib.nullcontext

    # ------------------------------------------------------------------ report + verification
    def probe_probs(self, image_path: str) -> np.ndarray:
        from src.models.probe import predict
        from src.retrieval.embed import load_for_siglip

        model, info = self.probe
        emb = self.embedder([load_for_siglip(image_path)])
        p, _ = predict(model, emb, info["thresholds"])
        return p[0], emb

    def boxes_for(self, image, findings: list[str]) -> dict[str, dict]:
        from src.models.medgemma_zeroshot import generate_batch
        from src.verify.boxes import box_zone, localize_prompt, pad_to_square, parse_boxes, patient_side, to_image_box

        if not findings:
            return {}
        model, proc, ctx = self._localizer()
        bc, band = self.cfg["box"], float(self.cfg["verify"]["side_center_band"])
        sq, pad = pad_to_square(image.convert("RGB"))
        with ctx():
            outs = generate_batch(model, proc, [sq] * len(findings),
                                  [localize_prompt(bc["phrases"][f]) for f in findings], int(bc["max_new_tokens"]))
        res = {}
        for f, o in zip(findings, outs):
            bx, st = parse_boxes(o["text"])
            ib = [to_image_box(b["box_2d"], pad) for b in (bx or [])]
            res[f] = {"status": st, "boxes": ib, "sides": [patient_side(b, band) for b in ib],
                      "zones": [box_zone(b) for b in ib]}
        return res

    def report(self, image, indication: str = "", prior: str = "") -> dict:
        from src.data.text_clean import parse_generated
        from src.models.probe import predict  # noqa: F401
        from src.retrieval.rerank import CHEXBERT_LABELS, rerank
        from src.verify.claims import chexbert_classes, claims_of, split_sentences
        from src.verify.verdict import NO_BOX, claim_verdict, probe_state, sentence_verdict

        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            path = f.name
        text = self.system.generate([{"image_path": path, "indication": indication, "prior": prior}], 1)[0]
        t_gen = time.time() - t0
        p = parse_generated(text)
        sents = [(sec, s) for sec in ("findings", "impression") for s in split_sentences(p[sec])]
        cls = chexbert_classes([s for _, s in sents], self.chexbert) if sents else np.zeros((0, 14), np.int8)
        probs, emb = self.probe_probs(path)
        thr = np.asarray(self.probe[1]["thresholds"])
        m = float(self.cfg["verify"]["strong_margin"])
        claims = [(sec, s, claims_of(s, c)) for (sec, s), c in zip(sents, cls)]
        boxes = {}
        if self.ac["boxes"]["enabled"]:
            want = sorted({c["finding"] for _, _, cl in claims for c in cl
                           if c["polarity"] == "positive" and c["finding"] not in NO_BOX})
            boxes = self.boxes_for(image, want)
        out_s, drawn = [], []
        for sec, s, cl in claims:
            vs, outc = [], []
            for c in cl:
                i = CHEXBERT_LABELS.index(c["finding"])
                bx = boxes.get(c["finding"]) if c["polarity"] == "positive" else None
                v, why = claim_verdict(c, probe_state(probs[i], thr[i], m), bx)
                vs.append(v)
                outc.append({**c, "verdict": v, "reason": why, "p": round(float(probs[i]), 3)})
                if bx and bx["status"] == "ok":
                    drawn += [{"finding": c["finding"], "box": b, "verdict": v} for b in bx["boxes"]]
            out_s.append({"section": sec, "text": s, "verdict": sentence_verdict(vs), "claims": outc})
        # supporting cases (Step 2 fact-aware retrieval over TRAIN images)
        index, ids, lab, texts = self.index
        rc = self.cfg["retrieval"]
        sims, idx = index.search(emb, int(rc["top_k"]))
        pbin = (probs >= thr).astype(np.int8)
        ci = idx[0]
        chosen = rerank(ci, sims[0], lab[ci], pbin, ids["patient_id"].iloc[ci].tolist(), [texts[i] for i in ci],
                        "__app_query__", int(rc["n_refs"]), True)
        cases = [{"study_id": str(ids["study_id"].iloc[c["index"]]), "sim": c["sim"], "agreement": c["agreement"],
                  "snippet": " ".join(texts[c["index"]].split())[:240]} for c in chosen]
        Path(path).unlink(missing_ok=True)
        return {"system": self.system.name, "indication": indication, "report": text, "findings": p["findings"],
                "impression": p["impression"], "sentences": out_s, "boxes": drawn, "cases": cases,
                "probe": {n: round(float(v), 3) for n, v in zip(CHEXBERT_LABELS, probs)},
                "timing_s": {"generate": round(t_gen, 2), "total": round(time.time() - t0, 2)}}

    # ------------------------------------------------------------------ Step 7 + Step 8 + ASR
    def summary(self, report: str, language: str) -> dict:
        from src.multilingual.translate import LANGS, SIMPLIFY_PROMPT, LLMTranslator, label_agreement, text_generate, translate_text
        from src.retrieval.rerank import chexbert_labels

        model, proc, ctx = self._localizer()          # the base MedGemma 1.5 (adapter disabled for Model A)
        with ctx():
            en = text_generate(model, proc, [SIMPLIFY_PROMPT.format(report=report)], 300)[0]
        out = {"language": language, "summary_en": en, "translation": "", "back_translation": "", "faithful": True,
               "lost": [], "added": []}
        if language == "english":
            return out
        mc = self.cfg["multilingual"]
        if mc["backend"] == "llm":
            fwd = back = LLMTranslator(model, proc)
        else:
            if "fwd" not in self._tr:
                import torch

                from src.models.medgemma_zeroshot import hf_token
                from src.multilingual.translate import IndicTrans2

                dev = "cuda" if torch.cuda.is_available() else "cpu"
                it = mc["indictrans2"]
                self._tr["fwd"] = IndicTrans2("en-indic", it["en_indic"], dev, hf_token(), it["dtype"])
                self._tr["back"] = IndicTrans2("indic-en", it["indic_en"], dev, hf_token(), it["dtype"])
            fwd, back = self._tr["fwd"], self._tr["back"]
        with ctx():
            out["translation"] = translate_text(fwd, en, "eng_Latn", LANGS[language])
            out["back_translation"] = translate_text(back, out["translation"], LANGS[language], "eng_Latn")
        lab = chexbert_labels([report, out["back_translation"]])
        a = label_agreement(lab[0].tolist(), lab[1].tolist())
        out.update(faithful=a["exact"], lost=a["lost"], added=a["added"])
        return out

    def second_reader(self, image, report_text: str) -> dict:
        from src.verify.claims import chexbert_classes, split_sentences
        from src.verify.second_reader import flag, mentioned

        tp = run_dir(self.cfg, "step8", self.ac["step8_run"] + self.sfx) / "thresholds.json"
        if not tp.exists():
            raise SystemExit(f"no {tp} (run step8_second_reader.py first)")
        tau = json.loads(tp.read_text(encoding="utf-8"))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name)
            probs, _ = self.probe_probs(f.name)
        Path(f.name).unlink(missing_ok=True)
        sents = split_sentences(report_text)
        cls = chexbert_classes(sents, self.chexbert) if sents else np.zeros((0, 14), np.int8)
        return {"mentioned": sorted(mentioned(cls)), "possible_misses": flag(probs, mentioned(cls), tau)}

    def transcribe(self, audio_path: str, language: str = "english") -> str:
        from transformers import pipeline as hf_pipeline

        asr = self.ac["asr"]
        mid = asr["english"] if language == "english" else asr["indic"]
        if mid not in self._asr:
            from src.models.medgemma_zeroshot import hf_token

            self._asr[mid] = hf_pipeline("automatic-speech-recognition", model=mid, token=hf_token())
        kw = {"chunk_length_s": 20, "stride_length_s": 2}       # MedASR model card settings
        if language != "english" and "whisper" in mid:
            kw = {"generate_kwargs": {"language": language, "task": "transcribe"}}
        return self._asr[mid](audio_path, **kw)["text"].strip()
