"""Step 4 (and Step 5): Model B training. Resumable; optional 2-GPU data parallel.

Phase 1 (configs/step4.yaml): ONLY the projector trains (encoder + LLM frozen).
Phase 2 (configs/step5.yaml, via scripts/step5_train_joint.py): projector (higher lr) + LoRA on the LLM (lower lr),
real reports only, prompt with indication + prior report when available; initialised from the Step 4 projector.

  python scripts/step4_train_projector.py --run-name b1 --smoke
  python scripts/step4_train_projector.py --run-name b1                    # one T4
  torchrun --nproc_per_node 2 scripts/step4_train_projector.py --run-name b1   # both T4s (data parallel)

Data: real TRAIN reports (ReXGradient subset + IU) + template reports from CheXpert-train / NIH-train_val labels.
Writes outputs/runs/step4/<run>[_smoke]/: checkpoint.pt (resume), projector.pt + projector.json (final),
losses.jsonl, val_samples.jsonl, acceptance.json. Rerun the same command to resume after a timeout.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import append_jsonl, data_dir, load_config, read_jsonl, run_dir, set_seed, write_run_info  # noqa: E402
from src.data.text_clean import format_report, parse_generated  # noqa: E402
from src.models.sft_data import select_train_rows  # noqa: E402


def prior_texts(rows: pd.DataFrame, pool: pd.DataFrame) -> list[str]:
    """Report text of each row's prior study (same patient, earlier date; Step 0 loader), '' if not in `pool`.
    `pool` is the SAME split, so a prior never crosses the patient-disjoint split boundary."""
    if "prior_study_id" not in rows:
        return [""] * len(rows)
    look = {str(k): format_report(f, i) for k, f, i in zip(pool["study_id"], pool["findings"], pool["impression"])}
    return [look.get(str(p), "") if isinstance(p, str) and p else "" for p in rows["prior_study_id"]]


def add_prior(prompt: str, prior: str | None) -> str:
    """CLAUDE.md prompt format: '... Indication: {indication}' (+ 'Prior report: {prior}' when available),
    inserted right after the Indication line."""
    if not prior or not str(prior).strip():
        return prompt
    lines = prompt.split("\n")
    k = next((i for i, l in enumerate(lines) if l.startswith("Indication:")), len(lines) - 1)
    return "\n".join(lines[:k + 1] + [f"Prior report: {' '.join(str(prior).split())}"] + lines[k + 1:])


def build_rows(cfg: dict, smoke: bool, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    from src.data.synthetic import load_chexpert, load_nih

    ad, sfx = cfg["align_data"], "_smoke" if smoke else ""
    ddir = data_dir(cfg)
    tr_real = pd.read_parquet(ddir / f"train{sfx}.parquet")
    va_real = pd.read_parquet(ddir / f"val{sfx}.parquet")
    have = set(tr_real["dataset"])
    real_cfg = {d: n for d, n in ad["real"].items() if d in have}
    tr = select_train_rows(tr_real, real_cfg, seed, "train")
    tr = tr.assign(target=[format_report(f, i) for f, i in zip(tr["findings"], tr["impression"])], source="real",
                   prior=prior_texts(tr, tr_real))
    vcfg = {d: n for d, n in ad["val"].items() if d in set(va_real["dataset"])}
    va = select_train_rows(va_real, vcfg, seed, "val") if vcfg else va_real.iloc[:0]
    va = va.assign(target=[format_report(f, i) for f, i in zip(va["findings"], va["impression"])], source="real",
                   prior=prior_texts(va, va_real))
    parts_tr, parts_va = [tr], [va]
    n_cx, n_nih = int(ad["chexpert"]["n"]), int(ad["nih"]["n"])
    if smoke:
        n_cx, n_nih = min(16, n_cx), min(16, n_nih)
    if n_cx and Path(ad["chexpert"]["root"]).exists():   # n = 0 -> no CheXpert templates (Step 5)
        parts_tr.append(load_chexpert(Path(ad["chexpert"]["root"]), "train", n_cx, seed))
        parts_va.append(load_chexpert(Path(ad["chexpert"]["root"]), "val", ad["val"].get("chexpert_valid"), seed))
    else:
        print(f"WARNING: CheXpert not found at {ad['chexpert']['root']} - no CheXpert templates")
    if n_nih and Path(ad["nih"]["root"]).exists():
        parts_tr.append(load_nih(Path(ad["nih"]["root"]), n_nih, seed))
    else:
        print(f"WARNING: NIH not found at {ad['nih']['root']} - no NIH templates")
    for p_ in parts_tr[1:] + parts_va[1:]:
        p_["prior"] = ""     # template reports have no prior study
    cols = ["dataset", "study_id", "patient_id", "split", "image_path", "indication", "target", "source", "prior"]
    tr = pd.concat([p[cols] for p in parts_tr], ignore_index=True).sample(frac=1.0, random_state=seed)
    va = pd.concat([p[cols] for p in parts_va], ignore_index=True)
    assert not set(tr["patient_id"]) & set(va["patient_id"]), "train/val patient overlap"
    test_pat = set(pd.read_parquet(ddir / f"test{sfx}.parquet", columns=["patient_id"])["patient_id"])
    assert not set(tr["patient_id"]) & test_pat, "a training patient is in the test split"
    return tr.reset_index(drop=True), va.reset_index(drop=True)


class AlignDS:
    def __init__(self, rows: list[dict], processor, tokenizer, template: str, empty: str, with_target=True,
                 use_prior=False, augment_p: float = 0.0, seed: int = 0):
        self.rows, self.proc, self.tok = rows, processor, tokenizer
        self.template, self.empty, self.with_target = template, empty, with_target
        self.use_prior, self.augment_p, self.seed = use_prior, augment_p, seed
        self.calls = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        from src.models.medgemma_zeroshot import build_prompt
        from src.models.model_b import prompt_pieces, target_ids
        from src.retrieval.embed import load_for_siglip

        r = self.rows[i]
        tf = None
        if self.augment_p > 0:   # Step 6: 40% phone-photo / WhatsApp style images (src/data/augment.py)
            import random as _r

            from src.data.augment import photo_augment

            self.calls += 1
            rng = _r.Random(hash((self.seed, i, self.calls)))
            if rng.random() < self.augment_p:
                tf = lambda im: photo_augment(im, rng)  # noqa: E731
        px = self.proc(images=[load_for_siglip(r["image_path"], transform=tf)], return_tensors="pt")["pixel_values"][0]
        prompt = build_prompt(self.template, r.get("indication"), self.empty)
        if self.use_prior:
            prompt = add_prior(prompt, r.get("prior"))
        pre, post = prompt_pieces(self.tok, prompt)
        return {"px": px, "pre": pre, "post": post, "tgt": target_ids(self.tok, r["target"]) if self.with_target else None,
                "i": i}


def collate(b):
    import torch

    return {"px": torch.stack([x["px"] for x in b]), "pre": [x["pre"] for x in b], "post": [x["post"] for x in b],
            "tgt": [x["tgt"] for x in b], "i": [x["i"] for x in b]}


def converged(losses: list[float], first_ratio: float, plateau_rel: float) -> dict:
    n = len(losses)
    if n < 20:
        return {"converged": False, "reason": f"only {n} steps"}
    a = float(np.mean(losses[: max(1, n // 20)]))
    last = float(np.mean(losses[-max(1, n // 10):]))
    prev = float(np.mean(losses[-2 * max(1, n // 10): -max(1, n // 10)]))
    rel = abs(prev - last) / max(prev, 1e-9)
    ok = last <= first_ratio * a and rel < plateau_rel
    return {"converged": bool(ok), "first_5pct": round(a, 4), "prev_10pct": round(prev, 4), "last_10pct": round(last, 4),
            "plateau_rel_change": round(rel, 4)}


def coherence(texts: list[str]) -> dict:
    from src.verify.claims import split_sentences

    fmt = [parse_generated(t)["parse_ok"] for t in texts]
    rep = []
    for t in texts:
        ss = [" ".join(s.lower().split()) for s in split_sentences(t.replace("FINDINGS:", "").replace("IMPRESSION:", ""))]
        rep.append(len(ss) != len(set(ss)))
    return {"n": len(texts), "format_ok": round(float(np.mean(fmt)), 4), "repeated_sentence": round(float(np.mean(rep)), 4),
            "mean_words": round(float(np.mean([len(t.split()) for t in texts])), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step4.yaml")
    ap.add_argument("--run-name", default="b1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-hours", type=float)
    args = ap.parse_args()

    import torch
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    is_main = rank == 0
    log = (lambda m: print(m, flush=True)) if is_main else (lambda m: None)

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    mb, tc, ac, g = cfg["model_b"], cfg["train"], cfg["acceptance"], cfg["generation"]
    max_hours = args.max_hours if args.max_hours is not None else float(tc["max_hours"])
    stage = cfg.get("run_step", "step4")
    rdir = run_dir(cfg, stage, args.run_name + ("_smoke" if args.smoke else ""))
    use_prior = bool(tc.get("use_prior", False))
    augment_p = float(tc.get("augment_p", 0.0))

    tr, va = build_rows(cfg, args.smoke, seed)
    if args.smoke:
        tr, va = tr.head(int(tc["smoke"]["n_train"])), va.head(int(tc["smoke"]["n_val"]))
    log(f"train: {tr.groupby(['source', 'dataset']).size().to_dict()} | val: {va['dataset'].value_counts().to_dict()}")

    from src.models.medgemma_zeroshot import hf_token
    from src.models.model_b import ModelB, Projector, load_components

    vision, proc, llm, tok = load_components(mb["encoder"], mb["llm"], mb["encoder_dtype"], mb["llm_dtype"], device,
                                             hf_token())
    for p in llm.parameters():
        p.requires_grad = False
    if tc["gradient_checkpointing"]:
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    llm.train()   # HF only checkpoints in train mode; Gemma 3 has no dropout (attention_dropout = 0)
    d_vis, d_llm = vision.config.hidden_size, llm.config.hidden_size
    proj = Projector(d_vis, d_llm, int(mb["merge"]), mb["projector_hidden"]).to(device)
    init_from = tc.get("init_projector")   # Step 5: start from the Step 4 projector (run name under runs/step4)
    if init_from:
        ip = run_dir(cfg, "step4", init_from + ("_smoke" if args.smoke else "")) / "projector.pt"
        if not ip.exists():
            raise SystemExit(f"no Step 4 projector at {ip} (finish Step 4 first)")
        proj.load_state_dict(torch.load(ip, map_location="cpu"))
        log(f"projector initialised from {ip}")
    lora = tc.get("lora")
    if lora:   # Step 5: LoRA on all linear layers of the LLM (peft 0.21 LoraConfig; Gemma 3 module names)
        from peft import LoraConfig, get_peft_model

        llm = get_peft_model(llm, LoraConfig(r=int(lora["r"]), lora_alpha=int(lora["alpha"]),
                                             lora_dropout=float(lora["dropout"]), bias="none",
                                             target_modules=list(lora["target_modules"])))
        if is_main:
            llm.print_trainable_parameters()
    model = ModelB(vision, llm, proj, torch.float16 if mb["encoder_dtype"] == "fp16" else torch.float32)
    llm_trainable = [(n, p_) for n, p_ in llm.named_parameters() if p_.requires_grad]
    if not lora and llm_trainable:
        raise AssertionError("phase 1 must not train the LLM")

    # drop over-long targets (never truncate: a cut report loses its stop token)
    from src.models.model_b import target_ids

    keep = [len(target_ids(tok, t)) <= int(tc["max_target_tokens"]) for t in tr["target"]]
    log(f"dropped {len(keep) - sum(keep)} training reports longer than {tc['max_target_tokens']} tokens")
    tr = tr[keep].reset_index(drop=True)
    meta = {"encoder": mb["encoder"], "llm": mb["llm"], "dtypes": [mb["encoder_dtype"], mb["llm_dtype"]],
            "merge": int(mb["merge"]), "d_vis": d_vis, "d_llm": d_llm, "projector_hidden": mb["projector_hidden"],
            "n_train": len(tr), "lr": float(tc["lr"]), "seed": seed, "world_size": world, "lora": lora,
            "lr_llm": tc.get("lr_llm"), "init_projector": init_from, "use_prior": use_prior, "augment_p": augment_p}
    if is_main:
        mp = rdir / "projector.json"
        if (rdir / "checkpoint.pt").exists() and mp.exists():
            old = json.loads(mp.read_text(encoding="utf-8"))
            diff = {k: (old.get(k), v) for k, v in meta.items() if old.get(k) != v}
            if diff:
                raise SystemExit(f"settings differ from the existing run: {diff}. Use a new --run-name.")
        mp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        tr[["dataset", "study_id", "source"]].to_csv(rdir / "train_rows.csv", index=False)
        write_run_info(rdir, cfg, args, {"step": f"{stage}_train", **meta})

    from torch.utils.data import DataLoader

    tmpl, empty = g["prompt_template"], g["empty_indication"]
    ds = AlignDS(tr.to_dict("records"), proc, tok, tmpl, empty, use_prior=use_prior, augment_p=augment_p, seed=seed)
    bs, acc = int(tc["batch_size"]), int(tc["grad_accum"])
    steps_per_epoch = len(ds) // (bs * acc * world)
    total = int(tc["smoke"]["max_steps"]) if args.smoke else steps_per_epoch * int(tc["epochs"])
    if total < 1:
        raise SystemExit(f"not enough data for one optimizer step ({len(ds)} examples)")

    # ---- sanity check on one real batch (CLAUDE.md §8)
    b0 = collate([ds[0], ds[1]])
    from src.models.model_b import assemble

    with torch.no_grad():
        img = model.image_tokens(b0["px"].to(device)).to(llm.dtype)
        bt = assemble(llm.get_input_embeddings(), img, b0["pre"], b0["post"], b0["tgt"], pad_left=False)
        sup = tok.decode([lab for lab in bt["labels"][0].tolist() if lab != -100])
        loss0 = float(llm(**bt).loss)
    probs = []
    if not sup.lstrip().startswith("FINDINGS:"):
        probs.append(f"supervised text does not start with FINDINGS: {sup[:80]!r}")
    if not sup.rstrip().endswith("<end_of_turn>"):
        probs.append("<end_of_turn> is not supervised (the model would not learn to stop)")
    if not math.isfinite(loss0):
        probs.append(f"initial loss is {loss0}")
    if probs:
        raise SystemExit("SANITY CHECK FAILED: " + "; ".join(probs))
    log(f"sanity OK: {img.shape[1]} image tokens of dim {img.shape[2]}, prompt {len(b0['pre'][0])}+{len(b0['post'][0])} "
        f"tokens, initial loss {loss0:.3f}\n  supervised: {sup[:160]!r}")

    # Data parallel by hand: every rank computes its own micro-batches, gradients of ALL trainable tensors
    # (projector + LoRA) are averaged with all_reduce before each optimizer step. No DDP wrapper is needed.
    pmodel = proj
    trainable = list(proj.parameters()) + [p_ for _, p_ in llm_trainable]
    groups = [{"params": list(proj.parameters()), "lr": float(tc["lr"])}]
    if llm_trainable:
        groups.append({"params": [p_ for _, p_ in llm_trainable], "lr": float(tc["lr_llm"])})
    opt = torch.optim.AdamW(groups, weight_decay=float(tc["weight_decay"]))
    warm = max(1, int(float(tc["warmup_frac"]) * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total)))))
    step = 0
    ck = rdir / "checkpoint.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        proj.load_state_dict(st["projector"])
        if llm_trainable:
            cur = dict(llm.named_parameters())
            with torch.no_grad():
                for n, v in st["lora"].items():
                    cur[n].copy_(v)
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        step = int(st["step"])
        log(f"resumed at step {step}/{total}")

    def save():
        if is_main:
            torch.save({"projector": proj.state_dict(), "optimizer": opt.state_dict(),
                        "lora": {n: p_.detach().cpu() for n, p_ in llm_trainable},
                        "scheduler": sched.state_dict(), "step": step}, str(ck) + ".tmp")
            os.replace(str(ck) + ".tmp", ck)

    def unwrapped(fn):
        """Validation / generation run on rank 0 only: use the plain projector, never the DDP wrapper."""
        model.projector = proj
        try:
            return fn()
        finally:
            model.projector = pmodel

    def _val_loss() -> float:
        dl = DataLoader(AlignDS(va.to_dict("records"), proc, tok, tmpl, empty, use_prior=use_prior), batch_size=bs,
                        collate_fn=collate,
                        num_workers=int(tc["num_workers"]))
        tot, n = 0.0, 0
        with torch.no_grad():
            for b in dl:
                out = model(b["px"].to(device), b["pre"], b["post"], b["tgt"])
                k = sum(len(t) for t in b["tgt"])
                tot, n = tot + float(out.loss) * k, n + k
        return tot / max(1, n)

    def val_loss() -> float:
        return unwrapped(_val_loss)

    # One deterministic example stream: global position p -> permutation of epoch p // N. Step s uses positions
    # [s*G, (s+1)*G), G = batch * accum * world; rank r takes its contiguous share. Resuming = starting at `step`.
    N, G, per_rank = len(ds), bs * acc * world, bs * acc
    perms: dict[int, np.ndarray] = {}

    def example(pos: int) -> int:
        e = pos // N
        if e not in perms:
            perms[e] = np.random.default_rng(seed + e).permutation(N)
        return int(perms[e][pos % N])

    my_idx = [example(s_ * G + rank * per_rank + j) for s_ in range(step, total) for j in range(per_rank)]
    deadline = time.time() + max_hours * 3600
    t0, done_this_session = time.time(), 0
    loss_p = rdir / "losses.jsonl"
    dl = DataLoader(torch.utils.data.Subset(ds, my_idx), batch_size=bs, collate_fn=collate,
                    num_workers=int(tc["num_workers"]))
    it = iter(dl)
    while step < total:
        opt.zero_grad(set_to_none=True)
        accum_loss = []
        for _ in range(acc):
            b = next(it)
            out = model(b["px"].to(device), b["pre"], b["post"], b["tgt"])
            (out.loss / acc).backward()
            accum_loss.append(float(out.loss.detach()))
        if world > 1:
            for p_ in trainable:
                if p_.grad is not None:
                    dist.all_reduce(p_.grad)
                    p_.grad /= world
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        sched.step()
        step += 1
        done_this_session += 1
        lv = float(np.mean(accum_loss))
        if not math.isfinite(lv):
            raise SystemExit(f"non-finite loss at step {step}")
        if is_main:
            append_jsonl(loss_p, [{"step": step, "loss": round(lv, 5), "lr": sched.get_last_lr()[0]}])
        if is_main and (step % int(tc["log_every"]) == 0 or step == total):
            rate = (time.time() - t0) / done_this_session
            mem = f"  mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if torch.cuda.is_available() else ""
            log(f"step {step}/{total}  loss {lv:.4f}  lr {sched.get_last_lr()[0]:.2e}  {rate:.1f}s/step  "
                f"ETA {rate * (total - step) / 3600:.1f} h{mem}")
        if step % int(tc["save_every"]) == 0:
            save()
        if is_main and step % int(tc["eval_every"]) == 0:
            log(f"  val loss {val_loss():.4f}")
        if time.time() > deadline and step < total:
            save()
            log(f"max_hours reached at step {step}/{total}: checkpoint saved; rerun the same command to resume.")
            return
    save()
    if world > 1:
        dist.barrier()
    if not is_main:
        return
    torch.save(proj.state_dict(), rdir / "projector.pt")
    if llm_trainable:
        llm.save_pretrained(str(rdir / "lora"))   # peft adapter (adapter_config.json + adapter_model.*)
    vl = val_loss()
    losses = [r["loss"] for r in read_jsonl(loss_p)]
    conv = converged(losses, float(ac["first_5pct_ratio"]), float(ac["plateau_rel"]))
    # coherence: greedy reports on val images
    vrows = va.head(int(ac["n_samples"])).to_dict("records")
    vdl = DataLoader(AlignDS(vrows, proc, tok, tmpl, empty, with_target=False, use_prior=use_prior), batch_size=bs,
                     collate_fn=collate,
                     num_workers=int(tc["num_workers"]))
    model.eval()

    def _gen():
        out = []
        for b in vdl:
            out += model.generate(b["px"].to(device), b["pre"], b["post"], tok, int(ac["max_new_tokens"]))
        return out

    texts = unwrapped(_gen)
    with open(rdir / "val_samples.jsonl", "w", encoding="utf-8") as f:
        for r, t in zip(vrows, texts):
            f.write(json.dumps({"study_id": r["study_id"], "dataset": r["dataset"], "reference": r["target"],
                                "generated": t}) + "\n")
    coh = coherence(texts)
    coh_ok = coh["format_ok"] >= float(ac["format_ok"]) and coh["repeated_sentence"] <= float(ac["max_repeat"])
    res = {"val_loss": round(vl, 4), "convergence": conv, "coherence": coh,
           "accepted": bool(conv["converged"] and coh_ok), "steps": step, "note": "also READ val_samples.jsonl"}
    (rdir / "acceptance.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(json.dumps(res, indent=2))
    for r, t in list(zip(vrows, texts))[:5]:
        log(f"--- {r['dataset']} {r['study_id']}\nREF: {r['target'][:300]}\nGEN: {t[:300]}")


if __name__ == "__main__":
    main()
