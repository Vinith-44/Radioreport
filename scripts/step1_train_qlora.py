"""Step 1a: Model A = QLoRA fine-tune of MedGemma 1.5 4B-it with Unsloth. Resumable.

  python scripts/step1_train_qlora.py --run-name a1 --smoke      # ~50 studies, 6 optimizer steps
  python scripts/step1_train_qlora.py --run-name a1              # full run; rerun the same command to resume

Writes outputs/runs/step1/<run>[_smoke]/
  checkpoints/checkpoint-*   (resume points, every save_steps)
  adapter/                   (final LoRA adapter + processor, written when training finishes)
  train_meta.json            (locked settings + the exact train/val study ids)
  log_history.json, run_info_*.json

APIs verified against the source of unsloth 2026.11.4 / unsloth_zoo 2026.11.4 / trl 1.15.0 / transformers 5.17.0:
  - FastVisionModel.from_pretrained(load_in_4bit=True) detects Gemma 3 on an fp16-only GPU (T4) and switches to
    Unsloth's forced-float32 path (UNSLOTH_FORCE_FLOAT32: 4-bit matmuls in fp16, everything else fp32); its
    trainer patch then sets fp16=bf16=False itself. So training is NOT plain fp16 - plain fp16 NaNs Gemma 3.
  - UnslothVisionDataCollator(train_on_responses_only=True, instruction_part, response_part) masks the prompt.
  - transformers 5.x removed TrainingArguments.warmup_ratio: warmup_steps < 1 is a fraction of total steps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INSTRUCTION_PART = "<start_of_turn>user\n"   # Gemma 3 chat template turn markers
RESPONSE_PART = "<start_of_turn>model\n"
LOCKED = ("base_model_id", "lora", "finetune", "lr", "effective_batch", "epochs", "max_seq_length", "seed",
          "train_ids_sha256", "prompt_template", "augment_p")


def _ids_sha(ids) -> str:
    return hashlib.sha256("\n".join(map(str, ids)).encode()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step1.yaml")
    ap.add_argument("--run-name", default="a1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-hours", type=float, help="override train.max_hours")
    ap.add_argument("--max-steps", type=int, help="override the number of optimizer steps (debugging)")
    ap.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for training (Unsloth OSS is single-GPU)")
    ap.add_argument("--allow-missing-datasets", action="store_true",
                    help="train even if a configured dataset has no train rows (e.g. train images not fetched)")
    args = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)

    # Unsloth must be imported before transformers / trl / peft so its patches apply.
    from unsloth import FastVisionModel  # noqa: I001
    from unsloth.trainer import UnslothVisionDataCollator
    import pandas as pd
    import torch
    from transformers import TrainerCallback
    from trl import SFTConfig, SFTTrainer

    from src.common import data_dir, load_config, run_dir, set_seed, write_run_info
    from src.models.medgemma_zeroshot import hf_token
    from src.models.sft_data import SFTImageDataset, build_messages, select_train_rows

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    t, g = cfg["train"], cfg["generation"]
    suffix = "_smoke" if args.smoke else ""
    model_id = cfg["models"][cfg["model_a"]["base"]]
    max_hours = args.max_hours if args.max_hours is not None else float(t["max_hours"])

    # ---------------- data (TRAIN split only; VAL subset for eval loss; TEST never touched) ----------------
    ddir = data_dir(cfg)
    train_df = pd.read_parquet(ddir / f"train{suffix}.parquet")
    val_df = pd.read_parquet(ddir / f"val{suffix}.parquet")
    test_pat = set(pd.read_parquet(ddir / f"test{suffix}.parquet", columns=["patient_id"])["patient_id"])
    per_ds = dict(t["smoke"]["datasets"] if args.smoke else t["datasets"])
    have = set(train_df["dataset"])
    absent = [d for d in per_ds if d not in have]
    if absent:
        msg = (f"no TRAIN rows for {absent} in {ddir}/train{suffix}.parquet. For ReXGradient: fetch the train "
               f"subset (step0_fetch_rexgradient.py --splits train val ...), attach it, and rerun step0_prepare_data.py.")
        if not (args.allow_missing_datasets or args.smoke):
            raise SystemExit("ERROR: " + msg)
        print("WARNING: " + msg, flush=True)
        per_ds = {d: n for d, n in per_ds.items() if d in have}
    tr = select_train_rows(train_df, per_ds, seed, "train")
    n_val = int(t["smoke"]["val_studies"] if args.smoke else t["val_studies"])
    va = select_train_rows(val_df, {d: max(1, n_val // len(per_ds)) for d in per_ds if d in set(val_df["dataset"])},
                           seed, "val")
    leak = set(tr["patient_id"]) & test_pat
    assert not leak, f"PATIENT LEAKAGE: {len(leak)} training patients are in the test split, e.g. {sorted(leak)[:3]}"
    assert not (set(tr["patient_id"]) & set(va["patient_id"])), "train/val patient overlap"
    print(f"train studies: {tr['dataset'].value_counts().to_dict()} | val (eval loss): {va['dataset'].value_counts().to_dict()}",
          flush=True)

    # ---------------- run dir + locked settings (resume safety) ----------------
    rdir = run_dir(cfg, cfg.get("run_step", "step1"), args.run_name + suffix)   # Step 6 reuses this trainer
    ckdir = rdir / "checkpoints"
    eff_batch = int(t["per_device_batch"]) * int(t["grad_accum"])
    fin = {k: bool(t[k]) for k in ("finetune_vision_layers", "finetune_language_layers",
                                   "finetune_attention_modules", "finetune_mlp_modules")}

    # ---------------- model ----------------
    t0 = time.time()
    model, processor = FastVisionModel.from_pretrained(
        model_id, load_in_4bit=True, max_seq_length=int(t["max_seq_length"]), token=hf_token(),
        use_gradient_checkpointing="unsloth")
    model = FastVisionModel.get_peft_model(
        model, r=int(t["lora"]["r"]), lora_alpha=int(t["lora"]["alpha"]), lora_dropout=float(t["lora"]["dropout"]),
        bias="none", random_state=seed, use_rslora=False, **fin)
    FastVisionModel.for_training(model)
    print(f"loaded {model_id} (4-bit) + LoRA in {time.time() - t0:.0f}s; "
          f"UNSLOTH_FORCE_FLOAT32={os.environ.get('UNSLOTH_FORCE_FLOAT32')}", flush=True)
    model.print_trainable_parameters()
    tok = processor.tokenizer

    # Drop (never truncate) examples that would not fit: a truncated report would also lose its stop token.
    n_img = int(getattr(processor, "image_seq_length", 256)) + 8   # soft tokens + begin/end-of-image + newlines

    def n_tokens(r: dict) -> int:
        text = processor.apply_chat_template(build_messages(r, g["prompt_template"], g["empty_indication"]),
                                             tokenize=False)
        return len(tok(text, add_special_tokens=False)["input_ids"]) + n_img

    tr_rows, va_rows = tr.to_dict("records"), va.to_dict("records")
    lens = [n_tokens(r) for r in tr_rows]
    keep = [r for r, n in zip(tr_rows, lens) if n <= int(t["max_seq_length"])]
    print(f"token length: median {sorted(lens)[len(lens) // 2]}, max {max(lens)}; "
          f"dropped {len(tr_rows) - len(keep)} studies longer than {t['max_seq_length']}", flush=True)
    tr_rows = keep
    va_rows = [r for r in va_rows if n_tokens(r) <= int(t["max_seq_length"])]

    settings = {"base_model_id": model_id, "lora": t["lora"], "finetune": fin, "lr": float(t["lr"]),
                "effective_batch": eff_batch, "epochs": float(t["epochs"]), "max_seq_length": int(t["max_seq_length"]),
                "seed": seed, "train_ids_sha256": _ids_sha([r["study_id"] for r in tr_rows]),
                "prompt_template": g["prompt_template"], "augment_p": float(t.get("augment_p", 0.0))}
    meta_p = rdir / "train_meta.json"
    if meta_p.exists():
        old = json.loads(meta_p.read_text(encoding="utf-8"))
        diff = {k: (old.get(k), settings[k]) for k in LOCKED if old.get(k) != settings[k]}
        if diff:
            raise SystemExit(f"settings differ from the existing run {rdir.name}: {diff}. Use a new --run-name.")
    else:
        meta_p.write_text(json.dumps({**settings, "train_study_ids": [r["study_id"] for r in tr_rows],
                                      "val_study_ids": [r["study_id"] for r in va_rows],
                                      "train_counts": pd.Series([r["dataset"] for r in tr_rows]).value_counts().to_dict()},
                                     indent=1), encoding="utf-8")
    write_run_info(rdir, cfg, args, {"step": "step1_train", "settings": settings, "n_train": len(tr_rows),
                                     "n_val": len(va_rows), "max_hours": max_hours})

    train_ds = SFTImageDataset(tr_rows, g["prompt_template"], g["empty_indication"],
                               augment_p=float(t.get("augment_p", 0.0)), seed=seed)
    val_ds = SFTImageDataset(va_rows, g["prompt_template"], g["empty_indication"])
    collator = UnslothVisionDataCollator(model, processor, max_seq_length=int(t["max_seq_length"]),
                                         train_on_responses_only=True, instruction_part=INSTRUCTION_PART,
                                         response_part=RESPONSE_PART)

    # ---------------- sanity check of the exact tensors the model will see (CLAUDE.md §8) ----------------
    b = collator([train_ds[0], train_ds[1]])
    ids, lab, att = b["input_ids"][0], b["labels"][0], b["attention_mask"][0].bool()
    soft_id = getattr(tok, "image_token_id", None)
    sup = tok.decode(ids[lab != -100])
    shown = ids[att & (ids != soft_id)] if soft_id is not None else ids[att]
    print("---- sample 0, model input (image soft tokens hidden):\n" + tok.decode(shown)[-900:])
    print("---- sample 0, supervised tokens (loss is computed only on these):\n" + sup[:600])
    n_bos = int((ids[att] == tok.bos_token_id).sum())
    n_soft = int((ids == soft_id).sum()) if soft_id is not None else -1
    problems = []
    if not sup.lstrip().startswith("FINDINGS:"):
        problems.append("supervised text does not start with 'FINDINGS:' (prompt masking is off)")
    if "Indication:" in sup or "You are an expert radiologist" in sup:
        problems.append("prompt text is inside the loss")
    if "<end_of_turn>" not in sup:
        problems.append("<end_of_turn> is not supervised (the model would not learn to stop)")
    if n_bos != 1:
        problems.append(f"{n_bos} <bos> tokens (expected 1)")
    if soft_id is not None and n_soft != int(getattr(processor, "image_seq_length", 256)):
        problems.append(f"{n_soft} image soft tokens (expected {getattr(processor, 'image_seq_length', 256)})")
    if "pixel_values" not in b:
        problems.append("no pixel_values in the batch")
    if problems:
        raise SystemExit("SANITY CHECK FAILED:\n  - " + "\n  - ".join(problems))
    print(f"sanity check OK: 1 <bos>, {n_soft} image tokens, prompt masked, report + <end_of_turn> supervised, "
          f"pixel_values {tuple(b['pixel_values'].shape)} {b['pixel_values'].dtype}", flush=True)

    # ---------------- trainer ----------------
    class StopAfterHours(TrainerCallback):
        """Save and stop cleanly before Kaggle's 12 h session limit; rerun the same command to resume."""

        def __init__(self, hours: float):
            self.deadline = time.time() + hours * 3600

        def on_step_end(self, a, state, control, **kw):
            if time.time() > self.deadline:
                print(f"\nmax_hours reached at step {state.global_step}: saving a checkpoint and stopping.", flush=True)
                control.should_save = True
                control.should_training_stop = True
            return control

    smoke_steps = int(t["smoke"]["max_steps"]) if args.smoke else None
    sargs = SFTConfig(
        output_dir=str(ckdir), seed=seed, data_seed=seed,
        per_device_train_batch_size=int(t["per_device_batch"]), per_device_eval_batch_size=int(t["per_device_batch"]),
        gradient_accumulation_steps=int(t["grad_accum"]), num_train_epochs=float(t["epochs"]),
        max_steps=args.max_steps or smoke_steps or -1,
        learning_rate=float(t["lr"]), warmup_steps=float(t["warmup_ratio"]), weight_decay=float(t["weight_decay"]),
        lr_scheduler_type=t["lr_scheduler"], optim=t["optim"], max_grad_norm=float(t["max_grad_norm"]),
        logging_steps=1 if args.smoke else int(t["logging_steps"]),
        eval_strategy="steps", eval_steps=(smoke_steps or int(t["eval_steps"])),
        save_strategy="steps", save_steps=(smoke_steps or int(t["save_steps"])), save_total_limit=int(t["save_total_limit"]),
        fp16=False, bf16=False,  # T4: no bf16; Unsloth's forced-float32 path for Gemma 3 owns the precision
        report_to="none", dataloader_num_workers=int(t["num_workers"]),
        remove_unused_columns=False, dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
        max_length=int(t["max_seq_length"]),
    )
    trainer = SFTTrainer(model=model, processing_class=processor.tokenizer, data_collator=collator,
                         train_dataset=train_ds, eval_dataset=val_ds, args=sargs,
                         callbacks=[StopAfterHours(max_hours)])
    last = None
    if ckdir.exists():
        cks = sorted(ckdir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
        last = str(cks[-1]) if cks else None
    print(f"{'resuming from ' + last if last else 'starting fresh'}; {len(train_ds)} train examples, "
          f"effective batch {eff_batch}, max_hours {max_hours}", flush=True)
    t1 = time.time()
    out = trainer.train(resume_from_checkpoint=last)
    hrs = (time.time() - t1) / 3600
    (rdir / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=1), encoding="utf-8")
    mem = [round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())]
    total = trainer.state.max_steps
    done = trainer.state.global_step >= total
    print(f"\ntrained {trainer.state.global_step}/{total} steps this far ({hrs:.2f} h this session), "
          f"train_loss {out.training_loss:.4f}, peak GPU mem {mem} GB", flush=True)
    if not done:
        print("NOT FINISHED (time limit). Rerun the same command (attach this run's output) to resume.", flush=True)
        return
    adir = rdir / "adapter"
    model.save_pretrained(str(adir))
    processor.save_pretrained(str(adir))
    ev = trainer.evaluate()
    (rdir / "final_eval.json").write_text(json.dumps(ev, indent=2), encoding="utf-8")
    print(f"DONE. adapter -> {adir}\nfinal val loss {ev.get('eval_loss')}", flush=True)


if __name__ == "__main__":
    main()
