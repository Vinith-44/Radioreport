"""Step 9a: GRPO on the LoRA of the Step 5 winner with multiple rewards (src/rl). Resumable.

  python scripts/step9_grpo.py --run-name g1 --smoke
  python scripts/step9_grpo.py --run-name g1

Prompts = a seeded set of TRAIN studies with an image and a Step 2 train embedding (probe evidence for the verify
reward). Each step: prompts_per_step prompts x num_generations samples -> rewards -> group advantages -> GRPO loss
(src/rl/grpo.py) -> AdamW on the LoRA. Logs every reward component, KL, length and truncation rate.
Writes outputs/runs/step9/<run>/: checkpoint (LoRA + optimizer + step), log.jsonl, and the final model in the
same layout as its source (adapter/ for Model A; projector.* + lora/ for Model B), loadable by src/models/systems.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import append_jsonl, data_dir, load_config, read_jsonl, run_dir, set_seed, write_run_info  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/step9.yaml")
    ap.add_argument("--run-name", default="g1")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    import torch

    from src.models.medgemma_zeroshot import hf_token
    from src.models.probe import load_probe, predict
    from src.models.systems import spec_dir
    from src.retrieval.store import load_split
    from src.rl.grpo import group_advantages, grpo_loss
    from src.rl.policy import PolicyA, PolicyB
    from src.rl.rewards import RewardModel

    cfg = load_config(args.config)
    seed = int(cfg["seed"])
    set_seed(seed)
    gc_, g = cfg["grpo"], cfg["generation"]
    sfx = "_smoke" if args.smoke else ""
    rdir = run_dir(cfg, "step9", args.run_name + sfx)
    spec = gc_.get("source")
    if not spec:
        w = run_dir(cfg, "step5", gc_["winner_run"] + sfx) / "winner.json"
        if not w.exists():
            raise SystemExit(f"no {w}: run step5_score.py first (or set grpo.source)")
        spec = json.loads(w.read_text(encoding="utf-8"))["winner_spec"]
    src_run = spec_dir(cfg, spec, args.smoke)
    write_run_info(rdir, cfg, args, {"step": "step9_grpo", "source": spec, "grpo": gc_})
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # prompts: TRAIN studies that have a Step 2 embedding (-> probe probabilities)
    s2 = run_dir(cfg, "step2", gc_["step2_run"] + sfx)
    tr_ids, tr_emb = load_split(s2, "train")
    probe, pinfo = load_probe(s2 / "probe")
    p_tr, _ = predict(probe, tr_emb, pinfo["thresholds"])
    train = pd.read_parquet(data_dir(cfg) / f"train{sfx}.parquet")
    train = train.assign(study_id=train["study_id"].astype(str)).set_index("study_id")
    pidx = {str(s): i for i, s in enumerate(tr_ids["study_id"])}
    pool = sorted(s for s in pidx if s in train.index)
    n = int(gc_["smoke"]["n_prompts"]) if args.smoke else int(gc_["n_prompts"])
    chosen = sorted(random.Random(seed).sample(pool, min(n, len(pool))))
    test_pat = set(pd.read_parquet(data_dir(cfg) / f"test{sfx}.parquet", columns=["patient_id"])["patient_id"])
    assert not {train.loc[s, "patient_id"] for s in chosen} & test_pat, "a GRPO prompt patient is in the test split"
    rows = [{"study_id": s, "image_path": train.loc[s, "image_path"], "indication": train.loc[s, "indication"],
             "findings": train.loc[s, "findings"], "impression": train.loc[s, "impression"], "prior": ""}
            for s in chosen]
    print(f"{len(rows)} GRPO prompts from TRAIN (source model {spec})", flush=True)

    policy = PolicyA(cfg, src_run / "adapter", device, hf_token()) if spec["type"] == "a" else \
        PolicyB(src_run, device, hf_token())
    reward = RewardModel(gc_["rewards"], np.asarray(pinfo["thresholds"]), device)
    params = policy.trainable()
    opt = torch.optim.AdamW(params, lr=float(gc_["lr"]))
    P, G = int(gc_["prompts_per_step"]), int(gc_["num_generations"])
    total = int(gc_["smoke"]["steps"]) if args.smoke else len(rows) // P
    step, ck = 0, rdir / "checkpoint.pt"
    if ck.exists():
        st = torch.load(ck, map_location="cpu", weights_only=False)
        with torch.no_grad():
            for p, v in zip(params, st["params"]):
                p.copy_(v.to(p.device))
        opt.load_state_dict(st["optimizer"])
        step = int(st["step"])
        print(f"resumed at step {step}/{total}", flush=True)
    order = list(range(len(rows)))
    random.Random(seed + 1).shuffle(order)
    deadline, t0 = time.time() + float(gc_["max_hours"]) * 3600, time.time()
    log_p = rdir / "log.jsonl"

    def save():
        torch.save({"params": [p.detach().cpu() for p in params], "optimizer": opt.state_dict(), "step": step},
                   str(ck) + ".tmp")
        Path(str(ck) + ".tmp").replace(ck)

    while step < total:
        idx = [order[(step * P + j) % len(order)] for j in range(P)]
        batch = [rows[i] for i in idx]
        comps, trunc, texts, b = policy.sample(batch, g, G, int(gc_["max_completion_length"]), float(gc_["temperature"]))
        refs = [r["findings"] for r in batch for _ in range(G)]
        probs = np.stack([p_tr[pidx[r["study_id"]]] for r in batch for _ in range(G)])
        rew, parts = reward(texts, refs, probs)
        adv = group_advantages(rew, G)
        lp = policy.logprobs(b, comps, G)
        ref = policy.ref_logprobs(b, comps, G)
        loss, kl = grpo_loss(lp, ref, adv, float(gc_["beta"]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, float(gc_["max_grad_norm"]))
        opt.step()
        step += 1
        rec = {"step": step, "loss": round(float(loss.detach()), 5), "kl": round(kl, 5),
               "reward": round(float(np.mean(rew)), 4), "reward_std": round(float(np.std(rew)), 4),
               **{f"r_{k}": round(float(np.mean([x if x is not None else 0 for x in v])), 4) for k, v in parts.items()},
               "len_words": round(float(np.mean([len(t.split()) for t in texts])), 1),
               "truncated": round(float(np.mean(trunc)), 3)}
        if not math.isfinite(rec["loss"]):
            raise SystemExit(f"non-finite loss at step {step}: {rec}")
        append_jsonl(log_p, [rec])
        rate = (time.time() - t0) / max(1, step)
        print(f"step {step}/{total} reward {rec['reward']:+.3f} (clin {rec['r_clinical']:.3f}, ver {rec['r_verify']:+.3f}, "
              f"fmt {rec['r_format']:.2f}) kl {kl:.4f} len {rec['len_words']} trunc {rec['truncated']}  "
              f"{rate:.0f}s/step ETA {rate * (total - step) / 3600:.1f} h", flush=True)
        if step == 1:
            print("--- sample completion:\n" + texts[0][:500], flush=True)
        if step % int(gc_["save_every"]) == 0:
            save()
        if time.time() > deadline and step < total:
            save()
            print("max_hours reached: checkpoint saved, rerun the same command to resume.", flush=True)
            return
    save()
    policy.save(rdir, src_run)
    (rdir / "grpo_meta.json").write_text(json.dumps({"source": spec, "steps": step, "type": spec["type"]}, indent=2),
                                         encoding="utf-8")
    logs = read_jsonl(log_p)
    k = max(1, len(logs) // 10)
    print(f"DONE {step} steps. mean reward first 10% {np.mean([x['reward'] for x in logs[:k]]):.3f} -> "
          f"last 10% {np.mean([x['reward'] for x in logs[-k:]]):.3f}; model in {rdir}", flush=True)


if __name__ == "__main__":
    main()
