"""Step 9: a minimal on-policy GRPO update that works for both final model types (Model A = MedGemma + LoRA,
Model B = custom encoder/projector/Gemma + LoRA). TRL's GRPOTrainer cannot train the custom Model B and its VLM
path targets Qwen-style image tokens, so the objective is implemented here, matching TRL's GRPO with
num_iterations=1 (one update per batch of samples -> importance ratio 1, no clipping needed):

  A_ij  = (r_ij - mean_j r_i.) / (std_j r_i. + eps)                       group-normalised advantage
  loss  = mean_ij [ mean_t( -A_ij * log pi(y_ijt) + beta * KL_t ) ]        per-sequence token mean ("grpo")
  KL_t  = exp(ref_t - logp_t) - (ref_t - logp_t) - 1                      k3 estimator (DeepSeekMath / TRL)
The reference policy is the SFT model = the same network with the LoRA adapter disabled (peft disable_adapter()),
so no second copy of the model is needed.
"""
from __future__ import annotations

import numpy as np
import torch


def group_advantages(rewards: list[float], group: int, eps: float = 1e-4) -> list[float]:
    r = np.asarray(rewards, dtype=np.float64).reshape(-1, group)
    adv = (r - r.mean(1, keepdims=True)) / (r.std(1, keepdims=True) + eps)
    adv[r.std(1) < 1e-8] = 0.0      # all samples equally good -> no learning signal from this group
    return adv.reshape(-1).tolist()


def grpo_loss(logp: list[torch.Tensor], ref_logp: list[torch.Tensor], adv: list[float], beta: float) -> tuple:
    """logp / ref_logp: per completion, the log-probs of its tokens (1-D). Returns (loss, mean KL)."""
    losses, kls = [], []
    for lp, rp, a in zip(logp, ref_logp, adv):
        if lp.numel() == 0:
            continue
        d = rp.detach() - lp
        kl = torch.exp(d) - d - 1
        losses.append((-a * lp + beta * kl).mean())
        kls.append(kl.detach().mean())
    if not losses:
        z = torch.zeros((), requires_grad=True)
        return z, 0.0
    return torch.stack(losses).mean(), float(torch.stack(kls).mean())


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits (T, V) predicting targets (T,) -> log p(target_t)."""
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, targets[:, None]).squeeze(-1)
