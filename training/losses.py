"""Memory-lean log-probabilities for causal LMs with a huge vocabulary.

Gemma 3 270M projects every position onto 262,144 vocabulary entries. For a batch
of 4 × 384 tokens the full logit tensor (plus its gradient) is ~3 GB in fp32 —
more than the model itself, and the main reason CPU training swaps on a 16 GB
laptop. Two tricks keep this small, both mathematically exact:

* **label-only**: only positions that carry a label (assistant tokens in SFT,
  response tokens in DPO/GRPO) go through the output layer;
* **chunked + recomputed**: those positions are processed ``chunk`` at a time under
  activation checkpointing, so at most ``chunk × vocab`` logits exist at once
  (256 × 262k × 4 B ≈ 270 MB) in forward *and* backward.

All functions accept plain HF models, DDP-wrapped models and PEFT (LoRA) models.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

DEFAULT_CHUNK = 256


def unwrap(model):
    """Return the underlying ``*ForCausalLM`` (strips DDP and PEFT wrappers; LoRA layers stay injected)."""
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "get_base_model"):
        m = m.get_base_model()
    return m


def _chunk_logps(core, hidden: torch.Tensor, targets: torch.Tensor, chunk: int) -> torch.Tensor:
    cap = getattr(core.config, "final_logit_softcapping", None)

    def f(h, t):
        logits = core.lm_head(h).float()
        if cap:
            logits = torch.tanh(logits / cap) * cap
        return -F.cross_entropy(logits, t, reduction="none")

    if hidden.shape[0] == 0:
        return hidden.new_zeros(0, dtype=torch.float32)
    out = []
    for i in range(0, hidden.shape[0], chunk):
        h, t = hidden[i:i + chunk], targets[i:i + chunk]
        if torch.is_grad_enabled() and h.requires_grad:
            out.append(checkpoint(f, h, t, use_reentrant=False))
        else:
            out.append(f(h, t))
    return torch.cat(out)


def label_token_logps(model, input_ids, attention_mask, labels, chunk: int = DEFAULT_CHUNK):
    """Log-prob of every labelled next token.

    Returns ``(logps, keep)``: ``logps`` is a flat tensor over positions where
    ``labels[:, 1:] != -100`` (row-major order), ``keep`` the (B, L-1) boolean mask.
    """
    core = unwrap(model)
    hidden = core.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    targets = labels[:, 1:]
    keep = targets != -100
    return _chunk_logps(core, hidden[:, :-1][keep], targets[keep], chunk), keep


def label_only_loss(model, input_ids, attention_mask, labels, num_items_in_batch=None,
                    chunk: int = DEFAULT_CHUNK) -> torch.Tensor:
    """Mean next-token NLL over labelled positions (identical to the HF loss, far less memory)."""
    logps, keep = label_token_logps(model, input_ids, attention_mask, labels, chunk)
    denom = num_items_in_batch if num_items_in_batch is not None else keep.sum()
    return -logps.sum() / torch.as_tensor(denom, device=logps.device).clamp(min=1)


def sequence_logps(model, input_ids, attention_mask, labels, chunk: int = DEFAULT_CHUNK):
    """Per-sequence summed log-prob of the labelled tokens and their counts: ``(sum (B,), n (B,))``."""
    logps, keep = label_token_logps(model, input_ids, attention_mask, labels, chunk)
    rows = keep.nonzero(as_tuple=True)[0]
    total = torch.zeros(keep.shape[0], dtype=logps.dtype, device=logps.device).index_add(0, rows, logps)
    return total, keep.sum(dim=1)


def per_token_logps(model, input_ids, attention_mask, labels, chunk: int = DEFAULT_CHUNK):
    """Dense (B, L-1) log-probs, zero where unlabelled, plus the mask (used by GRPO for per-token KL)."""
    logps, keep = label_token_logps(model, input_ids, attention_mask, labels, chunk)
    dense = torch.zeros(keep.shape, dtype=logps.dtype, device=logps.device)
    dense[keep] = logps
    return dense, keep
