"""Tokenizer-independent language-modelling evaluation.

Token-level perplexity is NOT comparable across tokenizers (a tokenizer that
splits text into more, easier tokens gets a lower PPL). We therefore report, on
the exact same raw documents:

* ``bits_per_byte``  = total NLL / ln 2 / UTF-8 bytes        ← primary, comparable
* ``bits_per_char``  = total NLL / ln 2 / code points
* ``token_ppl``      = exp(total NLL / predicted tokens)   (same-tokenizer comparisons only)
* ``word_ppl``       = exp(total NLL / whitespace words)   (comparable, intuitive scale)

Long documents are scored with a sliding window (``max_length``, ``stride``) so
each token is predicted with up to ``max_length - stride`` tokens of context.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from common.utils import get_logger

log = get_logger(__name__)


@torch.no_grad()
def document_nll(model, ids: list[int], max_length: int, stride: int, device) -> tuple[float, int]:
    """Sum of NLL (nats) over tokens ids[1:], and how many tokens were predicted."""
    total, count = 0.0, 0
    prev_end = 1  # position 0 (<bos>) is never predicted
    L = len(ids)
    begin = 0
    while prev_end < L:
        end = min(begin + max_length, L)
        window = torch.tensor(ids[begin:end], device=device).unsqueeze(0)
        logits = model(window).logits[0, :-1].float()
        targets = window[0, 1:]
        nll = F.cross_entropy(logits, targets, reduction="none")
        # nll[j] predicts absolute position begin + j + 1; keep positions not scored before.
        first_new = max(prev_end - begin - 1, 0)
        total += float(nll[first_new:].sum())
        count += int(nll.shape[0] - first_new)
        prev_end = end
        if end == L:
            break
        begin = end - (max_length - stride) if stride < max_length else end - 1
        begin = max(begin, 0)
    return total, count


def evaluate_texts(model, tok, texts: list[str], max_length: int = 1024, stride: int = 512,
                   device: str | torch.device | None = None) -> dict[str, float]:
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    nll = tokens = n_bytes = n_chars = n_words = 0.0
    for t in texts:
        ids = [tok.bos_token_id] + tok(t, add_special_tokens=False)["input_ids"]
        if len(ids) < 2:
            continue
        s, c = document_nll(model, ids, max_length, stride, device)
        nll += s
        tokens += c
        n_bytes += len(t.encode("utf-8"))
        n_chars += len(t)
        n_words += len(t.split())
    if was_training:
        model.train()
    ln2 = math.log(2)
    return {
        "docs": len(texts),
        "tokens": int(tokens),
        "nll_nats": nll,
        "bits_per_byte": nll / ln2 / max(n_bytes, 1),
        "bits_per_char": nll / ln2 / max(n_chars, 1),
        "token_ppl": math.exp(min(nll / max(tokens, 1), 50)),
        "word_ppl": math.exp(min(nll / max(n_words, 1), 50)),
    }


def evaluate_model(model, tok, text_sets: dict[str, list[str]], **kw: Any) -> dict[str, dict[str, float]]:
    out = {}
    for name, texts in text_sets.items():
        if not texts:
            continue
        out[name] = evaluate_texts(model, tok, texts, **kw)
        log.info("  %-12s bpb=%.4f  token_ppl=%.2f  word_ppl=%.1f  (%d docs)", name, out[name]["bits_per_byte"],
                 out[name]["token_ppl"], out[name]["word_ppl"], out[name]["docs"])
    return out
