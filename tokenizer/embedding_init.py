"""Initialize embeddings for new/changed vocabulary from the base model's embeddings.

For each target token we find its surface string, tokenize it with the *base*
tokenizer and average the base embeddings of those sub-tokens (the standard
"mean of sub-word embeddings" heuristic; cf. Chinese-LLaMA, WECHSEL/FOCUS without
the auxiliary embeddings). Tokens whose exact piece exists in the base vocabulary
(overlap — special tokens, byte-fallback tokens, shared pieces) are copied.

A new token initialized this way starts with a representation close to what the
model already "reads" for that character sequence, which dramatically reduces
the initial loss spike compared with random init.
"""

from __future__ import annotations

from typing import Iterable

import torch

from common.utils import get_logger

log = get_logger(__name__)


def piece_to_text(tokenizer, token_id: int) -> str:
    piece = tokenizer.convert_ids_to_tokens(token_id)
    return tokenizer.convert_tokens_to_string([piece]) if piece is not None else ""


@torch.no_grad()
def build_init_matrix(base_emb: torch.Tensor, base_tok, new_tok, target_ids: Iterable[int],
                      copy_exact: bool = True) -> tuple[torch.Tensor, dict[str, int]]:
    """Returns ``(rows [len(target_ids), dim], stats)`` computed in float32."""
    target_ids = list(target_ids)
    base_vocab = base_tok.get_vocab()
    emb = base_emb.float()
    fallback = emb.mean(dim=0)
    out = torch.empty(len(target_ids), emb.shape[1], dtype=torch.float32)
    stats = {"copied": 0, "mean_subword": 0, "fallback": 0}
    for row, tid in enumerate(target_ids):
        piece = new_tok.convert_ids_to_tokens(tid)
        if copy_exact and piece in base_vocab and base_vocab[piece] < emb.shape[0]:
            out[row] = emb[base_vocab[piece]]
            stats["copied"] += 1
            continue
        text = piece_to_text(new_tok, tid)
        ids = [i for i in base_tok(text, add_special_tokens=False)["input_ids"] if i < emb.shape[0]] if text else []
        if ids:
            out[row] = emb[ids].mean(dim=0)
            stats["mean_subword"] += 1
        else:
            out[row] = fallback
            stats["fallback"] += 1
    return out, stats


@torch.no_grad()
def resize_and_init(model, base_tok, new_tok, target_ids: Iterable[int] | None, pad_to_multiple_of: int = 64,
                    copy_exact: bool = True) -> dict[str, int]:
    """Resize ``model`` embeddings to ``len(new_tok)`` and initialize ``target_ids`` rows.

    ``target_ids=None`` re-initializes every row (vocabulary replacement).
    Rows added purely for padding are set to the mean embedding.
    """
    old_in = model.get_input_embeddings().weight.detach().clone()
    out_layer = model.get_output_embeddings()
    tied = out_layer is None or out_layer.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    old_out = None if tied else out_layer.weight.detach().clone()

    n_new = len(new_tok)
    model.resize_token_embeddings(n_new, pad_to_multiple_of=pad_to_multiple_of, mean_resizing=False)
    in_w = model.get_input_embeddings().weight
    full = in_w.shape[0]
    ids = list(range(n_new)) if target_ids is None else sorted(set(target_ids))
    rows, stats = build_init_matrix(old_in, base_tok, new_tok, ids, copy_exact)
    in_w[ids] = rows.to(in_w.dtype)
    if full > n_new:
        in_w[n_new:] = old_in.float().mean(dim=0).to(in_w.dtype)
    if not tied:
        out_w = model.get_output_embeddings().weight
        out_rows, _ = build_init_matrix(old_out, base_tok, new_tok, ids, copy_exact)
        out_w[ids] = out_rows.to(out_w.dtype)
        if full > n_new:
            out_w[n_new:] = old_out.float().mean(dim=0).to(out_w.dtype)
    stats.update({"initialized": len(ids), "embedding_rows": full, "tied": int(tied)})
    log.info("Embedding init: %s", stats)
    return stats
