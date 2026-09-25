"""Build (model, tokenizer) pairs from config.

``model.source``:

* ``pretrained`` – load ``base_model.name_or_path`` (Gemma 3 270M) and adapt the
  vocabulary according to ``tokenizer.adaptation``:
    - ``none``    keep Gemma's tokenizer
    - ``extend``  Gemma tokenizer + Tamil pieces (``tokenizer/adapt.py``); only the
                  new rows are initialized (mean of Gemma sub-token embeddings)
    - ``replace`` a Tamil/bilingual tokenizer replaces the vocabulary; every row
                  is initialized from Gemma embeddings (copy on exact match, else mean)
* ``scratch``    – random-init Gemma 3 architecture from ``model.architecture``
  (used for the ~10M experimental run)
* ``checkpoint`` – load model + tokenizer saved by a previous stage (``model.path``)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from common.config import to_container
from common.utils import get_logger, torch_dtype
from tokenizer.embedding_init import resize_and_init

log = get_logger(__name__)


def _load_tokenizer(path: str, revision: str | None = None):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, revision=revision)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _model_kwargs(mcfg: dict[str, Any]) -> dict[str, Any]:
    kw: dict[str, Any] = {"torch_dtype": torch_dtype(mcfg.get("load_dtype", "fp32"))}
    if mcfg.get("attn_implementation"):
        kw["attn_implementation"] = mcfg["attn_implementation"]
    return kw


def build_scratch(mcfg: dict[str, Any], tok):
    from transformers import Gemma3ForCausalLM, Gemma3TextConfig

    arch = dict(mcfg["architecture"])
    mult = int(mcfg.get("pad_vocab_to_multiple_of", 64))
    vocab = ((len(tok) + mult - 1) // mult) * mult
    config = Gemma3TextConfig(
        vocab_size=vocab, pad_token_id=tok.pad_token_id, bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id, **arch)
    if mcfg.get("attn_implementation"):
        config._attn_implementation = mcfg["attn_implementation"]
    torch.manual_seed(int(mcfg.get("init_seed", 0)))
    model = Gemma3ForCausalLM(config)
    return model


def build_model_and_tokenizer(cfg) -> tuple[Any, Any, dict[str, Any]]:
    from transformers import AutoModelForCausalLM

    mcfg = to_container(cfg.model)
    tcfg = to_container(cfg.get("tokenizer") or {})
    source = mcfg.get("source", "pretrained")
    info: dict[str, Any] = {"source": source}

    if source == "checkpoint":
        path = mcfg["path"]
        tok = _load_tokenizer(path)
        model = AutoModelForCausalLM.from_pretrained(path, **_model_kwargs(mcfg))
    elif source == "scratch":
        tok = _load_tokenizer(tcfg["path"])
        model = build_scratch(mcfg, tok)
    elif source == "pretrained":
        base_name = cfg.base_model.name_or_path
        revision = cfg.base_model.get("revision")
        model = AutoModelForCausalLM.from_pretrained(base_name, revision=revision, **_model_kwargs(mcfg))
        adaptation = tcfg.get("adaptation", "none")
        info["adaptation"] = adaptation
        base_tok = _load_tokenizer(base_name, revision)
        if adaptation == "none":
            tok = base_tok
        elif adaptation == "extend":
            tok = _load_tokenizer(tcfg["path"])
            manifest = json.loads((Path(tcfg["path"]) / "extension_manifest.json").read_text(encoding="utf-8"))
            info["init"] = resize_and_init(model, base_tok, tok, manifest["new_token_ids"],
                                           int(mcfg.get("pad_vocab_to_multiple_of", 64)))
        elif adaptation == "replace":
            tok = _load_tokenizer(tcfg["path"])
            info["init"] = resize_and_init(model, base_tok, tok, None, int(mcfg.get("pad_vocab_to_multiple_of", 64)))
        else:
            raise ValueError(f"Unknown tokenizer.adaptation {adaptation!r}")
        model.config.pad_token_id = tok.pad_token_id
        model.config.bos_token_id = tok.bos_token_id
        model.config.eos_token_id = tok.eos_token_id
    else:
        raise ValueError(f"Unknown model.source {source!r}")

    if model.get_input_embeddings().weight.shape[0] < len(tok):
        raise ValueError(f"Model vocab ({model.get_input_embeddings().weight.shape[0]}) < tokenizer ({len(tok)})")
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tok.pad_token_id
        model.generation_config.bos_token_id = tok.bos_token_id
        eos = [tok.eos_token_id]
        eot = tok.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(eot, int) and eot != tok.unk_token_id:
            eos.append(eot)
        model.generation_config.eos_token_id = eos
    info.update(param_report(model))
    log.info("Model: %s | params total=%.1fM (embedding %.1fM, non-embedding %.1fM) | vocab=%d",
             type(model).__name__, info["params_total"] / 1e6, info["params_embedding"] / 1e6,
             info["params_non_embedding"] / 1e6, len(tok))
    return model, tok, info


def param_report(model) -> dict[str, int]:
    emb = model.get_input_embeddings().weight.numel()
    seen, total = set(), 0
    for p in model.parameters():
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        total += p.numel()
    return {"params_total": total, "params_embedding": emb, "params_non_embedding": total - emb}


def apply_trainable(model, policy: str) -> dict[str, int]:
    """``all`` | ``embeddings`` | ``embeddings_and_norms``. Returns trainable counts."""
    if policy == "all":
        for p in model.parameters():
            p.requires_grad_(True)
    elif policy in ("embeddings", "embeddings_and_norms"):
        for p in model.parameters():
            p.requires_grad_(False)
        model.get_input_embeddings().weight.requires_grad_(True)
        out = model.get_output_embeddings()
        if out is not None:
            out.weight.requires_grad_(True)
        if policy == "embeddings_and_norms":
            for name, p in model.named_parameters():
                if "norm" in name:
                    p.requires_grad_(True)
    else:
        raise ValueError(f"Unknown trainable policy {policy!r}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Trainable policy %s: %.2fM trainable parameters", policy, trainable / 1e6)
    return {"trainable_params": trainable}
