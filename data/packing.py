"""Tokenize processed shards, build a token-weighted mixture, and pack into fixed-length blocks.

    python -m data.packing --config configs/train/experimental_10m.yaml

Reads the ``data`` section of a *training* config (mixture, seq_len, tokenizer), so
the packed dataset is tied to the run that uses it. Mixture weights are shares of
**tokens** in the final training set:

    target_tokens[c] = min(weight[c] * budget, available[c] * max_epochs[c])
    budget           = data.token_budget, or (default) the budget at which the anchor
                       category (data.budget_anchor, default: largest weight) is used for
                       data.anchor_epochs passes (default 1.0)

Categories below target are up-sampled (whole-document repeats + a partial pass) up to
their max_epochs cap; categories above target are down-sampled. Capped categories end
up below their nominal share — the final shares are logged and stored. Every choice
is seeded, and ``manifest.json`` records available / selected tokens, epochs, shares.

Output (``data.packed_dir``)::

    train/            HF dataset, column input_ids: int32[seq_len]
    eval_<name>/      one per data.eval_sets entry (Tamil, English, Tanglish, ...)
    manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from common.config import parse_config, save_run_metadata, to_container
from common.utils import get_logger, set_seed, setup_logging, text_hash

log = get_logger("data.packing")


def _files(processed_dir: Path, category: str, split: str) -> list[str]:
    d = processed_dir / category
    return sorted(str(p) for p in d.glob(f"{split}-*.jsonl*")) if d.is_dir() else []


def _tokenize(ds, tok, num_proc: int, add_bos: bool, add_eos: bool):
    bos, eos = tok.bos_token_id, tok.eos_token_id

    def fn(batch):
        enc = tok(batch["text"], add_special_tokens=False)["input_ids"]
        ids = [([bos] if add_bos else []) + e + ([eos] if add_eos else []) for e in enc]
        return {"input_ids": ids, "n_tokens": [len(x) for x in ids]}

    return ds.map(fn, batched=True, batch_size=1000, num_proc=num_proc if num_proc > 1 else None,
                  remove_columns=ds.column_names, desc="tokenize")


def _pack(ds, seq_len: int, num_proc: int):
    from datasets import Features, Sequence, Value

    def fn(batch):
        flat = [t for ids in batch["input_ids"] for t in ids]
        n = len(flat) // seq_len
        return {"input_ids": [flat[i * seq_len:(i + 1) * seq_len] for i in range(n)]}

    feats = Features({"input_ids": Sequence(Value("int32"))})
    return ds.map(fn, batched=True, batch_size=2000, num_proc=num_proc if num_proc > 1 else None,
                  remove_columns=ds.column_names, features=feats, desc="pack")


def _ntokens(tds) -> np.ndarray:
    return tds.data.column("n_tokens").to_numpy()


def _select(n_tokens: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    """Indices (with repeats) whose token sum ≈ target: full passes + a partial pass."""
    total = int(n_tokens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    full = target // total
    idx = [rng.permutation(len(n_tokens)) for _ in range(int(full))]
    remainder = target - full * total
    if remainder > 0:
        perm = rng.permutation(len(n_tokens))
        csum = np.cumsum(n_tokens[perm])
        k = int(np.searchsorted(csum, remainder)) + 1
        idx.append(perm[:k])
    return np.concatenate(idx) if idx else np.zeros(0, dtype=np.int64)


def compute_plan(available: dict[str, int], weights: dict[str, float], budget: int | None,
                 max_epochs: dict[str, float], default_max_epochs: float, anchor: str | None = None,
                 anchor_epochs: float = 1.0) -> dict[str, dict[str, float]]:
    w = {c: float(x) for c, x in weights.items() if x > 0 and available.get(c, 0) > 0}
    missing = [c for c, x in weights.items() if x > 0 and available.get(c, 0) == 0]
    if missing:
        log.warning("Mixture categories with no data (weights renormalized): %s", missing)
    if not w:
        raise ValueError("No mixture category has data")
    z = sum(w.values())
    w = {c: x / z for c, x in w.items()}
    cap = {c: float(max_epochs.get(c, default_max_epochs)) for c in w}
    if budget is None:
        anchor = anchor or max(w, key=w.get)
        if anchor not in w:
            raise ValueError(f"budget_anchor {anchor!r} has no data / zero weight")
        budget = round(available[anchor] * anchor_epochs / w[anchor])
        log.info("Token budget %.4gM set by anchor %s (%.2f epochs)", budget / 1e6, anchor, anchor_epochs)
    plan = {}
    for c in w:
        target = min(round(w[c] * budget), round(available[c] * cap[c]))
        if target < round(w[c] * budget):
            log.warning("Category %s capped at %.1f epochs: %.3gM of %.3gM target tokens", c, cap[c],
                        target / 1e6, w[c] * budget / 1e6)
        plan[c] = {"weight": w[c], "available_tokens": available[c], "target_tokens": target,
                   "epochs": target / available[c]}
    return plan


def run(cfg) -> Path:
    from datasets import concatenate_datasets, load_dataset
    from transformers import AutoTokenizer

    set_seed(int(cfg.seed))
    d = to_container(cfg.data)
    out = Path(d["packed_dir"])
    if (out / "manifest.json").exists() and not d.get("overwrite_packed", False):
        log.info("Packed data already exists at %s (set data.overwrite_packed=true to rebuild)", out)
        return out
    out.mkdir(parents=True, exist_ok=True)
    save_run_metadata(cfg, out, "packing")
    processed = Path(d["processed_dir"])
    tok_path = d.get("tokenizer_path") or cfg.tokenizer.get("path") or cfg.base_model.name_or_path
    tok = AutoTokenizer.from_pretrained(tok_path)
    tok.model_max_length = 10**9  # documents are packed, so long-sequence warnings are noise
    seq_len = int(d["seq_len"])
    num_proc = int(d.get("num_proc", 1))
    rng = np.random.default_rng(int(cfg.seed))
    add_bos, add_eos = bool(d.get("add_bos", True)), bool(d.get("add_eos", True))

    mixture = {c: float(w) for c, w in d["mixture"].items()}
    tokenized: dict[str, Any] = {}
    available: dict[str, int] = {}
    for cat in mixture:
        files = _files(processed, cat, "train")
        if not files:
            available[cat] = 0
            continue
        ds = load_dataset("json", data_files=files, split="train").select_columns(["text"])
        tds = _tokenize(ds, tok, num_proc, add_bos, add_eos)
        tokenized[cat] = tds
        available[cat] = int(_ntokens(tds).sum())
        log.info("  %-22s %8d docs %12d tokens", cat, len(tds), available[cat])

    budget = d.get("token_budget")
    plan = compute_plan(available, mixture, int(float(budget)) if budget else None,
                        d.get("max_epochs_per_category") or {}, float(d.get("max_epochs", 4.0)),
                        d.get("budget_anchor"), float(d.get("anchor_epochs", 1.0)))
    parts = []
    for cat, p in plan.items():
        tds = tokenized[cat]
        ntok = _ntokens(tds)
        idx = _select(ntok, int(p["target_tokens"]), rng)
        sel = tds.select(idx)
        p["selected_docs"] = int(len(idx))
        p["selected_tokens"] = int(ntok[idx].sum()) if len(idx) else 0
        parts.append(_pack(sel.select_columns(["input_ids"]), seq_len, num_proc))
    train = concatenate_datasets(parts).shuffle(seed=int(cfg.seed)).flatten_indices()
    train.save_to_disk(str(out / "train"))
    total_sel = sum(p["selected_tokens"] for p in plan.values())
    for p in plan.values():
        p["share"] = p["selected_tokens"] / max(total_sel, 1)

    eval_info = {}
    for name, cats in (d.get("eval_sets") or {}).items():
        files = [f for c in cats for f in _files(processed, c, "eval")]
        if not files:
            log.warning("Eval set %s has no eval shards (%s); skipped", name, cats)
            continue
        ds = load_dataset("json", data_files=files, split="train").select_columns(["text"])
        packed = _pack(_tokenize(ds, tok, num_proc, add_bos, add_eos).select_columns(["input_ids"]),
                       seq_len, num_proc)
        if len(packed) == 0:
            log.warning("Eval set %s is shorter than one %d-token block; skipped", name, seq_len)
            continue
        max_blocks = int(d.get("eval_max_blocks", 500))
        if len(packed) > max_blocks:
            packed = packed.shuffle(seed=int(cfg.seed)).select(range(max_blocks)).flatten_indices()
        packed.save_to_disk(str(out / f"eval_{name}"))
        eval_info[name] = {"categories": cats, "blocks": len(packed), "tokens": len(packed) * seq_len}

    manifest = {
        "tokenizer": str(tok_path),
        "tokenizer_vocab_size": len(tok),
        "tokenizer_fingerprint": text_hash(json.dumps(sorted(tok.get_vocab().items()), ensure_ascii=False)),
        "seq_len": seq_len,
        "train_blocks": len(train),
        "train_tokens": len(train) * seq_len,
        "plan": plan,
        "eval": eval_info,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Packed %d train blocks (%.2fM tokens) → %s", len(train), len(train) * seq_len / 1e6, out)
    for cat, p in plan.items():
        log.info("  %-22s share=%5.1f%% (target %4.1f%%) epochs=%.2f", cat, 100 * p["share"],
                 100 * p["weight"], p["epochs"])
    return out


def main(argv=None) -> None:
    setup_logging()
    run(parse_config("Tokenize, mix and pack pretraining data", argv))


if __name__ == "__main__":
    main()
