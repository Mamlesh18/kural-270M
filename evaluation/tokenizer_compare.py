"""Compare tokenizers (e.g. original Gemma vs Tamil BPE vs Gemma+Tamil extension) on identical text.

Metrics per tokenizer × text set:

* ``fertility``          tokens per whitespace word (lower is better)
* ``chars_per_token``    / ``bytes_per_token`` — compression
* ``tokens_per_grapheme`` Tamil aksharas are 1–3 code points; ~1.0 means character-level
* ``continued_word_ratio`` share of words split into ≥2 tokens
* ``byte_fallback_rate``  share of tokens that are raw ``<0xNN>`` bytes (unseen characters)
* ``roundtrip_exact``     decode(encode(x)) == x
* ``vocab_utilization``   distinct ids used / vocab size

Fewer tokens for the same text means a longer effective context and cheaper
training/inference; it does *not* by itself mean better modelling — compare
bits-per-byte in ``evaluation.perplexity`` for that.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from common.tamil import count_graphemes
from common.utils import get_logger

log = get_logger(__name__)
_BYTE_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")


def tokenizer_metrics(tok, texts: list[str]) -> dict[str, float]:
    vocab_size = len(tok)
    unk = tok.unk_token_id
    byte_ids = {i for t, i in tok.get_vocab().items() if _BYTE_RE.match(t)}

    @lru_cache(maxsize=200_000)
    def word_len(w: str) -> int:
        return len(tok(" " + w, add_special_tokens=False)["input_ids"])

    n_tok = n_words = n_chars = n_bytes = n_graph = n_bytefb = n_unk = roundtrip = continued = 0
    used: set[int] = set()
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n_tok += len(ids)
        words = t.split()
        n_words += len(words)
        n_chars += len(t)
        n_bytes += len(t.encode("utf-8"))
        n_graph += count_graphemes(t)
        n_bytefb += sum(i in byte_ids for i in ids)
        n_unk += sum(i == unk for i in ids)
        used.update(ids)
        roundtrip += tok.decode(ids) == t
        continued += sum(word_len(w) > 1 for w in words)
    return {
        "docs": len(texts),
        "tokens": n_tok,
        "fertility": n_tok / max(n_words, 1),
        "chars_per_token": n_chars / max(n_tok, 1),
        "bytes_per_token": n_bytes / max(n_tok, 1),
        "tokens_per_grapheme": n_tok / max(n_graph, 1),
        "continued_word_ratio": continued / max(n_words, 1),
        "byte_fallback_rate": n_bytefb / max(n_tok, 1),
        "unk_rate": n_unk / max(n_tok, 1),
        "roundtrip_exact": roundtrip / max(len(texts), 1),
        "vocab_size": vocab_size,
        "vocab_utilization": len(used) / max(vocab_size, 1),
    }


def compare(tokenizers: dict[str, Any], text_sets: dict[str, list[str]]) -> dict[str, dict[str, dict[str, float]]]:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for tname, tok in tokenizers.items():
        out[tname] = {}
        for sname, texts in text_sets.items():
            if texts:
                out[tname][sname] = tokenizer_metrics(tok, texts)
        log.info("tokenizer %s: %s", tname,
                 ", ".join(f"{s}={m['fertility']:.2f}" for s, m in out[tname].items()))
    return out


def to_markdown(results: dict[str, dict[str, dict[str, float]]], baseline: str | None = None) -> str:
    sets = sorted({s for r in results.values() for s in r})
    lines = ["| tokenizer | vocab | " + " | ".join(f"{s} fertility" for s in sets) + " | " +
             " | ".join(f"{s} bytes/tok" for s in sets) + " |",
             "|---|---:|" + "---:|" * (2 * len(sets))]
    for t, r in results.items():
        vocab = next(iter(r.values()))["vocab_size"] if r else 0
        fert = []
        for s in sets:
            if s not in r:
                fert.append("–")
                continue
            cell = f"{r[s]['fertility']:.2f}"
            if baseline and baseline in results and s in results[baseline] and t != baseline:
                delta = r[s]["tokens"] / max(results[baseline][s]["tokens"], 1) - 1
                cell += f" ({100 * delta:+.0f}%)"
            fert.append(cell)
        bpt = [f"{r[s]['bytes_per_token']:.2f}" if s in r else "–" for s in sets]
        lines.append(f"| {t} | {vocab:,} | " + " | ".join(fert) + " | " + " | ".join(bpt) + " |")
    return "\n".join(lines)
