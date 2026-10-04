"""Exact and near-duplicate removal.

* Exact: xxhash64 of an aggressively canonicalized text (case/punctuation/space
  insensitive, Tamil combining marks preserved).
* Near: MinHash over word n-gram shingles + banded LSH. With ``bands × rows``
  permutations, the collision probability for Jaccard similarity *s* is
  ``1 - (1 - s^rows)^bands``; the default 10 × 12 has its 50% point at s ≈ 0.82.
  Candidate pairs are optionally verified by estimated Jaccard.

The index is in-memory and single-process (first occurrence wins, so results are
deterministic given source order). For corpora beyond ~50M documents, run the
pipeline per source shard and dedup across shards with a second pass, or swap in
a disk-backed LSH (e.g. datatrove / text-dedup) — the interface here is tiny.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import xxhash

from data.normalize import dedup_key

_MERSENNE = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)


@dataclass
class DedupConfig:
    exact: bool = True
    near: bool = True
    ngram: int = 5              # word shingles; falls back to char shingles for short docs
    char_ngram: int = 12
    bands: int = 10
    rows: int = 12
    verify_threshold: float | None = 0.8   # None → accept any LSH collision
    seed: int = 1234

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "DedupConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class MinHasher:
    def __init__(self, num_perm: int, seed: int):
        rng = np.random.RandomState(seed)
        # a, b < 2^32 so a * h (h < 2^32) fits in uint64 before the Mersenne modulus.
        self.a = rng.randint(1, 2**32 - 1, size=num_perm, dtype=np.uint64)
        self.b = rng.randint(0, 2**32 - 1, size=num_perm, dtype=np.uint64)
        self.num_perm = num_perm

    def signature(self, shingles: set[str]) -> np.ndarray:
        if not shingles:
            return np.full(self.num_perm, _MAX_HASH, dtype=np.uint64)
        h = np.fromiter((xxhash.xxh32_intdigest(s.encode("utf-8")) for s in shingles),
                        dtype=np.uint64, count=len(shingles))
        # (a*h + b) mod p, truncated to 32 bits; shape (num_perm, n_shingles)
        phv = ((np.outer(self.a, h) + self.b[:, None]) % _MERSENNE) & _MAX_HASH
        return phv.min(axis=1)


def shingles(text: str, n: int, char_n: int) -> set[str]:
    words = dedup_key_words(text)
    if len(words) >= n:
        return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
    s = "".join(words)
    if len(s) <= char_n:
        return {s} if s else set()
    return {s[i:i + char_n] for i in range(len(s) - char_n + 1)}


def dedup_key_words(text: str) -> list[str]:
    return [w for w in (dedup_key(tok) for tok in text.split()) if w]


class Deduplicator:
    def __init__(self, config: DedupConfig | Mapping[str, Any] | None = None):
        if not isinstance(config, DedupConfig):
            config = DedupConfig.from_dict(config)
        self.cfg = config
        self._exact: set[int] = set()
        self._hasher = MinHasher(config.bands * config.rows, config.seed) if config.near else None
        self._buckets: list[dict[bytes, int]] = [dict() for _ in range(config.bands)] if config.near else []
        self._sigs: list[np.ndarray] = []
        self.stats = {"exact_dups": 0, "near_dups": 0, "unique": 0}

    def signature(self, text: str) -> np.ndarray:
        assert self._hasher is not None
        return self._hasher.signature(shingles(text, self.cfg.ngram, self.cfg.char_ngram))

    def is_duplicate(self, text: str, signature: np.ndarray | None = None) -> str | None:
        """Returns ``"exact"``/``"near"`` for duplicates, else registers the doc and returns None.

        ``signature`` may be precomputed (e.g. in a worker process) to keep the
        single-threaded index step cheap.
        """
        c = self.cfg
        if c.exact:
            key = xxhash.xxh64_intdigest(dedup_key(text).encode("utf-8"))
            if key in self._exact:
                self.stats["exact_dups"] += 1
                return "exact"
        if c.near:
            sig = signature if signature is not None else self.signature(text)
            band_keys = [sig[i * c.rows:(i + 1) * c.rows].tobytes() for i in range(c.bands)]
            for band, bk in zip(self._buckets, band_keys):
                other = band.get(bk)
                if other is None:
                    continue
                if c.verify_threshold is None or float(np.mean(self._sigs[other] == sig)) >= c.verify_threshold:
                    self.stats["near_dups"] += 1
                    if c.exact:
                        self._exact.add(key)
                    return "near"
            idx = len(self._sigs)
            self._sigs.append(sig if c.verify_threshold is not None else np.empty(0, dtype=np.uint64))
            for band, bk in zip(self._buckets, band_keys):
                band.setdefault(bk, idx)
        if c.exact:
            self._exact.add(key)
        self.stats["unique"] += 1
        return None
