"""Heuristic document-quality filters (Gopher / C4 / FineWeb style), adapted for Tamil.

Two stages:

1. ``LineFilter`` removes boilerplate lines (navigation, cookie banners, share
   buttons) and very short fragments, returning the cleaned document.
2. ``QualityFilter`` computes document statistics and rejects documents that
   fail any threshold. Thresholds can differ per language label because Tamil
   words are much longer than English ones (agglutination) and spoken/Tanglish
   text is naturally shorter and noisier.

Every rejection carries a reason string so the pipeline can report exactly why
data was dropped.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from data.langid import RESOURCES, load_wordlist

_WORD_RE = re.compile(r"\S+")
_BULLET_RE = re.compile(r"^\s*(?:[-*•●▪➤►✓✔]|\d+[.)])\s+")
_ELLIPSIS_RE = re.compile(r"(?:\.\.\.|…)\s*$")
_PUNCT_STRIP = ".,;:!?\"'()[]{}«»“”‘’-–—…"


def _load_patterns(name: str) -> list[str]:
    path = Path(name) if Path(name).exists() else RESOURCES / name
    return [
        ln.strip().lower()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]


@dataclass
class LineFilterConfig:
    enabled: bool = True
    min_chars: int = 0           # drop lines shorter than this (0 disables)
    min_words: int = 0
    boilerplate_patterns: str | None = "boilerplate_patterns.txt"
    drop_lines_with_curly_braces: bool = True   # C4: code / templates
    max_line_repeats: int = 1                   # identical line kept at most N times

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "LineFilterConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class LineFilter:
    def __init__(self, config: LineFilterConfig | Mapping[str, Any] | None = None):
        if not isinstance(config, LineFilterConfig):
            config = LineFilterConfig.from_dict(config)
        self.cfg = config
        self.patterns = _load_patterns(config.boilerplate_patterns) if config.boilerplate_patterns else []

    def __call__(self, text: str) -> str:
        c = self.cfg
        if not c.enabled:
            return text
        kept: list[str] = []
        seen: Counter[str] = Counter()
        for line in text.split("\n"):
            s = line.strip()
            if not s:
                kept.append("")
                continue
            low = s.lower()
            if c.min_chars and len(s) < c.min_chars:
                continue
            if c.min_words and len(s.split()) < c.min_words:
                continue
            if c.drop_lines_with_curly_braces and ("{" in s or "}" in s):
                continue
            if self.patterns and any(p in low for p in self.patterns):
                continue
            if c.max_line_repeats and seen[s] >= c.max_line_repeats:
                continue
            seen[s] += 1
            kept.append(s)
        out = "\n".join(kept)
        return re.sub(r"\n{3,}", "\n\n", out).strip()


@dataclass
class QualityThresholds:
    min_chars: int = 200
    max_chars: int = 500_000
    min_words: int = 30
    max_words: int = 100_000
    min_mean_word_len: float = 2.0
    max_mean_word_len: float = 20.0
    max_symbol_word_ratio: float = 0.1      # '#' and '…' per word
    max_bullet_line_frac: float = 0.9
    max_ellipsis_line_frac: float = 0.3
    max_digit_frac: float = 0.25
    max_punct_frac: float = 0.25
    max_dup_line_frac: float = 0.3
    max_dup_line_char_frac: float = 0.2
    max_top_ngram_frac: dict[int, float] = field(default_factory=lambda: {2: 0.20, 3: 0.18, 4: 0.16})
    max_dup_ngram_frac: dict[int, float] = field(default_factory=lambda: {5: 0.15, 7: 0.13, 10: 0.10})
    min_stopwords: int = 0
    stopwords: str | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None, base: "QualityThresholds | None" = None) -> "QualityThresholds":
        import dataclasses

        out = dataclasses.replace(base) if base else cls()
        for k, v in dict(d or {}).items():
            if k not in cls.__dataclass_fields__:
                raise KeyError(f"Unknown quality threshold: {k}")
            if k in ("max_top_ngram_frac", "max_dup_ngram_frac") and v is not None:
                v = {int(n): float(x) for n, x in dict(v).items()}
            setattr(out, k, v)
        return out


def _top_ngram_frac(words: list[str], n: int) -> float:
    """Share of characters covered by the single most frequent n-gram (Gopher)."""
    if len(words) < n + 1:
        return 0.0
    grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    gram, count = grams.most_common(1)[0]
    if count <= 1:
        return 0.0
    total = sum(len(w) for w in words)
    return (sum(len(w) for w in gram) * count) / max(total, 1)


def _dup_ngram_frac(words: list[str], n: int) -> float:
    """Share of characters inside n-grams that occur more than once (Gopher)."""
    if len(words) < n:
        return 0.0
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(grams)
    covered = [False] * len(words)
    for i, g in enumerate(grams):
        if counts[g] > 1:
            for j in range(i, i + n):
                covered[j] = True
    total = sum(len(w) for w in words)
    dup = sum(len(w) for w, c in zip(words, covered) if c)
    return dup / max(total, 1)


class QualityFilter:
    """Per-language thresholds: ``thresholds = {"default": {...}, "ta": {...}, "tanglish": {...}}``."""

    def __init__(self, thresholds: Mapping[str, Any] | None = None):
        thresholds = dict(thresholds or {})
        default = QualityThresholds.from_dict(thresholds.pop("default", None))
        self.by_lang = {"default": default}
        for lang, th in thresholds.items():
            self.by_lang[lang] = QualityThresholds.from_dict(th, base=default)
        self._stopwords: dict[str, frozenset[str]] = {}

    def _stops(self, name: str | None) -> frozenset[str]:
        if not name:
            return frozenset()
        if name not in self._stopwords:
            self._stopwords[name] = load_wordlist(name)
        return self._stopwords[name]

    def metrics(self, text: str) -> dict[str, float]:
        words = _WORD_RE.findall(text)
        lines = [ln for ln in text.split("\n") if ln.strip()]
        n_chars = len(text)
        n_words = len(words)
        stripped = [w.strip(_PUNCT_STRIP) for w in words]
        clean_words = [w for w in stripped if w]
        m: dict[str, float] = {
            "n_chars": n_chars,
            "n_words": n_words,
            "n_lines": len(lines),
            "mean_word_len": (sum(len(w) for w in clean_words) / len(clean_words)) if clean_words else 0.0,
            "symbol_word_ratio": (text.count("#") + text.count("...") + text.count("…")) / max(n_words, 1),
            "bullet_line_frac": sum(bool(_BULLET_RE.match(ln)) for ln in lines) / max(len(lines), 1),
            "ellipsis_line_frac": sum(bool(_ELLIPSIS_RE.search(ln)) for ln in lines) / max(len(lines), 1),
            "digit_frac": sum(ch.isdigit() for ch in text) / max(n_chars, 1),
            "punct_frac": sum(ch in _PUNCT_STRIP for ch in text) / max(n_chars, 1),
        }
        if lines:
            counts = Counter(lines)
            dup_lines = sum(c for c in counts.values() if c > 1)
            dup_chars = sum(len(ln) * c for ln, c in counts.items() if c > 1)
            m["dup_line_frac"] = dup_lines / len(lines)
            m["dup_line_char_frac"] = dup_chars / max(sum(len(ln) for ln in lines), 1)
        else:
            m["dup_line_frac"] = m["dup_line_char_frac"] = 0.0
        lw = [w.lower() for w in clean_words]
        for n in (2, 3, 4):
            m[f"top_{n}gram_frac"] = _top_ngram_frac(lw, n)
        for n in (5, 7, 10):
            m[f"dup_{n}gram_frac"] = _dup_ngram_frac(lw, n)
        m["_words"] = lw  # type: ignore[assignment]  # consumed by check(); stripped before output
        return m

    def check(self, text: str, lang: str) -> tuple[bool, str | None, dict[str, float]]:
        th = self.by_lang.get(lang, self.by_lang["default"])
        m = self.metrics(text)
        words = m.pop("_words")  # type: ignore[arg-type]

        def fail(reason: str):
            return False, reason, m

        if m["n_chars"] < th.min_chars:
            return fail("too_short_chars")
        if m["n_chars"] > th.max_chars:
            return fail("too_long_chars")
        if m["n_words"] < th.min_words:
            return fail("too_few_words")
        if m["n_words"] > th.max_words:
            return fail("too_many_words")
        if not th.min_mean_word_len <= m["mean_word_len"] <= th.max_mean_word_len:
            return fail("mean_word_len")
        if m["symbol_word_ratio"] > th.max_symbol_word_ratio:
            return fail("symbol_ratio")
        if m["bullet_line_frac"] > th.max_bullet_line_frac:
            return fail("bullet_lines")
        if m["ellipsis_line_frac"] > th.max_ellipsis_line_frac:
            return fail("ellipsis_lines")
        if m["digit_frac"] > th.max_digit_frac:
            return fail("digit_frac")
        if m["punct_frac"] > th.max_punct_frac:
            return fail("punct_frac")
        if m["dup_line_frac"] > th.max_dup_line_frac:
            return fail("dup_lines")
        if m["dup_line_char_frac"] > th.max_dup_line_char_frac:
            return fail("dup_line_chars")
        for n, limit in (th.max_top_ngram_frac or {}).items():
            if m.get(f"top_{n}gram_frac", 0.0) > limit:
                return fail(f"top_{n}gram")
        for n, limit in (th.max_dup_ngram_frac or {}).items():
            if m.get(f"dup_{n}gram_frac", 0.0) > limit:
                return fail(f"dup_{n}gram")
        if th.min_stopwords and th.stopwords:
            stops = self._stops(th.stopwords)
            if sum(w in stops for w in words) < th.min_stopwords:
                return fail("few_stopwords")
        return True, None, m
