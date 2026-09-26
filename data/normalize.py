"""Unicode + Tamil-specific text normalization.

Order matters: we decode HTML entities, strip invisible/control characters,
apply NFC (which composes split two-part vowel signs such as ெ+ா → ொ), then
repair common Tamil typing errors, and finally normalize whitespace.
"""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Mapping

import regex

from common.tamil import TAMIL_DIGITS, VIRAMA, VOWEL_SIGNS

# Zero-width and formatting characters that carry no meaning in Tamil running text.
# ZWJ/ZWNJ (U+200D/U+200C) are occasionally used to force rendering variants; they
# fragment tokenization and are dropped by default.
_INVISIBLE = "​‌‍⁠﻿­‎‏‪‫‬‭‮"
_INVISIBLE_RE = re.compile(f"[{_INVISIBLE}]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_MULTISPACE_RE = re.compile(r"[ \t  -   　]+")

# Two dependent vowel signs / double virama in a row are always typing errors.
_DOUBLE_SIGN_RE = re.compile(f"([{VOWEL_SIGNS}{VIRAMA}])\\1+")
# A virama immediately followed by a vowel sign (e.g. க்ா) is invalid; keep the vowel sign.
_VIRAMA_SIGN_RE = re.compile(f"{VIRAMA}([{VOWEL_SIGNS}])")

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Indian mobile numbers (+91 optional) and generic long digit runs with separators.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")

_QUOTES = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "′": "'", "″": '"',
}
_DASHES = {"‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-", "―": "-"}
_NON_WORD_RE = regex.compile(r"[^\p{L}\p{M}\p{N}]+")
_TAMIL_DIGIT_MAP = {ord(d): str(i) for i, d in enumerate(TAMIL_DIGITS)}
# ஶ (U+0BB6, SHA) is often used interchangeably with ஸ in ஸ்ரீ; the grantha form is kept by default.
_SRI_VARIANTS = {"ஶ்ரீ": "ஸ்ரீ"}


@dataclass
class NormalizerConfig:
    unicode_form: str = "NFC"            # NFC | NFKC | none
    html_unescape: bool = True
    remove_invisible: bool = True
    remove_control: bool = True
    fix_tamil_signs: bool = True
    unify_sri: bool = True
    tamil_digits_to_ascii: bool = False
    normalize_quotes: bool = True
    normalize_dashes: bool = True
    urls: str = "keep"                   # keep | remove | mask
    emails: str = "mask"                 # keep | remove | mask
    phones: str = "mask"                 # keep | remove | mask
    collapse_whitespace: bool = True
    max_consecutive_newlines: int = 2
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "NormalizerConfig":
        d = dict(d or {})
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__ and k != "extra"}
        return cls(**known, extra=d)


def _apply_pii(text: str, pattern: re.Pattern, mode: str, token: str) -> str:
    if mode == "keep":
        return text
    return pattern.sub("" if mode == "remove" else token, text)


class TextNormalizer:
    def __init__(self, config: NormalizerConfig | Mapping[str, Any] | None = None):
        if not isinstance(config, NormalizerConfig):
            config = NormalizerConfig.from_dict(config)
        self.cfg = config
        table: dict[int, str] = {}
        if config.normalize_quotes:
            table.update({ord(k): v for k, v in _QUOTES.items()})
        if config.normalize_dashes:
            table.update({ord(k): v for k, v in _DASHES.items()})
        if config.tamil_digits_to_ascii:
            table.update(_TAMIL_DIGIT_MAP)
        self._table = table

    def __call__(self, text: str) -> str:
        c = self.cfg
        if not text:
            return ""
        if c.html_unescape and "&" in text:
            text = html.unescape(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        if c.remove_invisible:
            text = _INVISIBLE_RE.sub("", text)
        if c.remove_control:
            text = _CONTROL_RE.sub("", text)
        if c.unicode_form and c.unicode_form.lower() != "none":
            text = unicodedata.normalize(c.unicode_form.upper(), text)
        if c.fix_tamil_signs:
            text = _DOUBLE_SIGN_RE.sub(r"\1", text)
            text = _VIRAMA_SIGN_RE.sub(r"\1", text)
        if c.unify_sri:
            for src, dst in _SRI_VARIANTS.items():
                text = text.replace(src, dst)
        if self._table:
            text = text.translate(self._table)
        text = _apply_pii(text, _URL_RE, c.urls, "<url>")
        text = _apply_pii(text, _EMAIL_RE, c.emails, "<email>")
        text = _apply_pii(text, _PHONE_RE, c.phones, "<phone>")
        if c.collapse_whitespace:
            text = _MULTISPACE_RE.sub(" ", text)
            text = "\n".join(line.strip() for line in text.split("\n"))
            if c.max_consecutive_newlines:
                text = re.sub(r"\n{%d,}" % (c.max_consecutive_newlines + 1),
                              "\n" * c.max_consecutive_newlines, text)
        return text.strip()


def dedup_key(text: str) -> str:
    """Aggressive canonical form used only for exact-duplicate hashing."""
    text = unicodedata.normalize("NFC", text).lower()
    # \p{M} keeps Tamil vowel signs / virama, which plain \w would strip.
    return _NON_WORD_RE.sub("", text)
