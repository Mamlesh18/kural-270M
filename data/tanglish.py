"""Synthetic spoken-Tamil and Tanglish generation.

Real Tanglish (romanized, code-switched Tamil) and spoken-register Tamil are
scarce in public corpora. Two cheap, clearly-labelled augmentations help the
tokenizer and model see these forms:

* :func:`colloquialize` rewrites common formal (செந்தமிழ்) verb/pronoun endings to
  their spoken (பேச்சுத் தமிழ்) forms, e.g. போகிறேன் → போறேன், அவர்கள் → அவங்க.
* :class:`Transliterator` produces colloquial romanization as people actually
  type it (not ISO-15919): contextual voicing (க → k/g, த → th/dh, ற்ற → tr, ன்ற → ndr)
  and optional random spelling variation (aa → a, zh → l, ...).

Both are heuristics. Synthetic documents are written to their own categories
(``tanglish_synthetic``, ``ta_spoken_synthetic``) so the mixture can cap them.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Any, Mapping

import regex

from common.tamil import TAMIL_DIGITS, VIRAMA, is_tamil_char

# --------------------------------------------------------------------------- colloquial

_END = r"(?![\p{L}\p{M}])"  # end of a Tamil word (vowel signs are \p{M})

# Ordered: specific words, then geminate verb endings, then generic endings.
_COLLOQUIAL_RULES: list[tuple[str, str]] = [
    # irregular / high-frequency verbs
    ("இருக்கிறது", "இருக்கு"), ("இருக்கின்றது", "இருக்கு"), ("இருக்கிறேன்", "இருக்கேன்"),
    ("இருக்கிறாய்", "இருக்கே"), ("இருக்கிறீர்கள்", "இருக்கீங்க"), ("இருக்கிறார்கள்", "இருக்காங்க"),
    ("இருக்கின்றனர்", "இருக்காங்க"), ("இருக்கிறார்", "இருக்காரு"), ("இருந்தது", "இருந்துச்சு"),
    ("வருகிறேன்", "வரேன்"), ("வருகிறது", "வருது"), ("வருகிறார்கள்", "வராங்க"), ("வருகிறார்", "வராரு"),
    ("வருகிறான்", "வரான்"), ("வருகிறாள்", "வரா"),
    # present tense, geminate stems: படிக்கிறேன் → படிக்குறேன்
    ("க்கிறேன்", "க்குறேன்"), ("க்கிறோம்", "க்குறோம்"), ("க்கிறாய்", "க்குற"),
    ("க்கிறீர்கள்", "க்குறீங்க"), ("க்கிறார்கள்", "க்குறாங்க"), ("க்கின்றனர்", "க்குறாங்க"),
    ("க்கிறார்", "க்குறாரு"), ("க்கிறது", "க்குது"), ("க்கின்றது", "க்குது"), ("க்கிறான்", "க்குறான்"),
    ("க்கிறாள்", "க்குறா"),
    # present tense, other stems: போகிறேன் → போறேன், சொல்கிறேன் → சொல்றேன்
    ("கிறேன்", "றேன்"), ("கின்றேன்", "றேன்"), ("கிறோம்", "றோம்"), ("கிறாய்", "றே"),
    ("கிறீர்கள்", "றீங்க"), ("கிறார்கள்", "றாங்க"), ("கின்றனர்", "றாங்க"), ("கிறார்", "றாரு"),
    ("கிறது", "குது"), ("கின்றது", "குது"), ("கிறான்", "றான்"), ("கிறாள்", "றா"),
    # negation
    ("வில்லை", "ல"),
    # pronouns / plurals
    ("அவர்கள்", "அவங்க"), ("இவர்கள்", "இவங்க"), ("நீங்கள்", "நீங்க"), ("நாங்கள்", "நாங்க"),
    ("உங்கள்", "உங்க"), ("எங்கள்", "எங்க"), ("ர்கள்", "ங்க"),
    # adverbs / particles
    ("இல்லை", "இல்ல"), ("வேண்டும்", "வேணும்"), ("எப்படி", "எப்டி"), ("இப்போது", "இப்போ"),
    ("அப்போது", "அப்போ"), ("எப்போது", "எப்போ"), ("இங்கே", "இங்க"), ("அங்கே", "அங்க"),
    ("எங்கே", "எங்க"), ("ஆமாம்", "ஆமா"),
]
_COLLOQUIAL_RE = [(regex.compile(src + _END), dst) for src, dst in _COLLOQUIAL_RULES]


def colloquialize(text: str) -> str:
    for pattern, repl in _COLLOQUIAL_RE:
        text = pattern.sub(repl, text)
    return text


# --------------------------------------------------------------------------- transliteration

_VOWELS = {"அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ee", "உ": "u", "ஊ": "oo", "எ": "e",
           "ஏ": "e", "ஐ": "ai", "ஒ": "o", "ஓ": "o", "ஔ": "au", "ஃ": "h"}
_SIGNS = {"ா": "aa", "ி": "i", "ீ": "ee", "ு": "u", "ூ": "oo", "ெ": "e", "ே": "e",
          "ை": "ai", "ொ": "o", "ோ": "o", "ௌ": "au"}
_BASIC = {"ங": "ng", "ஞ": "nj", "ண": "n", "ந": "n", "ம": "m", "ய": "y", "ர": "r", "ல": "l",
          "வ": "v", "ழ": "zh", "ள": "l", "ன": "n", "ஜ": "j", "ஷ": "sh", "ஸ": "s", "ஹ": "h",
          "ஶ": "sh"}
_DIGITS = {d: str(i) for i, d in enumerate(TAMIL_DIGITS)}

# Contextual consonants: (initial, geminate, after-nasal, intervocalic, after-other-consonant)
_CONTEXTUAL = {
    "க": ("k", "kk", "g", "g", "k"),
    "ச": ("s", "ch", "j", "s", "ch"),
    "ட": ("t", "tt", "d", "d", "t"),
    "த": ("th", "th", "dh", "dh", "th"),
    "ப": ("p", "pp", "b", "p", "p"),
    "ற": ("r", "tr", "dr", "r", "r"),
}
_NASALS = {"ங", "ஞ", "ண", "ந", "ம", "ன"}

# Spelling variants applied with probability ``variant_prob`` per occurrence.
_VARIANTS = [("aa", "a"), ("ee", "i"), ("oo", "u"), ("zh", "l"), ("dh", "th"), ("th", "t")]


@dataclass
class _Unit:
    kind: str           # "C" consonant, "V" independent vowel, "O" other
    base: str
    sign: str | None = None   # vowel sign, VIRAMA, or None (inherent a)


def _parse(word: str) -> list[_Unit]:
    units: list[_Unit] = []
    for ch in word:
        if ch in _CONTEXTUAL or ch in _BASIC:
            units.append(_Unit("C", ch))
        elif ch in _VOWELS:
            units.append(_Unit("V", ch))
        elif (ch in _SIGNS or ch == VIRAMA) and units and units[-1].kind == "C" and units[-1].sign is None:
            units[-1].sign = ch
        else:
            units.append(_Unit("O", ch))
    return units


class Transliterator:
    def __init__(self, variant_prob: float = 0.0, seed: int | None = None):
        self.variant_prob = variant_prob
        self.rng = random.Random(seed)

    def _consonant(self, units: list[_Unit], i: int) -> str:
        u = units[i]
        if u.base not in _CONTEXTUAL:
            return _BASIC[u.base]
        initial, geminate, after_nasal, intervocalic, after_cons = _CONTEXTUAL[u.base]
        prev = units[i - 1] if i > 0 else None
        if prev is None or prev.kind == "O":
            return initial
        if prev.kind == "C" and prev.sign == VIRAMA:
            if prev.base == u.base:
                return geminate
            if prev.base in _NASALS:
                return after_nasal
            return after_cons
        return intervocalic

    def word(self, word: str) -> str:
        if word == "ஸ்ரீ":
            return "sri"
        units = _parse(word)
        out: list[str] = []
        for i, u in enumerate(units):
            if u.kind == "V":
                out.append(_VOWELS[u.base])
            elif u.kind == "O":
                out.append(_DIGITS.get(u.base, u.base))
            else:
                nxt = units[i + 1] if i + 1 < len(units) else None
                # Geminate clusters are written once at the second consonant (ச்ச → "ch", ற்ற → "tr").
                if (u.sign == VIRAMA and nxt is not None and nxt.kind == "C" and nxt.base == u.base
                        and u.base in _CONTEXTUAL):
                    continue
                if u.base in ("ங", "ஞ") and u.sign == VIRAMA and nxt is not None and nxt.kind == "C":
                    out.append("n")  # ங்க → "ng", ஞ்ச → "nj": the following stop supplies g/j
                    continue
                out.append(self._consonant(units, i))
                if u.sign is None:
                    out.append("a")
                elif u.sign != VIRAMA:
                    out.append(_SIGNS[u.sign])
        s = "".join(out)
        if self.variant_prob > 0:
            for src, dst in _VARIANTS:
                if src in s and self.rng.random() < self.variant_prob:
                    s = s.replace(src, dst)
        return s

    def __call__(self, text: str) -> str:
        # Split into Tamil and non-Tamil runs; transliterate only Tamil runs.
        return re.sub(r"[஀-௿]+", lambda m: self.word(m.group(0)), text)


# --------------------------------------------------------------------------- augmentation

@dataclass
class AugmentConfig:
    enabled: bool = False
    tanglish_fraction: float = 0.0      # share of Tamil docs that spawn a synthetic Tanglish copy
    spoken_fraction: float = 0.0        # share of Tamil docs that spawn a synthetic spoken-Tamil copy
    colloquialize_before_translit: bool = True
    variant_prob: float = 0.15
    sentence_mix_prob: float = 0.0      # >0: per-sentence mix of Tamil script and Tanglish
    max_chars: int = 4000               # truncate long docs before augmenting
    tanglish_category: str = "tanglish_synthetic"
    spoken_category: str = "ta_spoken_synthetic"
    seed: int = 7

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "AugmentConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+")


class Augmenter:
    def __init__(self, config: AugmentConfig | Mapping[str, Any] | None = None):
        if not isinstance(config, AugmentConfig):
            config = AugmentConfig.from_dict(config)
        self.cfg = config
        self.rng = random.Random(config.seed)
        self.translit = Transliterator(config.variant_prob, seed=config.seed + 1)

    def _truncate(self, text: str) -> str:
        if len(text) <= self.cfg.max_chars:
            return text
        cut = text.rfind("\n", 0, self.cfg.max_chars)
        return text[: cut if cut > self.cfg.max_chars // 2 else self.cfg.max_chars]

    def to_tanglish(self, text: str) -> str:
        c = self.cfg
        if c.colloquialize_before_translit:
            text = colloquialize(text)
        if c.sentence_mix_prob > 0:
            parts = _SENT_SPLIT.split(text)
            return " ".join(self.translit(p) if self.rng.random() < c.sentence_mix_prob else p for p in parts)
        return self.translit(text)

    def __call__(self, text: str) -> list[tuple[str, str]]:
        """Returns a list of ``(category, synthetic_text)`` for one Tamil document."""
        c = self.cfg
        if not c.enabled:
            return []
        out: list[tuple[str, str]] = []
        if c.tanglish_fraction and self.rng.random() < c.tanglish_fraction:
            out.append((c.tanglish_category, self.to_tanglish(self._truncate(text))))
        if c.spoken_fraction and self.rng.random() < c.spoken_fraction:
            spoken = colloquialize(self._truncate(text))
            if spoken != text:
                out.append((c.spoken_category, spoken))
        return out


def has_tamil(text: str) -> bool:
    return any(is_tamil_char(ch) for ch in text)
