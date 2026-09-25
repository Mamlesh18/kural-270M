"""Tamil-aware language identification.

Off-the-shelf LID models do not distinguish the varieties we care about, so we
use script statistics plus a romanized-Tamil lexicon:

* ``ta``          – Tamil script dominant
* ``code_mixed``  – substantial Tamil script *and* Latin script (e.g. "இந்த movie semma")
* ``tanglish``    – Latin script, but lexically Tamil ("naan office ku late ah varen")
* ``en``          – Latin script, English function words present
* ``other``       – anything else (other scripts, too little text, other Latin languages)

An optional fastText ``lid.176.bin`` model can be supplied to reject non-English
Latin-script text that is neither English nor Tanglish.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from common.tamil import script_stats

RESOURCES = Path(__file__).resolve().parent / "resources"
_LATIN_WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
# Agglutinative Tamil suffixes as typically romanized. Longer word required to avoid noise.
_TANGLISH_SUFFIX_RE = re.compile(
    r"(?:kku|ukku|nga|ngala|ngalaa|uthu|udhu|uchu|ichu|rukku|ruku|kitta|thaan|dhaan|laam|"
    r"nom|raanga|raaru|duchu|ttom|ttaan|ttaanga|irukku|iruku|lanu|nnu)$"
)


def load_wordlist(name_or_path: str | os.PathLike) -> frozenset[str]:
    path = Path(name_or_path)
    if not path.is_absolute() and not path.exists():
        path = RESOURCES / path
    words = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            words.add(line.lower())
    return frozenset(words)


@dataclass
class LangResult:
    label: str
    tamil_ratio: float
    latin_ratio: float
    tanglish_score: float = 0.0
    english_score: float = 0.0
    letters: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "lang": self.label,
            "tamil_ratio": round(self.tamil_ratio, 4),
            "latin_ratio": round(self.latin_ratio, 4),
            "tanglish_score": round(self.tanglish_score, 4),
            "english_score": round(self.english_score, 4),
        }


@dataclass
class LangIdConfig:
    min_letters: int = 20
    tamil_min_ratio: float = 0.60        # ≥ this share of letters in Tamil script → ta / code_mixed
    code_mixed_min_latin: float = 0.15   # Tamil-dominant text with ≥ this Latin share → code_mixed
    code_mixed_min_tamil: float = 0.15   # Latin-dominant text with ≥ this Tamil share → code_mixed
    latin_min_ratio: float = 0.80
    tanglish_min_score: float = 0.12
    english_min_score: float = 0.08
    suffix_weight: float = 0.5
    lexicon: str = "tanglish_lexicon.txt"
    english_stopwords: str = "english_stopwords.txt"
    fasttext_model: str | None = None    # path to lid.176.bin; None disables
    fasttext_min_en_prob: float = 0.5

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "LangIdConfig":
        d = dict(d or {})
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@lru_cache(maxsize=4)
def _load_fasttext(path: str):
    import fasttext  # optional dependency

    return fasttext.load_model(path)


class LanguageIdentifier:
    def __init__(self, config: LangIdConfig | Mapping[str, Any] | None = None):
        if not isinstance(config, LangIdConfig):
            config = LangIdConfig.from_dict(config)
        self.cfg = config
        self.lexicon = load_wordlist(config.lexicon)
        self.en_stop = load_wordlist(config.english_stopwords)
        self._ft = None
        if config.fasttext_model:
            if Path(config.fasttext_model).exists():
                self._ft = _load_fasttext(str(config.fasttext_model))
            else:
                raise FileNotFoundError(f"fastText LID model not found: {config.fasttext_model}")

    def latin_scores(self, text: str) -> tuple[float, float]:
        """(tanglish_score, english_score) over Latin-script words."""
        words = [w.lower() for w in _LATIN_WORD_RE.findall(text)]
        if not words:
            return 0.0, 0.0
        ta = en = 0.0
        for w in words:
            if w in self.lexicon:
                ta += 1.0
            elif len(w) >= 5 and w not in self.en_stop and _TANGLISH_SUFFIX_RE.search(w):
                ta += self.cfg.suffix_weight
            if w in self.en_stop:
                en += 1.0
        return ta / len(words), en / len(words)

    def _fasttext_is_english(self, text: str) -> bool:
        if self._ft is None:
            return True
        labels, probs = self._ft.predict(text.replace("\n", " ")[:2000], k=1)
        return labels[0] == "__label__en" and probs[0] >= self.cfg.fasttext_min_en_prob

    def __call__(self, text: str) -> LangResult:
        c = self.cfg
        st = script_stats(text)
        tr, lr = st.tamil_ratio, st.latin_ratio
        if st.letters < c.min_letters:
            return LangResult("other", tr, lr, letters=st.letters)

        if tr >= c.tamil_min_ratio:
            label = "code_mixed" if lr >= c.code_mixed_min_latin else "ta"
            return LangResult(label, tr, lr, letters=st.letters)

        ta_score, en_score = self.latin_scores(text)
        if tr >= c.code_mixed_min_tamil and lr >= c.code_mixed_min_latin:
            return LangResult("code_mixed", tr, lr, ta_score, en_score, st.letters)

        if lr >= c.latin_min_ratio:
            # Tanglish wins over English when romanized-Tamil words are frequent, even if
            # English words are present too (Tanglish is usually code-switched).
            if ta_score >= c.tanglish_min_score and ta_score >= 0.5 * en_score:
                return LangResult("tanglish", tr, lr, ta_score, en_score, st.letters)
            if en_score >= c.english_min_score and self._fasttext_is_english(text):
                return LangResult("en", tr, lr, ta_score, en_score, st.letters)
        return LangResult("other", tr, lr, ta_score, en_score, st.letters)
