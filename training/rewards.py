"""Rule-based rewards for GRPO (no reward model needed).

Each reward maps ``(prompt, completion, reference)`` to a float in roughly [0, 1];
``combined`` sums them with configurable weights. They are deliberately simple and
transparent — the point of GRPO here is RL with *verifiable* signals:

* ``reference_f1``    token-overlap F1 with the reference answer (QA / factual prompts)
* ``contains_answer`` 1 if the normalized reference appears in the completion
* ``tamil_script``    share of letters in Tamil script, only when the prompt itself is Tamil
                      (teaches "answer Tamil questions in Tamil")
* ``no_repetition``   1 − share of characters inside repeated 4-grams (penalizes loops)
* ``length``          1 inside [min_words, max_words], decaying outside (penalizes empty /
                      rambling answers)
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Callable, Mapping

from common.tamil import script_stats
from data.quality import _dup_ngram_frac

_PUNCT = re.compile(r"[^\w\s஀-௿]")


def _norm_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text or "").lower()
    return _PUNCT.sub(" ", text).split()


def reference_f1(prompt: str, completion: str, reference: str | None, **_) -> float | None:
    if not reference:
        return None
    pred, gold = _norm_tokens(completion), _norm_tokens(reference)
    if not pred or not gold:
        return 0.0
    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(gold)
    return 2 * p * r / (p + r)


def contains_answer(prompt: str, completion: str, reference: str | None, **_) -> float | None:
    if not reference:
        return None
    ref = " ".join(_norm_tokens(reference))
    return float(bool(ref) and ref in " ".join(_norm_tokens(completion)))


def tamil_script(prompt: str, completion: str, reference: str | None, **_) -> float | None:
    if script_stats(prompt).tamil_ratio < 0.5:
        return None  # only for Tamil-script prompts
    return script_stats(completion).tamil_ratio


def no_repetition(prompt: str, completion: str, reference: str | None, **_) -> float:
    words = _norm_tokens(completion)
    return 1.0 - _dup_ngram_frac(words, 4) if len(words) >= 4 else 1.0


def length(prompt: str, completion: str, reference: str | None, min_words: int = 2, max_words: int = 120,
           **_) -> float:
    n = len(completion.split())
    if n < min_words:
        return n / max(min_words, 1)
    if n > max_words:
        return max(0.0, 1 - (n - max_words) / max_words)
    return 1.0


REWARDS: dict[str, Callable[..., float | None]] = {
    "reference_f1": reference_f1,
    "contains_answer": contains_answer,
    "tamil_script": tamil_script,
    "no_repetition": no_repetition,
    "length": length,
}


def combined(prompt: str, completion: str, reference: str | None, weights: Mapping[str, float],
             **kwargs) -> tuple[float, dict[str, float]]:
    """Weighted sum of the applicable rewards (``None`` = not applicable, weight ignored)."""
    parts: dict[str, float] = {}
    total = 0.0
    for name, w in weights.items():
        if not w:
            continue
        if name not in REWARDS:
            raise KeyError(f"Unknown reward {name!r}; available: {sorted(REWARDS)}")
        v = REWARDS[name](prompt, completion, reference, **kwargs)
        if v is None:
            continue
        parts[name] = v
        total += w * v
    return total, parts
