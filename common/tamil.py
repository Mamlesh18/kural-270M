"""Tamil script constants and character-class helpers (Unicode block U+0B80–U+0BFF)."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

TAMIL_START, TAMIL_END = 0x0B80, 0x0BFF

VIRAMA = "்"  # ் pulli
AYTHAM = "ஃ"  # ஃ
TAMIL_DIGITS = "௦௧௨௩௪௫௬௭௮௯"

VOWELS = "அஆஇஈஉஊஎஏஐஒஓஔ"
CONSONANTS = "கஙசஞடணதநபமயரலவழளறனஜஶஷஸஹ"
VOWEL_SIGNS = "ாிீுூெேைொோௌ"


def is_tamil_char(ch: str) -> bool:
    return TAMIL_START <= ord(ch) <= TAMIL_END


def is_latin_letter(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z") or (
        "À" <= ch <= "ɏ" and unicodedata.category(ch).startswith("L"))


def is_tamil_piece(piece: str, allow: str = "▁") -> bool:
    """True if a tokenizer piece is non-empty Tamil (optionally with the SentencePiece space marker)."""
    core = [c for c in piece if c not in allow]
    return bool(core) and all(is_tamil_char(c) for c in core)


@dataclass
class ScriptStats:
    tamil: int = 0
    latin: int = 0
    other_letters: int = 0
    digits: int = 0
    spaces: int = 0
    punct: int = 0
    total: int = 0

    @property
    def letters(self) -> int:
        return self.tamil + self.latin + self.other_letters

    @property
    def tamil_ratio(self) -> float:
        return self.tamil / self.letters if self.letters else 0.0

    @property
    def latin_ratio(self) -> float:
        return self.latin / self.letters if self.letters else 0.0

    @property
    def other_ratio(self) -> float:
        return self.other_letters / self.letters if self.letters else 0.0


def script_stats(text: str) -> ScriptStats:
    s = ScriptStats(total=len(text))
    for ch in text:
        if is_tamil_char(ch):
            if ch in TAMIL_DIGITS:
                s.digits += 1
            else:
                s.tamil += 1  # letters, vowel signs (Mn/Mc) and virama all count as Tamil
        elif ch.isspace():
            s.spaces += 1
        elif ch.isdigit():
            s.digits += 1
        elif is_latin_letter(ch):
            s.latin += 1
        else:
            cat = unicodedata.category(ch)
            if cat.startswith("L") or cat.startswith("M"):
                s.other_letters += 1
            elif cat.startswith("P") or cat.startswith("S"):
                s.punct += 1
    return s


def count_graphemes(text: str) -> int:
    """Approximate Tamil grapheme (akshara) count: combining marks attach to the previous base."""
    n = 0
    for ch in text:
        if unicodedata.category(ch) in ("Mn", "Mc"):
            continue
        n += 1
    return n
