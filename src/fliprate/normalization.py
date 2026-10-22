"""Conservative normalizer: only repairs things we are sure about, and keeps the original."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .rules import DEFAULT_RULES, RuleSet, is_word, tokenize_keep, word_key


@dataclass
class NormalizeResult:
    original: str
    normalized: str
    changes: List[Tuple[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        return self.normalized


# Repeated letter patterns:
#  - 3+ identical consonants  -> 1  (e.g. "avadlaaa" -> "avadla", "soooo" -> "so"... but vowels are
#    more nuanced — we shorten to 2 first, then the rules dict may further canonicalize)
#  - 3+ identical vowels      -> 2  (preserve possible emphasis like "soo" intentionally written,
#    while collapsing "soooo" -> "so" is too aggressive for unknown words)
_CONSONANT_RUN_RE = re.compile(r"([^aeiouAEIOU\W_])\1{2,}")   # 3+ identical consonants
_VOWEL_RUN_RE     = re.compile(r"([aeiouAEIOU])\1{2,}")        # 3+ identical vowels -> 2
_PUNC_RUN_RE      = re.compile(r"([^\w\s])\1+")                # repeated punctuation
_WS_RE            = re.compile(r"\s+")
_ZERO_WIDTH_RE    = re.compile(r"[\u200B\u200C\u200D\uFEFF]")  # ZWSP, ZWNJ, ZWJ, BOM


def _collapse_runs(word: str) -> str:
    """Collapse excessively repeated characters in a single word token.

    Strategy:
    - Consonant runs of 3+ → 1 copy  (e.g. "avadlaaa" has 'a' which is a vowel,
      but "sttring" -> "string").  Actually consonants run-collapse to 1.
    - Vowel runs of 3+ → 2 copies   (e.g. "sooooo" -> "soo").
      We keep 2 rather than 1 because "soo much" is common colloquial writing.
    This is applied only to words NOT found in the rules dict, so rule-covered
    words (like "nahi") go through canonical lookup instead.
    """
    out = _CONSONANT_RUN_RE.sub(r"\1", word)
    out = _VOWEL_RUN_RE.sub(r"\1\1", out)
    return out


_PROTECT_RE = re.compile(
    r"(?:https?://|www\.)\S+|"
    r"[\w.+-]+@[\w-]+\.[\w.-]+|"
    r"@\w+|"
    r"#\w+|"
    r"\b\d+(?:[.,]\d+)*\b|\d+"
)


def normalize(text: str, lowercase: bool = True, rules: Optional[RuleSet] = None) -> NormalizeResult:
    """Return a cleaned copy of `text`. The original is always preserved in the result.

    What it does:
      * Unicode NFC normalization and strips stray ZWJ/ZWNJ/zero-width spaces
      * known words and their safe spellings -> one canonical spelling
        ("nhi" -> "nahi", "avadlaaa" -> "avadla", "bohot" -> "bahut", "जरा" -> "ज़रा")
      * unknown words: consonant runs of 3+ collapsed to 1, vowel runs of 3+ collapsed to 2
      * repeated punctuation collapsed ("!!!" -> "!", "।।।" -> "।")
      * extra spaces removed
      * URLs, emails, @handles, #hashtags, and numbers are protected byte-for-byte

    What it never does: add, remove or reorder words, or touch ambiguous tokens like "nai",
    or tokens containing digits.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    # Mask URLs, emails, @handles, #hashtags, and numbers so normalization does not damage them
    masks: List[str] = []

    def _mask_match(match: re.Match) -> str:
        idx = len(masks)
        masks.append(match.group(0))
        return f"__FLIPMASK{idx}__"

    masked_text = _PROTECT_RE.sub(_mask_match, text)

    # Strip zero-width characters and normalize to NFC
    stripped = _ZERO_WIDTH_RE.sub("", masked_text)
    nfc_text = unicodedata.normalize("NFC", stripped)

    rules = rules or DEFAULT_RULES
    parts = tokenize_keep(nfc_text)
    changes: List[Tuple[str, str]] = []

    for i, chunk in enumerate(parts):
        if not is_word(chunk):
            # Collapse repeated punctuation: "!!!" -> "!", "..." -> ".", "।।।" -> "।"
            collapsed = _PUNC_RUN_RE.sub(r"\1", chunk)
            if collapsed != chunk:
                changes.append((chunk, collapsed))
            parts[i] = collapsed
            continue

        # Skip tokens with digits (including our mask tokens)
        if any(c.isdigit() for c in chunk):
            continue

        if chunk.isascii():
            word = chunk.lower() if lowercase else chunk
            # 1. Check rules dict first
            canon = rules.key_to_canon.get(word_key(word))
            if canon is not None:
                new = canon
            else:
                # 2. Unknown Latin word: collapse character runs conservatively
                new = _collapse_runs(word)

            if new.lower() != chunk.lower():
                changes.append((chunk, new))
            parts[i] = new
        else:
            # Non-ASCII (e.g. Devanagari)
            canon = rules.key_to_canon.get(word_key(chunk))
            if canon is not None and canon != chunk:
                changes.append((chunk, canon))
                parts[i] = canon

    normalized = _WS_RE.sub(" ", "".join(parts)).strip()

    # Restore masked entities byte-for-byte
    for idx, original_entity in enumerate(masks):
        normalized = normalized.replace(f"__FLIPMASK{idx}__", original_entity)

    return NormalizeResult(original=text, normalized=normalized, changes=changes)
