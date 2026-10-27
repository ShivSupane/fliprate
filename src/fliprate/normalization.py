"""Conservative normalizer: only repairs things we are sure about, and keeps the original."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from .rules import (
    DEFAULT_RULES,
    RuleSet,
    _DEVA_TO_ASCII_DIGITS,
    is_word,
    tokenize_keep,
    word_key,
)


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

_WORD_CHAR_CLASS = r"[\w\u0900-\u0963\u0966-\u097F]"
_IS_WORD_CHAR_RE = re.compile(_WORD_CHAR_CLASS, re.UNICODE)


def _match_case(old: str, new: str) -> str:
    """Best-effort case preservation: Title / Upper / lower."""
    if len(old) > 1 and old.isupper():
        return new.upper()
    if old[:1].isupper():
        return new.capitalize()
    return new


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


def mask_protected_terms(
    text: str,
    protected_terms: Optional[Sequence[str]] = None,
    case_sensitive: bool = False,
) -> Tuple[str, List[str]]:
    """Mask protected words/phrases before normalization or perturbation.

    Uses deterministic, boundary-safe matching (longest phrases first) so
    multiword entities ('Tata Motors') match before sub-terms ('Tata').
    Returns (masked_text, captured_spans).
    """
    if not protected_terms:
        return text, []

    # Clean, deduplicate and sort descending by length, then alphabetical for determinism
    clean_terms = sorted(
        set(t.strip() for t in protected_terms if t and t.strip()),
        key=lambda s: (-len(s), s),
    )
    if not clean_terms:
        return text, []

    masks: List[str] = []
    patterns: List[str] = []
    for t in clean_terms:
        nfc_t = unicodedata.normalize("NFC", t)
        pre = rf"(?<!{_WORD_CHAR_CLASS})" if _IS_WORD_CHAR_RE.match(nfc_t[0]) else ""
        post = rf"(?!{_WORD_CHAR_CLASS})" if _IS_WORD_CHAR_RE.match(nfc_t[-1]) else ""
        patterns.append(rf"{pre}{re.escape(nfc_t)}{post}")

    flags = re.UNICODE if case_sensitive else (re.UNICODE | re.IGNORECASE)
    comb_re = re.compile("|".join(patterns), flags)

    def _repl(m: re.Match) -> str:
        idx = len(masks)
        masks.append(m.group(0))
        return f"__FLIPPROT{idx}__"

    nfc_text = unicodedata.normalize("NFC", text)
    masked_text = comb_re.sub(_repl, nfc_text)
    return masked_text, masks


_PROT_UNMASK_RE = re.compile(r"__FLIPPROT([0-9\u0966-\u096F]+)__", re.IGNORECASE | re.UNICODE)


def unmask_protected_terms(text: str, masks: Sequence[str]) -> str:
    """Restore masked protected terms byte-for-byte in their original form."""
    if not masks:
        return text

    def _repl(m: re.Match) -> str:
        raw_idx = m.group(1).translate(_DEVA_TO_ASCII_DIGITS)
        idx = int(raw_idx)
        return masks[idx] if idx < len(masks) else m.group(0)

    return _PROT_UNMASK_RE.sub(_repl, text)


def normalize(
    text: str,
    lowercase: bool = True,
    rules: Optional[RuleSet] = None,
    protected_terms: Optional[Sequence[str]] = None,
) -> NormalizeResult:
    """Return a cleaned copy of `text`. The original is always preserved in the result.

    What it does:
      * Unicode NFC normalization and strips stray ZWJ/ZWNJ/zero-width spaces
      * known words and their safe spellings -> one canonical spelling
        ("nhi" -> "nahi", "avadlaaa" -> "avadla", "bohot" -> "bahut", "जरा" -> "ज़रा")
      * unknown words: consonant runs of 3+ collapsed to 1, vowel runs of 3+ collapsed to 2
      * repeated punctuation collapsed ("!!!" -> "!", "।।।" -> "।")
      * extra spaces removed
      * URLs, emails, @handles, #hashtags, and numbers are protected byte-for-byte
      * protected terms (e.g. brand/product names, account codes) are left unchanged byte-for-byte

    What it never does: add, remove or reorder words, or touch ambiguous tokens like "nai",
    or tokens containing digits.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    # 1. Mask URLs, emails, @handles, #hashtags, and numbers so normalization does not damage them
    masks: List[str] = []

    def _mask_match(match: re.Match) -> str:
        idx = len(masks)
        masks.append(match.group(0))
        return f"__FLIPMASK{idx}__"

    masked_text = _PROTECT_RE.sub(_mask_match, text)

    # 2. Mask user-specified protected terms byte-for-byte
    masked_text, prot_masks = mask_protected_terms(masked_text, protected_terms)

    # Strip zero-width characters and normalize to NFC
    stripped = _ZERO_WIDTH_RE.sub("", masked_text)
    nfc_text = unicodedata.normalize("NFC", stripped)

    rules = rules or DEFAULT_RULES
    parts = tokenize_keep(nfc_text)
    changes: List[Tuple[str, str]] = []

    for i, chunk in enumerate(parts):
        # Skip our mask tokens and protected tokens
        if chunk.startswith("__FLIPMASK") or chunk.startswith("__FLIPPROT"):
            continue

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
            word = chunk if not lowercase else chunk.lower()
            canon = rules.key_to_canon.get(word_key(word))
            if canon is not None:
                if lowercase:
                    new = canon
                else:
                    new = _match_case(chunk, canon)
            else:
                # Unknown Latin word: collapse character runs conservatively
                collapsed = _collapse_runs(word)
                if lowercase:
                    new = collapsed
                else:
                    new = _match_case(chunk, collapsed)

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

    # Restore protected terms byte-for-byte
    normalized = unmask_protected_terms(normalized, prot_masks)

    # Restore masked entities byte-for-byte
    for idx, original_entity in enumerate(masks):
        normalized = normalized.replace(f"__FLIPMASK{idx}__", original_entity)

    return NormalizeResult(original=text, normalized=normalized, changes=changes)
