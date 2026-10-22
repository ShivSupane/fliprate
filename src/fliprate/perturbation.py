"""Generate realistic, meaning-preserving variants of a message."""
from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .rules import (
    CROSS_SCRIPT_MAP,
    CROSS_SCRIPT_REV,
    DEFAULT_RULES,
    RuleSet,
    is_word,
    tokenize_keep,
)

KINDS = ("spelling", "repetition", "punctuation", "case", "unicode", "digits")
_VOWELS = "aeiouAEIOU"

_NUKTA_DECOMPOSE = {
    "\u0958": "\u0915\u093C",  # क़ (QA)   -> क (KA)  + ़
    "\u0959": "\u0916\u093C",  # ख़ (KHHA) -> ख (KHA) + ़
    "\u095A": "\u0917\u093C",  # ग़ (GHHA) -> ग (GA)  + ़
    "\u095B": "\u091C\u093C",  # ज़ (ZA)   -> ज (JA)  + ़  [was U+0918 GHA — fixed]
    "\u095C": "\u0921\u093C",  # ड़ (DDDHA)-> ड (DDA) + ़
    "\u095D": "\u0922\u093C",  # ढ़ (RHA)  -> ढ (DDHA)+ ़
    "\u095E": "\u092B\u093C",  # फ़ (FA)   -> फ (PHA) + ़
    "\u095F": "\u092F\u093C",  # य़ (YYA)  -> य (YA)  + ़
}
_NUKTA_COMPOSE = {v: k for k, v in _NUKTA_DECOMPOSE.items()}

_ASCII_TO_DEVA_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")
_DEVA_TO_ASCII_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


@dataclass(frozen=True)
class Variant:
    text: str
    kind: str


def _match_case(old: str, new: str) -> str:
    if len(old) > 1 and old.isupper():
        return new.upper()
    if old[:1].isupper():
        return new.capitalize()
    return new


def _spelling(text: str, rng: random.Random, rules: RuleSet) -> Optional[str]:
    parts = tokenize_keep(text)
    idx = [i for i, p in enumerate(parts) if is_word(p) and rules.group_of(p) is not None]
    if not idx:
        return None
    for i in rng.sample(idx, k=rng.randint(1, len(idx))):
        members = rules.groups[rules.group_of(parts[i])]
        options = [m for m in members if m.lower() != parts[i].lower()]
        if options:
            parts[i] = _match_case(parts[i], rng.choice(options))
    cand = "".join(parts)
    return cand if cand != text else None


def _repetition(text: str, rng: random.Random, rules: RuleSet = DEFAULT_RULES) -> Optional[str]:
    parts = tokenize_keep(text)
    idx = [i for i, p in enumerate(parts) if is_word(p) and len(p) >= 3 and any(c in _VOWELS for c in p)]
    if not idx:
        return None
    i = rng.choice(idx)
    word = parts[i]
    pos = rng.choice([j for j, c in enumerate(word) if c in _VOWELS])
    parts[i] = word[: pos + 1] + word[pos] * rng.randint(2, 3) + word[pos + 1 :]
    return "".join(parts)


def _punctuation(text: str, rng: random.Random, rules: RuleSet = DEFAULT_RULES) -> Optional[str]:
    base = text.rstrip()
    has_deva = any(0x0900 <= ord(c) <= 0x097F for c in base)
    if base.endswith(("।", "॥")):
        return base.rstrip("।॥") + rng.choice([".", "!", "..."])
    if base.endswith("?"):
        return base + "?"
    if base.endswith((".", "!")):
        if has_deva and rng.random() < 0.5:
            return base.rstrip(".!") + "।"
        return base.rstrip(".!") + rng.choice(["!", "!!", "...", "."])
    if rng.random() < 0.4:
        stripped = re.sub(r"[^\w\s]", "", base)
        return stripped if stripped.strip() else None
    if has_deva:
        return base + rng.choice(["।", "!", ".", "!!"])
    return base + rng.choice(["!", "!!", "...", "."])


def _case(text: str, rng: random.Random, rules: RuleSet = DEFAULT_RULES) -> Optional[str]:
    if text.lower() == text.upper():
        # Text has no case (e.g. pure Devanagari)
        return None
    return rng.choice([text.lower(), text.upper(), text.capitalize()])


def _unicode(text: str, rng: random.Random, rules: RuleSet = DEFAULT_RULES) -> Optional[str]:
    """Precomposed vs decomposed nuktas and NFC/NFD variations."""
    has_pre = [k for k in _NUKTA_DECOMPOSE if k in text]
    has_decomp = [k for k in _NUKTA_COMPOSE if k in text]

    if has_pre:
        out = text
        for ch in has_pre:
            out = out.replace(ch, _NUKTA_DECOMPOSE[ch])
        return out if out != text else None

    if has_decomp:
        out = text
        for ch in has_decomp:
            out = out.replace(ch, _NUKTA_COMPOSE[ch])
        return out if out != text else None

    # Check NFD vs NFC
    nfd = unicodedata.normalize("NFD", text)
    if nfd != text:
        return nfd
    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        return nfc
    return None


def _digits(text: str, rng: random.Random, rules: RuleSet = DEFAULT_RULES) -> Optional[str]:
    """Swap between ASCII and Devanagari digits (०-९ <-> 0-9)."""
    has_ascii_digits = any("0" <= c <= "9" for c in text)
    has_deva_digits = any("०" <= c <= "९" for c in text)

    if has_ascii_digits and (not has_deva_digits or rng.random() < 0.5):
        # Convert ASCII digits to Devanagari
        return text.translate(_ASCII_TO_DEVA_DIGITS)
    elif has_deva_digits:
        # Convert Devanagari digits to ASCII
        return text.translate(_DEVA_TO_ASCII_DIGITS)
    return None


def _cross_script(text: str, rng: random.Random, rules: RuleSet = DEFAULT_RULES) -> Optional[str]:
    """Opt-in bilingual swap for high-frequency 1-to-1 words."""
    parts = tokenize_keep(text)
    idx = [
        i
        for i, p in enumerate(parts)
        if is_word(p) and (p.lower() in CROSS_SCRIPT_MAP or p in CROSS_SCRIPT_REV)
    ]
    if not idx:
        return None
    i = rng.choice(idx)
    word = parts[i]
    if word.lower() in CROSS_SCRIPT_MAP:
        parts[i] = CROSS_SCRIPT_MAP[word.lower()]
    elif word in CROSS_SCRIPT_REV:
        parts[i] = CROSS_SCRIPT_REV[word]
    cand = "".join(parts)
    return cand if cand != text else None


_APPLY = {
    "spelling": _spelling,
    "repetition": _repetition,
    "punctuation": _punctuation,
    "case": _case,
    "unicode": _unicode,
    "digits": _digits,
    "cross_script": _cross_script,
}


def perturb(
    text: str,
    n: int = 10,
    kinds: Optional[Sequence[str]] = None,
    seed: int = 0,
    rules: Optional[RuleSet] = None,
    cross_script: bool = False,
) -> List[Variant]:
    """Return up to `n` distinct safe variants of `text` (deterministic for a given seed).

    Every variant is checked with `is_safe_variant`, so negations are never added or
    removed. Fewer than `n` variants may be returned for short or already-noisy text.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rules = rules or DEFAULT_RULES

    if kinds is not None:
        active_kinds = tuple(kinds)
    else:
        has_deva = any(0x0900 <= ord(c) <= 0x097F for c in text)
        has_latin = any(
            (0x0041 <= ord(c) <= 0x005A) or (0x0061 <= ord(c) <= 0x007A)
            for c in text
        )
        if has_deva and not has_latin:
            # Pure Devanagari
            active_kinds = ("unicode", "spelling", "punctuation", "digits")
        elif has_deva and has_latin:
            # Mixed
            active_kinds = ("unicode", "spelling", "repetition", "punctuation", "case", "digits")
        else:
            # Latin
            active_kinds = ("spelling", "repetition", "punctuation", "case", "digits")

    if cross_script and "cross_script" not in active_kinds:
        active_kinds = active_kinds + ("cross_script",)

    unknown = [k for k in active_kinds if k not in _APPLY]
    if unknown:
        raise ValueError(f"Unknown perturbation kinds: {unknown}. Choose from {list(_APPLY.keys())}")
    if not text.strip():
        return []
    rng = random.Random(seed)
    seen = {text}
    out: List[Variant] = []
    for attempt in range(n * 15):
        kind = active_kinds[attempt % len(active_kinds)]
        candidate = _APPLY[kind](text, rng, rules)
        if candidate and candidate not in seen and rules.is_safe_variant(text, candidate, cross_script=cross_script):
            seen.add(candidate)
            out.append(Variant(candidate, kind))
            if len(out) >= n:
                break
    return out
