"""Generate realistic, meaning-preserving variants of a message."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .rules import DEFAULT_RULES, RuleSet, is_word, tokenize_keep

KINDS = ("spelling", "repetition", "punctuation", "case")
_VOWELS = "aeiouAEIOU"


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
        options = [m for m in members if m != parts[i].lower()]
        if options:
            parts[i] = _match_case(parts[i], rng.choice(options))
    return "".join(parts)


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
    if rng.random() < 0.5:
        stripped = re.sub(r"[^\w\s]", "", base)
        return stripped if stripped.strip() else None
    if base.endswith("?"):
        return base + "?"
    return base.rstrip(".!") + rng.choice(["!", "!!", "...", "."])


def _case(text: str, rng: random.Random, rules: RuleSet = DEFAULT_RULES) -> Optional[str]:
    return rng.choice([text.lower(), text.upper(), text.capitalize()])


_APPLY = {"spelling": _spelling, "repetition": _repetition, "punctuation": _punctuation, "case": _case}


def perturb(
    text: str,
    n: int = 10,
    kinds: Optional[Sequence[str]] = None,
    seed: int = 0,
    rules: Optional[RuleSet] = None,
) -> List[Variant]:
    """Return up to `n` distinct safe variants of `text` (deterministic for a given seed).

    Every variant is checked with `is_safe_variant`, so negations are never added or
    removed. Fewer than `n` variants may be returned for short or already-noisy text.
    """
    rules = rules or DEFAULT_RULES
    kinds = tuple(kinds) if kinds else KINDS
    unknown = [k for k in kinds if k not in _APPLY]
    if unknown:
        raise ValueError(f"Unknown perturbation kinds: {unknown}. Choose from {KINDS}")
    if not text or not text.strip() or n <= 0:
        return []
    rng = random.Random(seed)
    seen = {text}
    out: List[Variant] = []
    for attempt in range(n * 10):
        kind = kinds[attempt % len(kinds)]
        candidate = _APPLY[kind](text, rng, rules)
        if candidate and candidate not in seen and rules.is_safe_variant(text, candidate):
            seen.add(candidate)
            out.append(Variant(candidate, kind))
            if len(out) >= n:
                break
    return out
