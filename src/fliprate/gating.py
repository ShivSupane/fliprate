import asyncio
import json
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence

from .events import Event, drain, make_event
from .models import Predictor
from .normalization import normalize
from .perturbation import perturb
from .rules import DEFAULT_RULES, RuleSet

PASS, TRANSFORM, REVIEW = "PASS", "TRANSFORM", "REVIEW"


@dataclass
class GateResult:
    decision: str
    reason: str
    original_text: str
    text_used: str
    prediction: Any
    raw_stability: Optional[float]
    normalized_stability: Optional[float]
    n_variants: int
    warnings: List[str] = field(default_factory=list)
    flipped_variants: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "prediction": self.prediction,
            "reason": self.reason,
            "original_text": self.original_text,
            "text_used": self.text_used,
            "raw_stability": self.raw_stability,
            "normalized_stability": self.normalized_stability,
            "n_variants": self.n_variants,
            "warnings": list(self.warnings),
            "flipped_variants": list(self.flipped_variants),
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _is_emoji_or_modifier(c: str) -> bool:
    cp = ord(c)
    # Emoji, pictographs, symbols, transport, skin tones, variation selectors, ZWJ
    if 0x1F000 <= cp <= 0x1FAFF:
        return True
    if 0x2600 <= cp <= 0x27BF:
        return True
    if 0x2300 <= cp <= 0x23FF:
        return True
    if cp in (0x200D, 0xFE0E, 0xFE0F):
        return True
    if 0xE0020 <= cp <= 0xE007F:
        return True
    cat = unicodedata.category(c)
    if cat in ("So", "Sk") and ("EMOJI" in unicodedata.name(c, "") or "FACE" in unicodedata.name(c, "") or "HEART" in unicodedata.name(c, "")):
        return True
    return False


def _latin_ratio(text: str) -> float:
    # Ignore emoji, variation selectors, ZWJ, and skin tones when evaluating script
    letters = [c for c in text if c.isalpha() and not _is_emoji_or_modifier(c)]
    if not letters:
        # Check if there are non-Latin script letters (like Devanagari or Cyrillic)
        non_latin = [c for c in text if c.isalpha() and not unicodedata.name(c, "").startswith("LATIN") and not _is_emoji_or_modifier(c)]
        if not non_latin:
            return 1.0  # text has no non-Latin alphabet letters (e.g. emoji or numbers)
        return 0.0
    return sum(unicodedata.name(c, "").startswith("LATIN") for c in letters) / len(letters)


def iter_gate(
    model: Callable[[str], Any],
    text: str,
    n_variants: int = 8,
    threshold: float = 0.8,
    min_variants: int = 3,
    seed: int = 0,
    kinds: Optional[Sequence[str]] = None,
    rules: Optional[RuleSet] = None,
    cache: bool = True,
    on_error: str = "raise",
) -> Generator[Event, None, GateResult]:
    """Decide whether `text` can go to `model` as is (PASS), after cleaning (TRANSFORM), or needs a human (REVIEW).

    "Stability" = share of safe variants for which the model gives the same answer as for the original.
    It measures consistency, NOT correctness: a model can be stably wrong.

    Logic:
      * stability 100% on the raw text               -> PASS
      * some variants flip:
          - normalization is safe, and the normalized pipeline is stabler and >= threshold -> TRANSFORM
          - otherwise, raw stability >= threshold    -> PASS, with a warning listing the flips
          - otherwise                                -> REVIEW
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    if not (0 < threshold <= 1.0):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if n_variants < 1:
        raise ValueError(f"n_variants must be >= 1, got {n_variants}")
    if min_variants < 1:
        raise ValueError(f"min_variants must be >= 1, got {min_variants}")
    if on_error not in ("raise", "review"):
        raise ValueError(f"on_error must be 'raise' or 'review', got {on_error!r}")

    rules = rules or DEFAULT_RULES
    predictor = Predictor(model, cache=cache)
    warnings: List[str] = []
    yield make_event("received", chars=len(text))

    def done(decision, reason, used, pred, raw, norm, n, flipped=()):
        return GateResult(decision, reason, text, used, pred, raw, norm, n, warnings, list(flipped))

    def finish(res: GateResult):
        return [make_event("gate", decision=res.decision, reason=res.reason),
                make_event("done", result=res.to_dict())]

    if not text.strip():
        res = done(REVIEW, "Empty input", text, None, None, None, 0)
        yield from finish(res)
        return res

    ratio = _latin_ratio(text)
    yield make_event("script_check", latin_ratio=round(ratio, 2))
    if ratio < 0.5:
        res = done(REVIEW, "Mostly non-Latin script; v0.1 supports romanized text only", text, None, None, None, 0)
        yield from finish(res)
        return res

    for tok in rules.ambiguous_tokens(text):
        warnings.append(f"'{tok}' is ambiguous ({rules.ambiguous[tok]}); left unchanged")

    variants = perturb(text, n_variants, kinds, seed, rules=rules)
    try:
        preds = predictor.predict([text] + [v.text for v in variants])
    except Exception as exc:
        if on_error == "review":
            res = done(REVIEW, f"model error: {type(exc).__name__}: {exc}", text, None, None, None, len(variants))
            yield from finish(res)
            return res
        raise

    base, vpreds = preds[0], preds[1:]
    yield make_event("variants", count=len(variants), prediction=base)
    if len(variants) < min_variants:
        warnings.append(f"Only {len(variants)} safe variants could be generated; stability is weakly measured")

    flipped = []
    for v, p in zip(variants, vpreds):
        same = p == base
        if not same:
            flipped.append(v.text)
        yield make_event("prediction", variant=v.text, kind=v.kind, label=p, same=same)
    raw = (len(variants) - len(flipped)) / len(variants) if variants else None
    yield make_event("raw_stability", value=raw)

    if raw is None:
        warnings.append("No variants available, so stability could not be measured")
        res = done(PASS, "No variants available to test stability", text, base, None, None, 0)
        yield from finish(res)
        return res
    if not flipped:
        res = done(PASS, "Prediction is stable across safe variants", text, base, raw, None, len(variants))
        yield from finish(res)
        return res

    # Some variants flip. Would approved normalization in front of the model help?
    norm = normalize(text, rules=rules)
    nstab: Optional[float] = None
    nbase = None
    can_transform = False
    if norm.normalized != text and rules.is_safe_variant(text, norm.normalized):
        yield make_event("normalized", text=norm.normalized, changes=norm.changes)
        try:
            npreds = predictor.predict([norm.normalized] + [normalize(v.text, rules=rules).normalized for v in variants])
        except Exception as exc:
            if on_error == "review":
                res = done(REVIEW, f"model error: {type(exc).__name__}: {exc}", norm.normalized, None, raw, None, len(variants))
                yield from finish(res)
                return res
            raise
        nbase = npreds[0]
        nstab = sum(p == nbase for p in npreds[1:]) / len(variants)
        yield make_event("normalized_stability", value=nstab)
        can_transform = nstab >= threshold and nstab > raw
    elif norm.normalized != text:
        warnings.append("Normalization would change the meaning; it was not applied")

    n = len(variants)
    if can_transform:
        res = done(TRANSFORM, "Stable after approved normalization", norm.normalized, nbase, raw, nstab, n, flipped)
    elif raw >= threshold:
        warnings.append(f"{len(flipped)} of {n} safe variants changed the answer; normalization did not improve this")
        res = done(PASS, "Mostly stable across safe variants", text, base, raw, nstab, n, flipped)
    else:
        res = done(REVIEW, "Unstable across safe variants and normalization does not fix it", text, base, raw, nstab, n, flipped)
    yield from finish(res)
    return res


def gate(model: Callable[[str], Any], text: str, n_variants: int = 8, threshold: float = 0.8,
         min_variants: int = 3, seed: int = 0, kinds: Optional[Sequence[str]] = None,
         on_event: Optional[Callable[[Event], None]] = None, rules: Optional[RuleSet] = None,
         cache: bool = True, on_error: str = "raise") -> GateResult:
    return drain(iter_gate(model, text, n_variants, threshold, min_variants, seed, kinds, rules, cache, on_error), on_event)


async def agate(
    model: Callable[[str], Any],
    text: str,
    n_variants: int = 8,
    threshold: float = 0.8,
    min_variants: int = 3,
    seed: int = 0,
    kinds: Optional[Sequence[str]] = None,
    on_event: Optional[Callable[[Event], None]] = None,
    rules: Optional[RuleSet] = None,
    cache: bool = True,
    on_error: str = "raise",
) -> GateResult:
    """Async version of gate(). Runs gate() in a background worker thread via asyncio.to_thread."""
    return await asyncio.to_thread(
        gate,
        model=model,
        text=text,
        n_variants=n_variants,
        threshold=threshold,
        min_variants=min_variants,
        seed=seed,
        kinds=kinds,
        on_event=on_event,
        rules=rules,
        cache=cache,
        on_error=on_error,
    )
