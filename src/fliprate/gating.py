import asyncio
import json
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence

from .events import Event, drain, make_event
from .models import Predictor, batch_predict
from .normalization import normalize
from .perturbation import perturb
from .rules import DEFAULT_RULES, RuleSet

PASS, TRANSFORM, REVIEW = "PASS", "TRANSFORM", "REVIEW"


@dataclass
class Decision:
    """Canonical result of single-message gating."""
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
    mode: str = "thorough"
    evaluation_complete: bool = True
    n_variants_requested: int = 0
    n_variants_evaluated: int = 0
    stop_reason: Optional[str] = None

    @property
    def is_safe(self) -> bool:
        """True if prediction survives safe variants (decision is PASS or TRANSFORM)."""
        return self.decision in (PASS, TRANSFORM)

    @property
    def should_route_to_human(self) -> bool:
        """True if prediction instability or high risk requires human review."""
        return self.decision == REVIEW

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
            "is_safe": self.is_safe,
            "should_route_to_human": self.should_route_to_human,
            "mode": self.mode,
            "evaluation_complete": self.evaluation_complete,
            "n_variants_requested": self.n_variants_requested,
            "n_variants_evaluated": self.n_variants_evaluated,
            "stop_reason": self.stop_reason,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]

    def __bool__(self) -> bool:
        return self.is_safe


GateResult = Decision


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


def _is_latin_char(c: str) -> bool:
    cp = ord(c)
    if (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
        return True
    if (0x00C0 <= cp <= 0x024F) or (0x1E00 <= cp <= 0x1EFF):
        return True
    return unicodedata.name(c, "").startswith("LATIN")


def _is_devanagari_char(c: str) -> bool:
    cp = ord(c)
    return (0x0900 <= cp <= 0x097F) or (0xA8E0 <= cp <= 0xA8FF)


def _detect_primary_script_name(letters: List[str]) -> str:
    for c in letters:
        name = unicodedata.name(c, "")
        if "BENGALI" in name:
            return "Bengali"
        if "TAMIL" in name:
            return "Tamil"
        if "TELUGU" in name:
            return "Telugu"
        if "KANNADA" in name:
            return "Kannada"
        if "MALAYALAM" in name:
            return "Malayalam"
        if "GUJARATI" in name:
            return "Gujarati"
        if "GURMUKHI" in name:
            return "Gurmukhi"
        if "ORIYA" in name:
            return "Odia"
        if "ARABIC" in name:
            return "Arabic"
        if "CYRILLIC" in name:
            return "Cyrillic"
    return "non-Latin/non-Devanagari"


def detect_script(text: str) -> str:
    """Classify text script into 'latin' | 'devanagari' | 'mixed' | 'other'.

    Counts Latin letters via str.isalpha() and Devanagari characters via the
    Unicode block range U+0900–U+097F (which includes consonants, vowels, matras,
    and combining marks — all of which count as Devanagari script units).
    Digits, punctuation, and emoji are ignored.
    Returns 'other' for empty text, digits/symbols only, or unsupported Indic scripts.
    The 85% rule applies: if >= 85% are one script it is 'latin' or 'devanagari';
    if both are present and together >= 85% it is 'mixed'; otherwise 'other'.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    latin_count = 0
    deva_count = 0
    for c in text:
        if _is_emoji_or_modifier(c):
            continue
        cp = ord(c)
        if (0x0900 <= cp <= 0x097F) or (0xA8E0 <= cp <= 0xA8FF):
            # Any Devanagari block character counts (consonants, matras, anusvara, visarga…)
            deva_count += 1
        elif c.isalpha() and _is_latin_char(c):
            latin_count += 1

    total = latin_count + deva_count
    if total == 0:
        return "other"

    if latin_count / total >= 0.85:
        return "latin"
    if deva_count / total >= 0.85:
        return "devanagari"
    if latin_count > 0 and deva_count > 0:
        return "mixed"
    return "other"


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
    script_aware: bool = True,
    mode: str = "thorough",
    early_stop: bool = False,
) -> Generator[Event, None, GateResult]:
    """Decide whether `text` can go to `model` as is (PASS), after cleaning (TRANSFORM), or needs a human (REVIEW).

    Modes:
      * 'fast': one normal inference path, no extra variant predictions. Robustness is not checked.
      * 'balanced': evaluates a deterministic configured sample of reviewed transformations.
      * 'thorough': evaluates all configured variants for complete metrics.

    Early stopping (early_stop=True):
      Stops on first detected flip or mathematical threshold breach, marking evaluation_complete=False.
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
    if mode not in ("fast", "balanced", "thorough"):
        raise ValueError(f"mode must be 'fast', 'balanced', or 'thorough', got {mode!r}")

    rules = rules or DEFAULT_RULES
    predictor = Predictor(model, cache=cache)
    warnings: List[str] = []
    yield make_event("received", chars=len(text))

    def done(
        decision: str,
        reason: str,
        used: str,
        pred: Any,
        raw: Optional[float],
        norm: Optional[float],
        n: int,
        flipped: Sequence[str] = (),
        eval_complete: bool = True,
        n_req: int = 0,
        n_eval: int = 0,
        stop_reason: Optional[str] = None,
    ) -> GateResult:
        return GateResult(
            decision,
            reason,
            text,
            used,
            pred,
            raw,
            norm,
            n,
            warnings,
            list(flipped),
            mode=mode,
            evaluation_complete=eval_complete,
            n_variants_requested=n_req,
            n_variants_evaluated=n_eval,
            stop_reason=stop_reason,
        )

    def finish(res: GateResult):
        return [
            make_event("gate", decision=res.decision, reason=res.reason),
            make_event("done", result=res.to_dict()),
        ]

    if not text.strip():
        res = done(REVIEW, "Empty input", text, None, None, None, 0, n_req=0, n_eval=0)
        yield from finish(res)
        return res

    if not script_aware:
        ratio = _latin_ratio(text)
        yield make_event("script_check", latin_ratio=round(ratio, 2))
        if ratio < 0.5:
            res = done(REVIEW, "Mostly non-Latin script; v0.1 supports romanized text only", text, None, None, None, 0, n_req=0, n_eval=0)
            yield from finish(res)
            return res
    else:
        script = detect_script(text)
        yield make_event("script_check", script=script)
        if script == "other":
            letters = [c for c in text if c.isalpha() and not _is_emoji_or_modifier(c)]
            if not letters:
                res = done(REVIEW, "No script letters found", text, None, None, None, 0, n_req=0, n_eval=0)
            else:
                sname = _detect_primary_script_name(letters)
                res = done(REVIEW, f"unsupported script: {sname}", text, None, None, None, 0, n_req=0, n_eval=0)
            yield from finish(res)
            return res

    for tok in rules.ambiguous_tokens(text):
        warnings.append(f"'{tok}' is ambiguous ({rules.ambiguous[tok]}); left unchanged")

    # Fast mode: single inference pass, zero variant evaluations
    if mode == "fast":
        warnings.append("Robustness was not checked (mode='fast')")
        try:
            base = predictor.predict([text])[0]
        except Exception as exc:
            if on_error == "review":
                res = done(REVIEW, f"model error: {type(exc).__name__}: {exc}", text, None, None, None, 0, n_req=0, n_eval=0)
                yield from finish(res)
                return res
            raise
        res = done(
            PASS,
            "Fast mode: inference executed; robustness not evaluated",
            text,
            base,
            raw=None,
            norm=None,
            n=0,
            n_req=0,
            n_eval=0,
            eval_complete=True,
            stop_reason=None,
        )
        yield from finish(res)
        return res

    # Balanced mode samples fewer variants deterministically
    target_variants = min(n_variants, 3) if mode == "balanced" else n_variants
    variants = perturb(text, target_variants, kinds, seed, rules=rules)
    n_requested = len(variants)

    try:
        base = predictor.predict([text])[0]
    except Exception as exc:
        if on_error == "review":
            res = done(REVIEW, f"model error: {type(exc).__name__}: {exc}", text, None, None, None, 0, n_req=n_requested, n_eval=0)
            yield from finish(res)
            return res
        raise

    yield make_event("variants", count=len(variants), prediction=base)
    if len(variants) < min_variants and mode != "balanced":
        warnings.append(f"Only {len(variants)} safe variants could be generated; stability is weakly measured")

    flipped = []
    n_evaluated = 0
    stopped_early = False
    stop_reason: Optional[str] = None

    if early_stop:
        for v in variants:
            try:
                p = predictor.predict([v.text])[0]
            except Exception as exc:
                if on_error == "review":
                    res = done(REVIEW, f"model error: {type(exc).__name__}: {exc}", text, None, None, None, len(variants), n_req=n_requested, n_eval=n_evaluated)
                    yield from finish(res)
                    return res
                raise
            n_evaluated += 1
            same = (p == base)
            if not same:
                flipped.append(v.text)
            yield make_event("prediction", variant=v.text, kind=v.kind, label=p, same=same)

            if not same:
                stopped_early = True
                stop_reason = "first_flip_detected"
                warnings.append("Early stop on flip: evaluation incomplete; stability is not an exact measurement")
                break

            max_possible_stable = (len(variants) - n_evaluated) + (n_evaluated - len(flipped))
            if (max_possible_stable / len(variants)) < threshold:
                stopped_early = True
                stop_reason = "threshold_unachievable"
                warnings.append("Early stop on threshold breach: evaluation incomplete")
                break
    else:
        try:
            vpreds = predictor.predict([v.text for v in variants])
        except Exception as exc:
            if on_error == "review":
                res = done(REVIEW, f"model error: {type(exc).__name__}: {exc}", text, None, None, None, len(variants), n_req=n_requested, n_eval=0)
                yield from finish(res)
                return res
            raise
        n_evaluated = len(variants)
        for v, p in zip(variants, vpreds):
            same = (p == base)
            if not same:
                flipped.append(v.text)
            yield make_event("prediction", variant=v.text, kind=v.kind, label=p, same=same)

    raw = None if stopped_early else ((len(variants) - len(flipped)) / len(variants) if variants else None)
    yield make_event("raw_stability", value=raw)

    if stopped_early:
        res = done(
            decision=REVIEW,
            reason=f"Instability detected ({stop_reason}); evaluation incomplete",
            used=text,
            pred=base,
            raw=None,
            norm=None,
            n=n_evaluated,
            flipped=flipped,
            eval_complete=False,
            n_req=n_requested,
            n_eval=n_evaluated,
            stop_reason=stop_reason,
        )
        yield from finish(res)
        return res

    if raw is None:
        warnings.append("No variants available, so stability could not be measured")
        res = done(PASS, "No variants available to test stability", text, base, None, None, 0, n_req=n_requested, n_eval=n_evaluated)
        yield from finish(res)
        return res
    if not flipped:
        res = done(PASS, "Prediction is stable across safe variants", text, base, raw, None, len(variants), n_req=n_requested, n_eval=n_evaluated)
        yield from finish(res)
        return res

    # Some variants flip. Would approved normalization in front of the model help?
    norm = normalize(text, rules=rules)
    nstab: Optional[float] = None
    nbase = None
    can_transform = False
    canonical_stable = False
    if norm.normalized != text and rules.is_safe_variant(text, norm.normalized):
        yield make_event("normalized", text=norm.normalized, changes=norm.changes)
        try:
            npreds = predictor.predict([norm.normalized] + [normalize(v.text, rules=rules).normalized for v in variants])
        except Exception as exc:
            if on_error == "review":
                res = done(REVIEW, f"model error: {type(exc).__name__}: {exc}", norm.normalized, None, raw, None, len(variants), n_req=n_requested, n_eval=n_evaluated)
                yield from finish(res)
                return res
            raise
        nbase = npreds[0]
        nstab = sum(p == nbase for p in npreds[1:]) / len(variants)
        yield make_event("normalized_stability", value=nstab)
        can_transform = nstab >= threshold and nstab > raw
    elif norm.normalized != text:
        warnings.append("Normalization would change the meaning; it was not applied")
    elif norm.normalized == text:
        norm_variants = [normalize(v.text, rules=rules).normalized for v in variants]
        try:
            npreds = batch_predict(predictor.model, norm_variants)
        except Exception as exc:
            if on_error == "review":
                res = done(REVIEW, f"model error: {type(exc).__name__}: {exc}", text, None, raw, None, len(variants), n_req=n_requested, n_eval=n_evaluated)
                yield from finish(res)
                return res
            raise
        nbase = base
        nstab = sum(p == nbase for p in npreds) / len(variants)
        yield make_event("normalized_stability", value=nstab)
        if nstab >= threshold and (raw < threshold or nstab > raw):
            canonical_stable = True
            warnings.append("input is already in canonical form; model is sensitive to other spellings (protect() recommended)")

    n = len(variants)
    if can_transform:
        res = done(TRANSFORM, "Stable after approved normalization", norm.normalized, nbase, raw, nstab, n, flipped, n_req=n_requested, n_eval=n_evaluated)
    elif canonical_stable:
        res = done(PASS, "Input is in canonical form and stable after normalization", text, base, raw, nstab, n, flipped, n_req=n_requested, n_eval=n_evaluated)
    elif raw >= threshold:
        warnings.append(f"{len(flipped)} of {n} safe variants changed the answer; normalization did not improve this")
        res = done(PASS, "Mostly stable across safe variants", text, base, raw, nstab, n, flipped, n_req=n_requested, n_eval=n_evaluated)
    else:
        res = done(REVIEW, "Unstable across safe variants and normalization does not fix it", text, base, raw, nstab, n, flipped, n_req=n_requested, n_eval=n_evaluated)
    yield from finish(res)
    return res


def gate(
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
    script_aware: bool = True,
    mode: str = "thorough",
    early_stop: bool = False,
) -> GateResult:
    return drain(
        iter_gate(
            model,
            text,
            n_variants=n_variants,
            threshold=threshold,
            min_variants=min_variants,
            seed=seed,
            kinds=kinds,
            rules=rules,
            cache=cache,
            on_error=on_error,
            script_aware=script_aware,
            mode=mode,
            early_stop=early_stop,
        ),
        on_event,
    )


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
    script_aware: bool = True,
    mode: str = "thorough",
    early_stop: bool = False,
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
        script_aware=script_aware,
        mode=mode,
        early_stop=early_stop,
    )
