"""Explainability, debugging, and audit diagnostics for FlipRate.

Provides clear, human-understandable explanations for why a gate decision was made,
which tokens triggered a model flip, and what rules match an input.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .gating import detect_script, gate
from .normalization import normalize
from .perturbation import perturb
from .rules import DEFAULT_RULES, RuleSet, tokenize_keep, word_key


def explain_gate_decision(text_or_result: Any, model: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
    """Provide a structured, human-readable explanation for a gate() decision.
    Accepts (text, model), (model, text), or a GateResult directly.

    Returns a dict with:
      - `decision`: "PASS" / "TRANSFORM" / "REVIEW"
      - `prediction`: model prediction on the selected text
      - `script`: detected script ("latin", "devanagari", "mixed", etc.)
      - `raw_stability`: stability score on raw variants (0.0 to 1.0)
      - `normalized_stability`: stability score after normalization (or None)
      - `reason`: short mechanical explanation
      - `summary`: plain-English explanation of why this verdict occurred
      - `recommendation`: what the user or production system should do next
      - `flipped_variants`: list of variant strings that caused a prediction flip
    """
    if hasattr(text_or_result, "decision") and hasattr(text_or_result, "text_used"):
        res = text_or_result
        text = res.text_used
    elif callable(text_or_result) and isinstance(model, str):
        text = model
        res = gate(text_or_result, text, **kwargs)
    else:
        text = str(text_or_result)
        if model is None:
            raise ValueError("model must be provided when passing a text string to explain_gate_decision()")
        res = gate(model, text, **kwargs)

    script = detect_script(text)

    if res.decision == "PASS":
        summary = (
            f"The model's prediction is highly stable across tested variations "
            f"({res.raw_stability:.0%} agreement). The message can be safely processed."
        )
        recommendation = "Safe to serve directly to your downstream application."
    elif res.decision == "TRANSFORM":
        summary = (
            f"The model was sensitive to raw spelling noise (raw stability {res.raw_stability:.0%}), "
            f"but normalizing the message fully resolved the instability "
            f"(normalized stability {res.normalized_stability:.0%})."
        )
        recommendation = "Use the normalized text ('" + res.text_used + "') or wrap your model with fr.protect()."
    else:  # REVIEW
        summary = (
            f"The model prediction is unstable across spelling variations and normalization could not "
            f"reliably stabilize it (stability {res.raw_stability:.0%})."
        )
        recommendation = "Route this message to a human agent or fallback handling system."

    return {
        "decision": res.decision,
        "prediction": str(res.prediction),
        "text_used": res.text_used,
        "script": script,
        "raw_stability": res.raw_stability,
        "normalized_stability": res.normalized_stability,
        "reason": res.reason,
        "summary": summary,
        "recommendation": recommendation,
        "flipped_variants": res.flipped_variants,
    }


def show_why_it_flipped(text_or_model: Any, model_or_text: Any, **kwargs) -> List[Dict[str, Any]]:
    """Inspect and highlight every variant that caused the model's prediction to flip.
    Accepts arguments in either order: `(text, model)` or `(model, text)`.

    Returns a list of dicts with:
      - `variant_text`: the modified variant string
      - `kind`: type of perturbation ("spelling", "repetition", "unicode", "punctuation", etc.)
      - `original_prediction`: label predicted on original text
      - `variant_prediction`: label predicted on this variant
      - `changed_words`: list of (original_word, variant_word) pairs that differ
    """
    if callable(text_or_model) and isinstance(model_or_text, str):
        model, text = text_or_model, model_or_text
    elif callable(model_or_text) and isinstance(text_or_model, str):
        text, model = text_or_model, model_or_text
    else:
        text, model = text_or_model, model_or_text

    orig_pred = str(model(text))
    variants = perturb(text, **kwargs)
    flips = []

    for v in variants:
        pred = str(model(v.text))
        if pred != orig_pred:
            orig_words = text.split()
            var_words = v.text.split()
            diffs = []
            if len(orig_words) == len(var_words):
                for ow, vw in zip(orig_words, var_words):
                    if ow != vw:
                        diffs.append((ow, vw))
            flips.append({
                "variant_text": v.text,
                "kind": v.kind,
                "original_prediction": orig_pred,
                "variant_prediction": pred,
                "diffs": diffs,
            })

    return flips


def preview_normalized_variants(text: str, n: int = 5, **kwargs) -> List[Dict[str, str]]:
    """Preview how generated variants of a message get normalized by the cleaner.

    Returns list of dicts with:
      - `original`: original variant text
      - `kind`: perturbation kind
      - `normalized`: cleaned text produced by normalize()
    """
    variants = perturb(text, n=n, **kwargs)
    out = []
    for v in variants:
        norm = normalize(v.text).normalized
        out.append({
            "original": v.text,
            "kind": v.kind,
            "normalized": norm,
        })
    return out


def print_rule_hits(text: str, rules: Optional[RuleSet] = None) -> List[Dict[str, Any]]:
    """Scan text and report every word token that matched an approved canonical rule or negation.

    Returns list of dicts with:
      - `token`: the word in the message
      - `canonical`: canonical form mapped by the rule (or None)
      - `is_negation`: boolean indicating whether the token is a negation word
    """
    rules = rules or DEFAULT_RULES
    tokens = tokenize_keep(text)
    hits = []

    for t in tokens:
        if not t.strip() or not any(c.isalnum() for c in t):
            continue
        key = word_key(t)
        canon = rules.key_to_canon.get(key)
        is_neg = (key in rules.negations or t.lower() in rules.negations)
        if canon is not None or is_neg:
            hits.append({
                "token": t,
                "canonical": canon,
                "is_negation": is_neg,
            })

    return hits
