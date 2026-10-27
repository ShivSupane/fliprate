"""Explainability, debugging, and audit diagnostics for FlipRate.

Provides clear, human-understandable explanations for why a gate decision was made,
which tokens triggered a model flip, and what rules match an input.
"""
from __future__ import annotations

import difflib
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


def _compute_token_diffs(orig_text: str, var_text: str) -> List[Tuple[str, str]]:
    """Compute exact word-level token diffs between orig_text and var_text using difflib.

    Handles substitutions, deletions, and insertions even when token counts change.
    """
    orig_words = orig_text.split()
    var_words = var_text.split()
    matcher = difflib.SequenceMatcher(None, orig_words, var_words)
    diffs: List[Tuple[str, str]] = []
    for tag, alo, ahi, blo, bhi in matcher.get_opcodes():
        if tag == "replace":
            diffs.append((" ".join(orig_words[alo:ahi]), " ".join(var_words[blo:bhi])))
        elif tag == "delete":
            diffs.append((" ".join(orig_words[alo:ahi]), ""))
        elif tag == "insert":
            diffs.append(("", " ".join(var_words[blo:bhi])))
    return diffs


def show_why_it_flipped(text_or_model: Any, model_or_text: Any, **kwargs) -> List[Dict[str, Any]]:
    """Inspect and highlight every variant that caused the model's prediction to flip.
    Accepts arguments in either order: `(text, model)` or `(model, text)`.

    Returns a list of dicts with:
      - `original_text`: original text string
      - `variant_text`: the modified variant string
      - `kind`: type of perturbation ("spelling", "repetition", "unicode", "punctuation", etc.)
      - `original_prediction`: label predicted on original text
      - `variant_prediction`: label predicted on this variant
      - `diffs`: list of (original_tokens, variant_tokens) pairs that differ
      - `changed_tokens`: alias for diffs
      - `description`: non-causal statement describing observed difference
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
            diffs = _compute_token_diffs(text, v.text)
            diff_str = (
                ", ".join(
                    f"'{a}' -> '{b}'" if a and b else (f"deleted '{a}'" if a else f"inserted '{b}'")
                    for a, b in diffs
                )
                if diffs
                else "no word-level diffs detected"
            )
            description = (
                f"Observed prediction shifted from '{orig_pred}' to '{pred}' when '{v.kind}' transformation was applied. "
                f"Changed tokens: {diff_str}. (Note: reflects observed sensitivity to input variation, not internal model reasoning.)"
            )
            flips.append({
                "original_text": text,
                "variant_text": v.text,
                "kind": v.kind,
                "original_prediction": orig_pred,
                "variant_prediction": pred,
                "diffs": diffs,
                "changed_tokens": diffs,
                "changed_words": diffs,
                "description": description,
            })

    return flips


def explain_report_flips(report: Any, opt_in_examples: bool = False, max_examples: int = 10) -> Dict[str, Any]:
    """Provide a structured, privacy-conscious explanation of model flips from an evaluation Report.

    By default (`opt_in_examples=False`), raw text examples are omitted for data privacy.
    Set `opt_in_examples=True` to inspect specific original and variant texts and diffs.
    """
    flips = getattr(report, "flips", [])
    by_kind = getattr(report, "by_kind", {})
    transitions = getattr(report, "transitions", {})

    kind_counts: Dict[str, int] = {}
    for f in flips:
        k = f.get("kind", "unknown")
        kind_counts[k] = kind_counts.get(k, 0) + 1

    summary = {
        "total_flips": len(flips),
        "flip_rate": getattr(report, "flip_rate", 0.0),
        "flips_by_transformation": kind_counts,
        "transitions": dict(transitions),
        "per_transformation_metrics": by_kind,
        "examples_redacted": not opt_in_examples,
    }

    if not opt_in_examples:
        summary["privacy_note"] = (
            "Raw text examples omitted for privacy. Pass `opt_in_examples=True` to inspect individual text examples."
        )
        summary["examples"] = []
    else:
        detailed_examples = []
        for f in flips[:max_examples]:
            orig = f.get("original", "")
            var = f.get("variant", "")
            k = f.get("kind", "unknown")
            op = f.get("original_pred", "")
            vp = f.get("variant_pred", "")
            diffs = _compute_token_diffs(orig, var) if orig and var else []
            diff_str = (
                ", ".join(
                    f"'{a}' -> '{b}'" if a and b else (f"deleted '{a}'" if a else f"inserted '{b}'")
                    for a, b in diffs
                )
                if diffs
                else "none"
            )
            detailed_examples.append({
                "kind": k,
                "original_prediction": op,
                "variant_prediction": vp,
                "original_text": orig,
                "variant_text": var,
                "changed_tokens": diffs,
                "description": (
                    f"Observed prediction shifted from '{op}' to '{vp}' under '{k}' transformation. "
                    f"Changed tokens: {diff_str}. (Note: reflects observed input/output difference, not causal model reasoning.)"
                ),
            })
        summary["examples"] = detailed_examples

    return summary


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
