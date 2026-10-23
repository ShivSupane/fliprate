"""Beginner and production-friendly high-level pipeline helpers.

Designed for learners exploring robustness and production engineers integrating FlipRate
into real systems without writing boilerplates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from .evaluation import Report, test
from .gating import GateResult, gate
from .models import ProtectedModel, protect
from .normalization import normalize


@dataclass
class QuickCheckResult:
    """Simplified, beginner-friendly verdict for quick checks."""
    is_safe: bool
    decision: str
    prediction: str
    suggested_text: str
    reason: str
    flipped_variants: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "decision": self.decision,
            "prediction": self.prediction,
            "suggested_text": self.suggested_text,
            "reason": self.reason,
            "flipped_variants": self.flipped_variants,
        }

    def __getitem__(self, item: str) -> Any:
        return self.to_dict()[item]

    def __bool__(self) -> bool:
        return self.is_safe

    def __str__(self) -> str:
        status = "SAFE (PASS)" if self.is_safe else f"ATTENTION ({self.decision})"
        return f"[{status}] Prediction: '{self.prediction}' | Text: '{self.suggested_text}' | {self.reason}"


def quick_check(model: Any, text: str, **kwargs) -> QuickCheckResult:
    """Run an intuitive, beginner-friendly quality check on a message and model.

    Example:
        res = fr.quick_check(my_model, "mala product avadlaaa nahi")
        if res.is_safe:
            print("Model prediction is stable:", res.prediction)
        else:
            print("Attention needed:", res.reason)
            print("Suggested normalized text:", res.suggested_text)
    """
    res = gate(model, text, **kwargs)
    is_safe = (res.decision in ("PASS", "TRANSFORM"))
    return QuickCheckResult(
        is_safe=is_safe,
        decision=res.decision,
        prediction=str(res.prediction),
        suggested_text=res.text_used,
        reason=res.reason,
        flipped_variants=res.flipped_variants,
    )


def check_model(model: Any, dataset: Any, **kwargs) -> Report:
    """Beginner-friendly alias to test() for auditing model robustness on a dataset."""
    return test(model, dataset, **kwargs)


def normalize_text(text: str, lowercase: bool = True, rules: Optional[Any] = None) -> str:
    """Simple string-in, string-out text normalizer.

    Example:
        clean = fr.normalize_text("Mala product avadlaaa nhi", lowercase=False)
        # -> "Mala product avadla nahi"
    """
    return normalize(text, lowercase=lowercase, rules=rules).normalized


def safe_predict(model: Any, text: str, on_review: str = "normalized", **kwargs) -> str:
    """Predict label with automatic robust normalization.

    Runs gate check on input:
      - If PASS: returns model(text)
      - If TRANSFORM: returns model(normalized_text)
      - If REVIEW: based on on_review:
          - "normalized": runs model on normalized text
          - "original": runs model on original text
          - "raise": raises RuntimeError explaining instability
    """
    res = gate(model, text, **kwargs)
    if res.decision in ("PASS", "TRANSFORM"):
        return str(res.prediction)
    if on_review == "raise":
        raise RuntimeError(f"Input triggered REVIEW ({res.reason}): {text}")
    if on_review == "original":
        return str(model(text))
    # default: predict on normalized text
    return str(model(res.text_used))


@dataclass
class RuntimeGuardResult:
    """Structured decision object for production gating systems."""
    decision: str
    prediction: Optional[str]
    clean_text: str
    should_route_to_human: bool
    reason: str
    metadata: Dict[str, Any]


def runtime_guard(model: Any, text: str, mode: str = "lenient", **kwargs) -> RuntimeGuardResult:
    """Production runtime gate with 'strict' or 'lenient' policy.

    In 'lenient' mode (default):
      - PASS & TRANSFORM are accepted automatically.
      - REVIEW routes to human review (`should_route_to_human=True`).

    In 'strict' mode:
      - Only PASS is accepted automatically without warning.
      - TRANSFORM and REVIEW both flag `should_route_to_human=True`.
    """
    res = gate(model, text, **kwargs)
    if mode == "strict":
        should_route = (res.decision != "PASS")
    else:
        should_route = (res.decision == "REVIEW")

    return RuntimeGuardResult(
        decision=res.decision,
        prediction=str(res.prediction) if res.prediction is not None else None,
        clean_text=res.text_used,
        should_route_to_human=should_route,
        reason=res.reason,
        metadata={
            "raw_stability": res.raw_stability,
            "normalized_stability": res.normalized_stability,
            "n_variants": res.n_variants,
            "flipped_variants": res.flipped_variants,
        },
    )


def batch_guard(model: Any, texts: Sequence[str], **kwargs) -> List[GateResult]:
    """Evaluate quality gate on a batch of messages."""
    return [gate(model, t, **kwargs) for t in texts]


def evaluate_pipeline(model: Any, dataset: Any, save_path: Optional[str] = None, **kwargs) -> Report:
    """Evaluate model robustness on a dataset and optionally save report to JSON."""
    report = test(model, dataset, **kwargs)
    if save_path:
        report.save(save_path)
    return report


def make_protected_model(model: Any, rules: Optional[Any] = None, lowercase: bool = True) -> ProtectedModel:
    """Create a model wrapped with automatic normalization. Production alias for `protect()`."""
    return protect(model, rules=rules, lowercase=lowercase)
