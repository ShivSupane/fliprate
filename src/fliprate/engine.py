"""FlipRate engine and testing assertions.

Provides the model-bound FlipRate configuration object and CI assertion helpers.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .evaluation import Report, test
from .gating import Decision, GateResult, gate
from .models import ProtectedModel, protect
from .normalization import NormalizeResult, normalize
from .perturbation import Variant, perturb
from .rules import RuleSet


def assert_stable(
    model: Any,
    dataset: Any,
    max_flip_rate: Optional[float] = None,
    min_accuracy: Optional[float] = None,
    max_accuracy_drop: Optional[float] = None,
    n_variants: int = 5,
    seed: int = 0,
    rules: Optional[Any] = None,
    mode: str = "thorough",
    early_stop: bool = False,
    protected_terms: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> Report:
    """Evaluate model robustness on a dataset and assert quality requirements pass.

    Raises:
        AssertionError: with a detailed message identifying each breached threshold.
        ValueError/FileNotFoundError/TypeError: for data loading or configuration errors.

    Example:
        fr.assert_stable(my_model, "data/test.csv", max_flip_rate=0.05, min_accuracy=0.80)
    """
    thresholds = {}
    if max_flip_rate is not None:
        thresholds["max_flip_rate"] = max_flip_rate
    if min_accuracy is not None:
        thresholds["min_accuracy"] = min_accuracy
    if max_accuracy_drop is not None:
        thresholds["max_accuracy_drop"] = max_accuracy_drop

    report = test(
        model,
        dataset,
        n_variants=n_variants,
        seed=seed,
        rules=rules,
        mode=mode,
        early_stop=early_stop,
        thresholds=thresholds,
        protected_terms=protected_terms,
        **kwargs,
    )

    violations: List[str] = []
    if max_flip_rate is not None and report.flip_rate > max_flip_rate:
        violations.append(
            f"flip rate {report.flip_rate:.1%} exceeds threshold max_flip_rate={max_flip_rate:.1%}"
        )

    if min_accuracy is not None and report.accuracy_variants < min_accuracy:
        violations.append(
            f"accuracy on variants {report.accuracy_variants:.1%} below min_accuracy={min_accuracy:.1%}"
        )

    if max_accuracy_drop is not None and report.accuracy_drop > max_accuracy_drop:
        violations.append(
            f"accuracy drop {report.accuracy_drop:+.1%} exceeds max_accuracy_drop={max_accuracy_drop:.1%}"
        )

    if not report.evaluation_complete and report.stop_reason:
        violations.append(f"early evaluation stopped: {report.stop_reason}")

    if violations:
        raise AssertionError("FlipRate stability gate failed: " + "; ".join(violations))

    return report


class FlipRate:
    """Convenience engine binding a model and default evaluation configuration.

    Example:
        engine = fr.FlipRate(
            model=my_model,
            rules=None,
            stability_threshold=0.8,
            n_variants=8,
            seed=0,
            mode="thorough",
            protected_terms=["Tata Motors", "UPI"],
        )

        res = engine.gate("mala product avadlaaa nahi")
        rep = engine.test(dataset)
        safe = engine.protect()
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        rules: Optional[RuleSet] = None,
        stability_threshold: float = 0.8,
        n_variants: int = 8,
        seed: int = 0,
        lowercase: bool = True,
        mode: str = "thorough",
        early_stop: bool = False,
        protected_terms: Optional[Sequence[str]] = None,
    ) -> None:
        self.model = model
        self.rules = rules
        self.stability_threshold = stability_threshold
        self.n_variants = n_variants
        self.seed = seed
        self.lowercase = lowercase
        self.mode = mode
        self.early_stop = early_stop
        self.protected_terms = list(protected_terms) if protected_terms else None

    def _resolve_model(self, model_override: Optional[Any] = None) -> Any:
        m = model_override if model_override is not None else self.model
        if m is None:
            raise ValueError("No model provided to FlipRate engine or method call")
        return m

    def gate(self, text: str, model: Optional[Any] = None, **kwargs: Any) -> Decision:
        """Run single-message quality gate using configured model and settings."""
        m = self._resolve_model(model)
        opts: Dict[str, Any] = {
            "rules": self.rules,
            "threshold": self.stability_threshold,
            "n_variants": self.n_variants,
            "seed": self.seed,
            "mode": self.mode,
            "early_stop": self.early_stop,
            "protected_terms": self.protected_terms,
        }
        opts.update(kwargs)
        return gate(m, text, **opts)

    def test(self, dataset: Any, model: Optional[Any] = None, **kwargs: Any) -> Report:
        """Run dataset robustness evaluation using configured model and settings."""
        m = self._resolve_model(model)
        opts: Dict[str, Any] = {
            "rules": self.rules,
            "n_variants": self.n_variants,
            "seed": self.seed,
            "mode": self.mode,
            "early_stop": self.early_stop,
            "protected_terms": self.protected_terms,
        }
        opts.update(kwargs)
        return test(m, dataset, **opts)

    def protect(self, model: Optional[Any] = None, **kwargs: Any) -> ProtectedModel:
        """Wrap model with inference-time normalization."""
        m = self._resolve_model(model)
        opts: Dict[str, Any] = {
            "rules": self.rules,
            "lowercase": self.lowercase,
        }
        opts.update(kwargs)
        return protect(m, **opts)

    def normalize(self, text: str, **kwargs: Any) -> NormalizeResult:
        """Normalize input text using engine's rules and protected terms."""
        opts: Dict[str, Any] = {
            "rules": self.rules,
            "lowercase": self.lowercase,
            "protected_terms": self.protected_terms,
        }
        opts.update(kwargs)
        return normalize(text, **opts)

    def perturb(self, text: str, **kwargs: Any) -> List[Variant]:
        """Generate safe variants of input text using engine's rules, seed, and protected terms."""
        opts: Dict[str, Any] = {
            "rules": self.rules,
            "n": self.n_variants,
            "seed": self.seed,
            "protected_terms": self.protected_terms,
        }
        opts.update(kwargs)
        return perturb(text, **opts)

    def augment(self, dataset: Any, n_per_example: int = 1, **kwargs: Any) -> Any:
        """Generate safe training-time augmented dataset using engine rules and seed."""
        from .augmentation import augment
        opts = {
            "rules": self.rules,
            "seed": self.seed,
            "n_per_example": n_per_example,
        }
        opts.update(kwargs)
        return augment(dataset, **opts)

    def assert_stable(
        self,
        dataset: Any,
        max_flip_rate: Optional[float] = None,
        min_accuracy: Optional[float] = None,
        max_accuracy_drop: Optional[float] = None,
        model: Optional[Any] = None,
        **kwargs: Any,
    ) -> Report:
        """Assert dataset stability requirements pass."""
        m = self._resolve_model(model)
        opts: Dict[str, Any] = {
            "rules": self.rules,
            "n_variants": self.n_variants,
            "seed": self.seed,
            "protected_terms": self.protected_terms,
        }
        opts.update(kwargs)
        return assert_stable(
            m,
            dataset,
            max_flip_rate=max_flip_rate,
            min_accuracy=min_accuracy,
            max_accuracy_drop=max_accuracy_drop,
            **opts,
        )

    def __repr__(self) -> str:
        model_name = getattr(self.model, "__name__", type(self.model).__name__) if self.model else "None"
        return (
            f"FlipRate(model={model_name}, n_variants={self.n_variants}, "
            f"stability_threshold={self.stability_threshold}, seed={self.seed})"
        )
