"""FlipRate: measure and reduce how often an NLP model changes its answer
on messy romanized Hindi / Marathi / Tamil / Telugu and other Indian-language text.

    import fliprate as fr

    # Happy path: quick 1-line check
    verdict = fr.quick_check(my_model, "mala product avadlaaa nahi")
    print(verdict.is_safe, verdict.prediction)

    # Clean text directly
    clean = fr.normalize_text("mala product avadlaaa nhi")

    # Safe prediction
    pred = fr.safe_predict(my_model, "mala product avadlaaa nahi")

    # Audit model on test dataset
    report = fr.check_model(my_model, "data/test.csv")
    print(report.summary())

Author: Shiv Supane
"""
from __future__ import annotations

# ── Core: most users only need these ──────────────────────────────────────────
from .evaluation import (
    Report,
    iter_test,
    load_dataset,
    macro_f1,
    paired_difference,
    read_csv_text,
    test,
)
from .engine import FlipRate, assert_stable
from .gating import Decision, GateResult, agate, detect_script, gate, iter_gate
from .models import Predictor, ProtectedModel, protect
from .normalization import NormalizeResult, normalize
from .perturbation import KINDS, Variant, perturb
from .rules import (
    CROSS_SCRIPT_MAP,
    CROSS_SCRIPT_REV,
    DEFAULT_RULES,
    DEVANAGARI_GROUPS,
    LANGUAGE_GROUPS,
    RuleSet,
    add_custom_rule,
    describe_rule_group,
    is_safe_variant,
    load_rules,
    preview_variants,
)

# ── Happy-path & Production Pipeline ─────────────────────────────────────────
from .pipeline import (
    QuickCheckResult,
    RuntimeGuardResult,
    batch_guard,
    check_model,
    evaluate_pipeline,
    make_protected_model,
    normalize_text,
    quick_check,
    runtime_guard,
    safe_predict,
)

# ── Explainability & Debugging ───────────────────────────────────────────────
from .explain import (
    explain_gate_decision,
    preview_normalized_variants,
    print_rule_hits,
    show_why_it_flipped,
)

# ── Validation ───────────────────────────────────────────────────────────────
from .validation import validate_dataset, validate_text

# ── Model adapters ──────────────────────────────────────────────────────────
from .adapters import (
    CallableAdapter,
    HFTextClassifier,
    HttpServiceAdapter,
    LLMClassifier,
    ModelAdapter,
    SklearnAdapter,
    as_adapter,
    from_sklearn,
    llm_classifier,
    twitter_preprocess,
    with_retries,
)

# ── Benchmarking & Quality Dashboard ─────────────────────────────────────────
from .benchmark import (
    BenchmarkSuite,
    ModelBenchmarkScore,
    benchmark,
    compare_models,
)

# ── Experiment runner ────────────────────────────────────────────────────────
from .experiment import ExperimentResult, run_experiment

# ── Service Storage & Rate Limiting ──────────────────────────────────────────
from .service import (
    BaseJobStore,
    BaseRateLimiter,
    InMemoryJobStore,
    InMemoryRateLimiter,
    JobStore,
    RateLimiter,
    RedisJobStore,
    RedisRateLimiter,
)

# ── Multi-language dataset loaders ────────────────────────────────────────────
from .dakshina import (
    SUPPORTED_LANGUAGES as DAKSHINA_LANGUAGES,
    dakshina_rules,
    load_dakshina_groups,
)
from .aksharantar import (
    SUPPORTED_LANGUAGES as AKSHARANTAR_LANGUAGES,
    aksharantar_rules,
    load_aksharantar_groups,
)

# ── Streaming helpers (for building custom event consumers) ───────────────────
from .events import drain, make_event, to_sse


def guard(model, text, **kwargs):
    """Thin convenience wrapper around gate()."""
    return gate(model, text, **kwargs)


def audit(model, dataset, **kwargs):
    """Thin convenience wrapper around test()."""
    return test(model, dataset, **kwargs)


def learn_rules_from_texts(texts, **kwargs):
    """Mine spelling-rule candidates from raw texts."""
    from .dataset import mine_variants
    return mine_variants(texts, **kwargs)


__version__ = "0.5.0"
__author__ = "Shiv Supane"

__all__ = [
    # ── Happy-Path & Beginner API ────────────────────────────────────────────
    "FlipRate",
    "assert_stable",
    "quick_check",
    "QuickCheckResult",
    "check_model",
    "normalize_text",
    "safe_predict",
    "guard",
    "audit",
    "learn_rules_from_texts",

    # ── Production Pipeline ──────────────────────────────────────────────────
    "runtime_guard",
    "RuntimeGuardResult",
    "batch_guard",
    "evaluate_pipeline",
    "make_protected_model",

    # ── Explainability & Debugging ───────────────────────────────────────────
    "explain_gate_decision",
    "show_why_it_flipped",
    "preview_normalized_variants",
    "print_rule_hits",

    # ── Validation ───────────────────────────────────────────────────────────
    "validate_text",
    "validate_dataset",

    # ── Test & Gate ──────────────────────────────────────────────────────────
    "test",
    "iter_test",
    "gate",
    "iter_gate",
    "agate",
    "detect_script",
    "Report",
    "Decision",
    "GateResult",
    "paired_difference",

    # ── Fix & protect ────────────────────────────────────────────────────────
    "protect",
    "normalize",
    "perturb",
    "is_safe_variant",

    # ── Rules & Rule Management ──────────────────────────────────────────────
    "RuleSet",
    "DEFAULT_RULES",
    "DEVANAGARI_GROUPS",
    "CROSS_SCRIPT_MAP",
    "CROSS_SCRIPT_REV",
    "LANGUAGE_GROUPS",
    "load_rules",
    "add_custom_rule",
    "preview_variants",
    "describe_rule_group",

    # ── Multi-language datasets ──────────────────────────────────────────────
    "dakshina_rules",
    "load_dakshina_groups",
    "DAKSHINA_LANGUAGES",
    "aksharantar_rules",
    "load_aksharantar_groups",
    "AKSHARANTAR_LANGUAGES",

    # ── Model adapters ───────────────────────────────────────────────────────
    "ModelAdapter",
    "CallableAdapter",
    "HttpServiceAdapter",
    "as_adapter",
    "HFTextClassifier",
    "SklearnAdapter",
    "from_sklearn",
    "LLMClassifier",
    "llm_classifier",
    "twitter_preprocess",
    "with_retries",
    "ProtectedModel",
    "Predictor",

    # ── Benchmarking & Quality Dashboard ─────────────────────────────────────
    "benchmark",
    "compare_models",
    "BenchmarkSuite",
    "ModelBenchmarkScore",

    # ── Experiment ───────────────────────────────────────────────────────────
    "run_experiment",
    "ExperimentResult",

    # ── Storage & Distributed Infrastructure ─────────────────────────────────
    "BaseJobStore",
    "BaseRateLimiter",
    "InMemoryJobStore",
    "InMemoryRateLimiter",
    "JobStore",
    "RateLimiter",
    "RedisJobStore",
    "RedisRateLimiter",

    # ── Data loading ─────────────────────────────────────────────────────────
    "load_dataset",
    "read_csv_text",

    # ── Types & helpers ──────────────────────────────────────────────────────
    "NormalizeResult",
    "Variant",
    "KINDS",
    "macro_f1",
    "drain",
    "make_event",
    "to_sse",

    # ── Version ──────────────────────────────────────────────────────────────
    "__version__",
    "__author__",
]
