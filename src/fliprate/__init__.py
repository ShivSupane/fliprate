"""FlipRate: measure and reduce how often an NLP model changes its answer
on messy romanized Hindi / Marathi / Tamil / Telugu and other Indian-language text.

    import fliprate as fr

    # Test robustness
    report = fr.test(my_model, "data/test.csv")
    print(report.summary())

    # Fix it
    safe_model = fr.protect(my_model)
    report2 = fr.test(safe_model, "data/test.csv")

    # Real-time gate
    result = fr.gate(my_model, "mala product avadlaaa nahi")
    print(result.decision)   # PASS / TRANSFORM / REVIEW

    # Multi-language rules from Aksharantar (428K groups, 20 languages)
    rules = fr.aksharantar_rules("tam", "tel")
    report = fr.test(my_model, "data/test.csv", rules=rules)

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
from .gating import GateResult, agate, detect_script, gate, iter_gate
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
    is_safe_variant,
)

# ── Model adapters ──────────────────────────────────────────────────────────
from .adapters import (
    HFTextClassifier,
    LLMClassifier,
    from_sklearn,
    llm_classifier,
    twitter_preprocess,
    with_retries,
)

# ── Experiment runner ────────────────────────────────────────────────────────
from .experiment import ExperimentResult, run_experiment

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
    """Thin convenience wrapper around gate().

    This keeps the common production use-case simple: "does this input look stable,
    transformable, or review-worthy for my model?"
    """
    return gate(model, text, **kwargs)


def audit(model, dataset, **kwargs):
    """Thin convenience wrapper around test().

    Returns the same Report object as `fliprate.test`, but with a friendlier name for
    product and QA workflows.
    """
    return test(model, dataset, **kwargs)


def learn_rules_from_texts(texts, **kwargs):
    """Mine spelling-rule candidates from raw texts.

    This turns a relatively advanced data-mining workflow into a one-line helper.
    """
    from .dataset import mine_variants

    return mine_variants(texts, **kwargs)


__version__ = "0.4.0"
__author__ = "Shiv Supane"

__all__ = [
    # ── Test & measure ──────────────────────────────────────────────────────
    "test",
    "iter_test",
    "gate",
    "iter_gate",
    "agate",
    "detect_script",
    "Report",
    "GateResult",
    "paired_difference",
    "guard",
    "audit",
    "learn_rules_from_texts",

    # ── Fix & protect ────────────────────────────────────────────────────────
    "protect",
    "normalize",
    "perturb",
    "is_safe_variant",

    # ── Rules ────────────────────────────────────────────────────────────────
    "RuleSet",
    "DEFAULT_RULES",
    "DEVANAGARI_GROUPS",
    "CROSS_SCRIPT_MAP",
    "CROSS_SCRIPT_REV",
    "LANGUAGE_GROUPS",

    # ── Multi-language datasets ──────────────────────────────────────────────
    "dakshina_rules",
    "load_dakshina_groups",
    "DAKSHINA_LANGUAGES",
    "aksharantar_rules",
    "load_aksharantar_groups",
    "AKSHARANTAR_LANGUAGES",

    # ── Model adapters ────────────────────────────────────────────────────────
    "HFTextClassifier",
    "LLMClassifier",
    "llm_classifier",
    "from_sklearn",
    "twitter_preprocess",
    "with_retries",
    "ProtectedModel",
    "Predictor",

    # ── Experiment ───────────────────────────────────────────────────────────
    "run_experiment",
    "ExperimentResult",

    # ── Data loading ────────────────────────────────────────────────────────
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

    # ── Version ─────────────────────────────────────────────────────────────
    "__version__",
    "__author__",
]
