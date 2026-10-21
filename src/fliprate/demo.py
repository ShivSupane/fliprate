"""A deliberately simple keyword classifier and a tiny sample dataset, for demos and tests.

The keyword model is a TOY. It matches exact spellings, so it is brittle on purpose:
that makes FlipRate's effect visible without downloading any ML model.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from .evaluation import load_dataset
from .rules import WORD_RE

KEYWORDS = {
    "delivery_complaint": {"delivery", "late", "parcel", "courier", "package", "tracking", "milala", "pohchala"},
    "payment_issue": {"payment", "paise", "refund", "upi", "card", "kat", "kaple", "amount"},
    "order_issue": {"order", "cancel", "wrong", "galat", "item"},
    "product_feedback": {"product", "quality", "avadla", "accha", "badhiya", "mast", "chhan", "pasand"},
    "human_support": {"human", "agent", "manager", "representative", "care", "insaan", "senior", "baat"},
}
_PRIORITY = ["delivery_complaint", "payment_issue", "order_issue", "product_feedback", "human_support"]


def keyword_classifier(text: str) -> str:
    words = {w.lower() for w in WORD_RE.findall(text)}
    scores = {label: len(words & kws) for label, kws in KEYWORDS.items()}
    best = max(scores.values())
    if best == 0:
        return "human_support"  # unknown -> escalate to a person
    return next(label for label in _PRIORITY if scores[label] == best)


def sample_dataset_path() -> Path:
    return Path(__file__).parent / "data" / "sample_support.csv"


def sample_dataset() -> List[Tuple[str, str]]:
    return load_dataset(sample_dataset_path())


def load_model(spec: Optional[str] = None) -> Callable[[str], str]:
    """`None` -> toy model. Or "package.module:function" to load your own classifier."""
    spec = spec or os.environ.get("FLIPRATE_MODEL")
    if not spec:
        return keyword_classifier
    module, _, func = spec.partition(":")
    if not func:
        raise ValueError("Model spec must look like 'package.module:function'")
    return getattr(importlib.import_module(module), func)
