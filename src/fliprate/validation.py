"""Validation and schema safety checks for inputs and datasets.

Prevents silent failures, null pointers, and corrupt dataset runs in production.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def validate_text(text: Any, max_length: int = 10000, allow_empty: bool = False) -> str:
    """Validate a single input string.

    Raises:
      TypeError: if input is not a string.
      ValueError: if input is empty (and allow_empty is False) or exceeds max_length.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected text as str, got {type(text).__name__}")
    stripped = text.strip()
    if not allow_empty and not stripped:
        raise ValueError("Text input cannot be empty or whitespace only")
    if len(text) > max_length:
        raise ValueError(f"Text length ({len(text)}) exceeds maximum allowed ({max_length})")
    return text


def validate_dataset(dataset: Any, text_col: str = "text", label_col: str = "label") -> Dict[str, Any]:
    """Validate a dataset (list of tuples, dicts, or CSV path) before running robustness tests.

    Returns diagnostic dictionary:
      - `valid`: bool
      - `row_count`: int
      - `empty_rows`: int
      - `duplicate_texts`: int
      - `label_conflicts`: list of texts that have multiple differing labels
      - `unique_labels`: list of detected labels
      - `rows`: list of validated (text, label) tuples
    """
    from .dataset import load_dataset

    raw_rows = load_dataset(dataset, text_col=text_col, label_col=label_col)

    empty_rows = 0
    clean_rows: List[Tuple[str, str]] = []
    text_to_labels: Dict[str, set] = {}

    for row in raw_rows:
        t, l = str(row[0]).strip(), str(row[1]).strip()
        if not t or not l:
            empty_rows += 1
            continue
        clean_rows.append((t, l))
        if t not in text_to_labels:
            text_to_labels[t] = set()
        text_to_labels[t].add(l)

    conflicts = [t for t, labels in text_to_labels.items() if len(labels) > 1]
    duplicate_count = len(clean_rows) - len(text_to_labels)
    all_labels = sorted(list({l for _, l in clean_rows}))

    return {
        "valid": len(clean_rows) > 0 and len(conflicts) == 0,
        "row_count": len(clean_rows),
        "empty_rows": empty_rows,
        "duplicate_texts": duplicate_count,
        "label_conflicts": conflicts,
        "unique_labels": all_labels,
        "rows": clean_rows,
    }
