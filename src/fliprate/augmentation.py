"""Training-time data augmentation for romanized Hindi/Marathi and Indic NLP.

[EXPERIMENTAL] This module provides controlled data augmentation using reviewed,
meaning-preserving transformations.

Safety Rules:
  * Augment training data only — never generate augmented examples from validation/test rows.
  * Preserves ground-truth labels ONLY for transformations passing strict safety and negation checks.
  * Preserves originals by default.
  * Deterministic output controlled by random seed.
  * Records detailed counts by transformation type and tracks all skipped unsafe candidates.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from .evaluation import Row, load_dataset
from .perturbation import KINDS, perturb
from .rules import DEFAULT_RULES, RuleSet


@dataclass
class AugmentedRow:
    """Detailed record for an augmented or preserved dataset row."""
    text: str
    label: Any
    source_row: int
    is_augmented: bool
    transformation_kind: Optional[str]
    original_text: str

    def to_tuple(self) -> Tuple[str, Any]:
        return (self.text, self.label)

    def to_dict(self, text_col: str = "text", label_col: str = "label") -> Dict[str, Any]:
        return {
            text_col: self.text,
            label_col: self.label,
            "is_augmented": self.is_augmented,
            "transformation_kind": self.transformation_kind,
            "original_text": self.original_text,
            "source_row": self.source_row,
        }


@dataclass
class AugmentationResult:
    """Result of training-time augmentation including metrics, diagnostics, and rows."""
    detailed_rows: List[AugmentedRow]
    n_original: int
    n_augmented: int
    by_kind: Dict[str, int]
    skipped_candidates: int
    seed: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_rows(self) -> int:
        return len(self.detailed_rows)

    def to_rows(self) -> List[Tuple[str, Any]]:
        """Return as canonical list of (text, label) tuples."""
        return [r.to_tuple() for r in self.detailed_rows]

    def to_dicts(self, text_col: str = "text", label_col: str = "label") -> List[Dict[str, Any]]:
        """Return list of dicts suitable for DataFrame construction or JSON serialization."""
        return [r.to_dict(text_col=text_col, label_col=label_col) for r in self.detailed_rows]

    def to_dataframe(self, text_col: str = "text", label_col: str = "label") -> Any:
        """Export to pandas DataFrame if pandas is installed (duck-typed)."""
        try:
            import pandas as pd
            return pd.DataFrame(self.to_dicts(text_col=text_col, label_col=label_col))
        except ImportError:
            raise ImportError("pandas is required for to_dataframe(). Install with 'pip install pandas'.")

    def summary(self) -> str:
        lines = [
            f"[EXPERIMENTAL] Augmentation Summary (seed={self.seed}):",
            f"  Original rows:       {self.n_original}",
            f"  Augmented variants:  {self.n_augmented}",
            f"  Total output rows:   {self.total_rows}",
            f"  Skipped unsafe/amb:  {self.skipped_candidates}",
            "  Augmentation by transformation kind:",
        ]
        if self.by_kind:
            for k, count in sorted(self.by_kind.items()):
                lines.append(f"    - {k:15}: {count}")
        else:
            lines.append("    (none)")
        lines.append("Note: Augment training splits only. Never evaluate models on augmented test rows.")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_original": self.n_original,
            "n_augmented": self.n_augmented,
            "total_rows": self.total_rows,
            "by_kind": self.by_kind,
            "skipped_candidates": self.skipped_candidates,
            "seed": self.seed,
            "metadata": self.metadata,
        }


def augment(
    source: Any,
    n_per_example: int = 1,
    kinds: Optional[Sequence[str]] = None,
    seed: int = 0,
    rules: Optional[RuleSet] = None,
    preserve_originals: bool = True,
    text_col: str = "text",
    label_col: str = "label",
) -> AugmentationResult:
    """Generate training-time augmented examples using reviewed safe transformations.

    [EXPERIMENTAL API]:
    Evaluates whether candidate transformations preserve meaning and negation before adding them.

    Args:
        source: CSV path, CSV text, list of (text, label) tuples, or DataFrame.
        n_per_example: Target number of safe augmented variants per training example.
        kinds: Optional subset of transformation kinds (e.g. ['spelling', 'repetition']).
        seed: Random seed for deterministic generation.
        rules: Optional custom RuleSet (defaults to DEFAULT_RULES).
        preserve_originals: Whether to keep original training rows (default: True).
        text_col: Text column name if tabular input is supplied.
        label_col: Label column name if tabular input is supplied.

    Returns:
        AugmentationResult containing generated rows and safety audit metadata.
    """
    if n_per_example < 1:
        raise ValueError(f"n_per_example must be >= 1, got {n_per_example}")

    rules = rules or DEFAULT_RULES
    rows = load_dataset(source, text_col=text_col, label_col=label_col)

    detailed: List[AugmentedRow] = []
    kind_counts: Counter = Counter()
    skipped_count = 0
    augmented_count = 0

    for i, (orig_text, label) in enumerate(rows, start=1):
        if preserve_originals:
            detailed.append(
                AugmentedRow(
                    text=orig_text,
                    label=label,
                    source_row=i,
                    is_augmented=False,
                    transformation_kind=None,
                    original_text=orig_text,
                )
            )

        if not orig_text.strip():
            continue

        # Generate candidates deterministically
        candidates = perturb(orig_text, n=n_per_example * 2, kinds=kinds, seed=seed + i, rules=rules)
        added_for_row = 0

        for cand in candidates:
            if added_for_row >= n_per_example:
                break
            # Safety gate: verify meaning preservation and negation invariance
            if cand.text != orig_text and rules.is_safe_variant(orig_text, cand.text):
                detailed.append(
                    AugmentedRow(
                        text=cand.text,
                        label=label,
                        source_row=i,
                        is_augmented=True,
                        transformation_kind=cand.kind,
                        original_text=orig_text,
                    )
                )
                kind_counts[cand.kind] += 1
                augmented_count += 1
                added_for_row += 1
            else:
                skipped_count += 1

    return AugmentationResult(
        detailed_rows=detailed,
        n_original=len(rows),
        n_augmented=augmented_count,
        by_kind=dict(kind_counts),
        skipped_candidates=skipped_count,
        seed=seed,
        metadata={
            "kinds": list(kinds) if kinds else list(KINDS),
            "preserve_originals": preserve_originals,
            "rules_fingerprint": rules.fingerprint(),
        },
    )


def compare_augmentation(
    model_factory: Callable[[], Any],
    train_data: Any,
    test_data: Any,
    n_per_example: int = 1,
    test_variants: int = 5,
    seed: int = 0,
    rules: Optional[RuleSet] = None,
) -> Dict[str, Any]:
    """Empirically compare model performance trained with vs without FlipRate augmentation.

    Trains two independent models using `model_factory()`:
      1. Baseline model: trained on original `train_data`.
      2. Augmented model: trained on `augment(train_data)`.

    Both are evaluated on identical held-out `test_data` using `fr.test`.
    Returns measured comparison of accuracy, macro-F1, and flip rate.
    """
    from .evaluation import test

    train_rows = load_dataset(train_data)
    test_rows = load_dataset(test_data)
    rules = rules or DEFAULT_RULES

    # 1. Baseline model
    model_base = model_factory()
    X_train = [t for t, _ in train_rows]
    y_train = [l for _, l in train_rows]
    if hasattr(model_base, "fit"):
        model_base.fit(X_train, y_train)

    rep_base = test(model_base, test_rows, n_variants=test_variants, seed=seed, rules=rules)

    # 2. Augmented model
    aug_res = augment(train_data, n_per_example=n_per_example, seed=seed, rules=rules)
    aug_rows = aug_res.to_rows()

    model_aug = model_factory()
    X_aug = [t for t, _ in aug_rows]
    y_aug = [l for _, l in aug_rows]
    if hasattr(model_aug, "fit"):
        model_aug.fit(X_aug, y_aug)

    rep_aug = test(model_aug, test_rows, n_variants=test_variants, seed=seed, rules=rules)

    return {
        "baseline": {
            "train_size": len(train_rows),
            "accuracy_original": rep_base.accuracy_original,
            "accuracy_variants": rep_base.accuracy_variants,
            "macro_f1_variants": rep_base.macro_f1_variants,
            "flip_rate": rep_base.flip_rate,
        },
        "augmented": {
            "train_size": len(aug_rows),
            "accuracy_original": rep_aug.accuracy_original,
            "accuracy_variants": rep_aug.accuracy_variants,
            "macro_f1_variants": rep_aug.macro_f1_variants,
            "flip_rate": rep_aug.flip_rate,
        },
        "diff": {
            "accuracy_original_diff": rep_aug.accuracy_original - rep_base.accuracy_original,
            "accuracy_variants_diff": rep_aug.accuracy_variants - rep_base.accuracy_variants,
            "macro_f1_diff": rep_aug.macro_f1_variants - rep_base.macro_f1_variants,
            "flip_rate_reduction": rep_base.flip_rate - rep_aug.flip_rate,
        },
        "augmentation_summary": aug_res.to_dict() if hasattr(aug_res, "to_dict") else aug_res.summary(),
    }
