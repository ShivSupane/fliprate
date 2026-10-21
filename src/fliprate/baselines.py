"""Small real classifiers you can train in seconds, to compare against transformers.

Character n-gram models are a natural robustness baseline: a stretched or misspelled word still shares
most of its character n-grams with the original.
"""
from __future__ import annotations

from typing import Optional, Sequence

from .adapters import SklearnAdapter
from .evaluation import Row


def train_tfidf(rows: Sequence[Row], analyzer: str = "word", name: Optional[str] = None) -> SklearnAdapter:
    """Train TF-IDF + logistic regression. analyzer: 'word' or 'char_wb' (character n-grams)."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:
        raise ImportError("train_tfidf needs `pip install scikit-learn`") from exc
    if analyzer not in ("word", "char_wb"):
        raise ValueError("analyzer must be 'word' or 'char_wb'")
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=(1, 2) if analyzer == "word" else (2, 5),
                          lowercase=True, sublinear_tf=True)
    pipe = make_pipeline(vec, LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced"))
    pipe.fit([t for t, _ in rows], [l for _, l in rows])
    return SklearnAdapter(pipe, name or f"tfidf-{analyzer}")
