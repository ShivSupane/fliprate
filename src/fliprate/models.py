"""The model layer: batching, caching, and `protect()`.

A "model" is any callable `text -> label`. If it also has `predict_batch(list_of_texts) -> list_of_labels`,
FlipRate uses that, which is what makes real transformer models fast enough to test.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

from .normalization import normalize
from .rules import RuleSet


def batch_predict(model: Callable, texts: Sequence[str]) -> List[Any]:
    fn = getattr(model, "predict_batch", None)
    if callable(fn):
        out = list(fn(list(texts)))
        if len(out) != len(texts):
            raise ValueError(f"predict_batch returned {len(out)} labels for {len(texts)} texts")
        return out
    return [model(t) for t in texts]


def model_name(model: Callable) -> str:
    return (getattr(model, "name", None) or getattr(model, "__qualname__", None)
            or type(model).__name__)


class Predictor:
    """Calls a model efficiently: de-duplicates texts, caches answers, sends batches.

    Caching assumes the model is deterministic for a given text (true for classifiers and for LLMs run at
    temperature 0). Pass cache=False to disable it.
    """

    def __init__(self, model: Callable, cache: bool = True, batch_size: int = 64, max_cache: int = 50_000):
        self.model, self.cache, self.batch_size, self.max_cache = model, cache, batch_size, max_cache
        self._cache: Dict[str, Any] = {}
        self.calls = 0          # unique texts actually sent to the model
        self.request_count = 0  # underlying batch/network requests executed
        self.cache_hits = 0
        self.total_lookups = 0

    @property
    def cache_hit_rate(self) -> float:
        return self.cache_hits / self.total_lookups if self.total_lookups > 0 else 0.0

    def invalidate_cache(self, reset_stats: bool = False) -> None:
        """Clear cache entries and optionally reset accounting counters."""
        self._cache.clear()
        if reset_stats:
            self.reset_stats()

    def reset_stats(self) -> None:
        """Reset cost counters without clearing cache entries."""
        self.calls = 0
        self.request_count = 0
        self.cache_hits = 0
        self.total_lookups = 0

    def predict(self, texts: Sequence[str]) -> List[Any]:
        self.total_lookups += len(texts)
        known: Dict[str, Any] = {}
        todo: List[str] = []
        for t in dict.fromkeys(texts):
            if self.cache and t in self._cache:
                known[t] = self._cache[t]
                self.cache_hits += 1
            else:
                todo.append(t)
        for i in range(0, len(todo), self.batch_size):
            chunk = todo[i:i + self.batch_size]
            self.request_count += 1
            for t, p in zip(chunk, batch_predict(self.model, chunk)):
                known[t] = p
                self.calls += 1
                if self.cache:
                    if len(self._cache) >= self.max_cache:
                        self._cache.clear()
                    self._cache[t] = p
        return [known[t] for t in texts]


class ProtectedModel:
    """`model` with conservative normalization in front of it. This is the production-facing wrapper."""

    def __init__(self, model: Callable, rules: Optional[RuleSet] = None, lowercase: bool = True):
        self.model, self.rules, self.lowercase = model, rules, lowercase
        self.name = f"protected({model_name(model)})"

    def _clean(self, text: str) -> str:
        return normalize(text, lowercase=self.lowercase, rules=self.rules).normalized

    def __call__(self, text: str):
        return self.model(self._clean(text))

    def predict_batch(self, texts: Sequence[str]) -> List[Any]:
        return batch_predict(self.model, [self._clean(t) for t in texts])


def protect(model: Callable, rules: Optional[RuleSet] = None, lowercase: bool = True) -> ProtectedModel:
    """Wrap `model` so every input is normalized first. Measure the effect with `fr.test(protect(model), data)`."""
    return ProtectedModel(model, rules, lowercase)
