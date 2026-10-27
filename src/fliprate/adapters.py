"""Adapters that turn real models into FlipRate models (text -> label, with fast `predict_batch`).

Heavy libraries (transformers, scikit-learn) are imported lazily, only when you use the adapter.
"""
from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence


def twitter_preprocess(text: str) -> str:
    """The @user / http placeholders that the cardiffnlp Twitter models were trained with."""
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")
    out = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        out.append(t)
    return " ".join(out)


class ModelAdapter(ABC):
    """Standardized adapter interface for all models evaluated or guarded by FlipRate.

    Subclasses must implement `predict(text)`. Subclasses can optionally override
    `predict_batch(texts)` for high-throughput batching.
    """

    @abstractmethod
    def predict(self, text: str) -> str:
        """Predict the label for a single input text string."""
        pass

    def predict_batch(self, texts: Sequence[str]) -> List[str]:
        """Predict labels for a batch of input text strings. Default loops over `predict`."""
        return [self.predict(t) for t in texts]

    def metadata(self) -> Dict[str, Any]:
        """Return diagnostic metadata about the adapter."""
        return {
            "adapter_type": self.__class__.__name__,
            "name": getattr(self, "name", self.__class__.__name__),
        }

    def supported_languages(self) -> List[str]:
        """Return list of supported languages or scripts, if known."""
        return ["any"]

    def __call__(self, text: str) -> str:
        """Allow calling the adapter instance directly like a regular Python function."""
        return self.predict(text)


class CallableAdapter(ModelAdapter):
    """Wrap any standard Python function or lambda `fn(text: str) -> str` as a ModelAdapter."""

    def __init__(self, fn: Callable[[str], str], name: Optional[str] = None):
        self.fn = fn
        self.name = name or getattr(fn, "__name__", "callable")

    def predict(self, text: str) -> str:
        return self.fn(text)

    def predict_batch(self, texts: Sequence[str]) -> List[str]:
        if hasattr(self.fn, "predict_batch"):
            return self.fn.predict_batch(texts)
        return [self.fn(t) for t in texts]


def as_adapter(model: Any) -> ModelAdapter:
    """Ensure `model` conforms to ModelAdapter; wraps raw callables if needed."""
    if isinstance(model, ModelAdapter):
        return model
    if callable(model):
        return CallableAdapter(model)
    raise TypeError(f"Expected ModelAdapter or callable, got {type(model).__name__}")


class HFTextClassifier(ModelAdapter):
    """Hugging Face text-classification model, batched.

        clf = HFTextClassifier("cardiffnlp/twitter-xlm-roberta-base-sentiment",
                               lowercase_labels=True, preprocess=twitter_preprocess)
        fr.test(clf, rows)

    `label_map` renames model labels to your dataset's labels. `lowercase_labels=True` maps
    "Positive" -> "positive". `pipeline_factory` lets tests inject a fake pipeline.
    """

    def __init__(self, model_name: str, device: Any = None, batch_size: int = 32, max_length: int = 256,
                 label_map: Optional[Dict[str, str]] = None, lowercase_labels: bool = False,
                 preprocess: Optional[Callable[[str], str]] = None,
                 pipeline_factory: Optional[Callable[[], Callable]] = None):
        self.name = model_name
        self.device, self.batch_size, self.max_length = device, batch_size, max_length
        self.label_map, self.lowercase_labels, self.preprocess = label_map or {}, lowercase_labels, preprocess
        self._factory = pipeline_factory
        self._pipe: Optional[Callable] = None

    def _get(self) -> Callable:
        if self._pipe is None:
            if self._factory:
                self._pipe = self._factory()
            else:
                try:
                    from transformers import pipeline
                except ImportError as exc:
                    raise ImportError("HFTextClassifier needs `pip install transformers torch`") from exc
                kwargs = {"model": self.name, "tokenizer": self.name}
                if self.device is not None:
                    kwargs["device"] = self.device
                self._pipe = pipeline("text-classification", **kwargs)
        return self._pipe

    def warmup(self) -> None:
        self.predict_batch(["warmup"])

    def predict(self, text: str) -> str:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> List[str]:
        texts = [self.preprocess(t) if self.preprocess else t for t in texts]
        outs = self._get()(texts, batch_size=self.batch_size, truncation=True, max_length=self.max_length)
        labels = []
        for o in outs:
            lab = o["label"]
            lab = self.label_map.get(lab, lab)
            labels.append(lab.lower() if self.lowercase_labels else lab)
        return labels

    def metadata(self) -> Dict[str, Any]:
        meta = super().metadata()
        meta.update({
            "model_name": self.name,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        })
        return meta


class SklearnAdapter(ModelAdapter):
    """Wrap a fitted scikit-learn Pipeline (e.g. TfidfVectorizer + LogisticRegression)."""

    def __init__(self, pipeline: Any, name: str = "sklearn"):
        self.pipeline, self.name = pipeline, name

    def predict(self, text: str) -> Any:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> List[Any]:
        out = self.pipeline.predict(list(texts))
        return out.tolist() if hasattr(out, "tolist") else list(out)


def from_sklearn(pipeline: Any, name: str = "sklearn") -> SklearnAdapter:
    return SklearnAdapter(pipeline, name)


try:
    from sklearn.base import BaseEstimator, TransformerMixin
    _SKLEARN_BASE_CLASSES = (BaseEstimator, TransformerMixin)
except ImportError:
    class _FallbackBaseEstimator:
        def get_params(self, deep: bool = True) -> Dict[str, Any]:
            return {
                "rules": getattr(self, "rules", None),
                "lowercase": getattr(self, "lowercase", True),
            }

        def set_params(self, **params: Any) -> Any:
            for k, v in params.items():
                setattr(self, k, v)
            return self

    class _FallbackTransformerMixin:
        def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> Any:
            return self.fit(X, y, **fit_params).transform(X)

    _SKLEARN_BASE_CLASSES = (_FallbackBaseEstimator, _FallbackTransformerMixin)


class FlipRateNormalizer(*_SKLEARN_BASE_CLASSES):
    """Scikit-learn compatible transformer for text normalization.

    Can be placed directly in a scikit-learn Pipeline ahead of a vectorizer:
        from fliprate import FlipRateNormalizer
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        pipe = Pipeline([
            ('normalizer', FlipRateNormalizer()),
            ('tfidf', TfidfVectorizer()),
            ('clf', LogisticRegression()),
        ])

    Note: Normalization can alter model behavior and does not guarantee correctness.
    Always evaluate downstream task impact using FlipRate evaluation tools.
    """

    def __init__(self, rules: Optional[Any] = None, lowercase: bool = True):
        self.rules = rules
        self.lowercase = lowercase

    def fit(self, X: Any, y: Any = None) -> "FlipRateNormalizer":
        """No-op fit method: does not learn from data or held-out labels."""
        return self

    def transform(self, X: Any) -> List[str]:
        """Normalize all input text items preserving order and row count."""
        from .normalization import normalize
        # Handle numpy arrays, pandas Series, or iterables
        if hasattr(X, "tolist"):
            items = X.tolist()
        elif hasattr(X, "values") and hasattr(X.values, "tolist"):
            items = X.values.tolist()
        else:
            items = list(X)

        return [
            normalize(str(item) if not isinstance(item, str) else item,
                      rules=self.rules, lowercase=self.lowercase).normalized
            for item in items
        ]

    def fit_transform(self, X: Any, y: Any = None, **fit_params: Any) -> List[str]:
        return self.fit(X, y, **fit_params).transform(X)


class LLMClassifier(ModelAdapter):
    """Classify with any LLM through a function `complete(prompt) -> str` that you provide.

    The reply is accepted only if it clearly names exactly one allowed label; otherwise it is recorded as
    `invalid_label`. Use temperature 0 so results are repeatable. `n_invalid` counts unusable replies.
    """

    def __init__(self, complete: Callable[[str], str], labels: Sequence[str], instruction: Optional[str] = None,
                 name: str = "llm", invalid_label: str = "__invalid__"):
        self.complete, self.labels, self.name, self.invalid_label = complete, list(labels), name, invalid_label
        self.instruction = instruction or "Classify the customer message into exactly one label."
        self.n_invalid = 0

    def _prompt(self, text: str) -> str:
        return (f"{self.instruction}\nAllowed labels: {', '.join(self.labels)}\n"
                f"Message: {text}\nAnswer with the label only.")

    def parse(self, reply: str) -> str:
        reply = (reply or "").strip().lower()
        if reply in self.labels:
            return reply
        found = [l for l in self.labels if re.search(rf"(?<![a-z_]){re.escape(l.lower())}(?![a-z_])", reply)]
        if len(found) == 1:
            return found[0]
        self.n_invalid += 1
        return self.invalid_label

    def predict(self, text: str) -> str:
        return self.parse(self.complete(self._prompt(text)))

    def predict_batch(self, texts: Sequence[str]) -> List[str]:
        return [self.predict(t) for t in texts]


def llm_classifier(complete: Callable[[str], str], labels: Sequence[str], **kw) -> LLMClassifier:
    return LLMClassifier(complete, labels, **kw)


class HttpServiceAdapter(ModelAdapter):
    """Adapter for REST/HTTP model endpoints with built-in timeouts and retries.

    Sends POST requests to `endpoint_url` with payload `{"text": text}` or batch payload `{"texts": texts}`.
    Expects JSON response `{"prediction": label}` or `{"predictions": [label1, ...]}`.
    """

    def __init__(self, endpoint_url: str, headers: Optional[Dict[str, str]] = None,
                 timeout: float = 5.0, retries: int = 2, name: str = "http_service",
                 parse_response: Optional[Callable[[Dict[str, Any]], str]] = None):
        self.endpoint_url = endpoint_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self.retries = retries
        self.name = name
        self.parse_response = parse_response

    def predict(self, text: str) -> str:
        import json
        import urllib.request
        import urllib.error

        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(self.endpoint_url, data=payload, headers=self.headers, method="POST")

        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                    if self.parse_response:
                        return self.parse_response(data)
                    if "prediction" in data:
                        return str(data["prediction"])
                    if "label" in data:
                        return str(data["label"])
                    raise ValueError(f"Unrecognized response format from {self.endpoint_url}: {data}")
            except Exception as exc:
                if attempt == self.retries:
                    raise RuntimeError(f"HttpServiceAdapter failed after {self.retries + 1} attempts: {exc}") from exc
                time.sleep(0.3 * (2 ** attempt))
        return ""

    def metadata(self) -> Dict[str, Any]:
        meta = super().metadata()
        meta.update({
            "endpoint_url": self.endpoint_url,
            "timeout": self.timeout,
            "retries": self.retries,
        })
        return meta


def with_retries(fn: Callable, retries: int = 3, backoff: float = 0.5,
                 exceptions: tuple = (Exception,)) -> Callable:
    """Retry flaky calls (network/LLM APIs) with exponential backoff; the last error is re-raised."""
    def wrapped(*args, **kwargs):
        for attempt in range(retries + 1):
            try:
                return fn(*args, **kwargs)
            except exceptions:
                if attempt == retries:
                    raise
                time.sleep(backoff * (2 ** attempt))
    wrapped.__name__ = getattr(fn, "__name__", "wrapped")
    return wrapped
