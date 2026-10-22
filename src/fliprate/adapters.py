"""Adapters that turn real models into FlipRate models (text -> label, with fast `predict_batch`).

Heavy libraries (transformers, scikit-learn) are imported lazily, only when you use the adapter.
"""
from __future__ import annotations

import re
import time
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


class HFTextClassifier:
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

    def predict_batch(self, texts: Sequence[str]) -> List[str]:
        texts = [self.preprocess(t) if self.preprocess else t for t in texts]
        outs = self._get()(texts, batch_size=self.batch_size, truncation=True, max_length=self.max_length)
        labels = []
        for o in outs:
            lab = o["label"]
            lab = self.label_map.get(lab, lab)
            labels.append(lab.lower() if self.lowercase_labels else lab)
        return labels

    def __call__(self, text: str) -> str:
        return self.predict_batch([text])[0]


class SklearnAdapter:
    """Wrap a fitted scikit-learn Pipeline (e.g. TfidfVectorizer + LogisticRegression)."""

    def __init__(self, pipeline: Any, name: str = "sklearn"):
        self.pipeline, self.name = pipeline, name

    def predict_batch(self, texts: Sequence[str]) -> List[Any]:
        out = self.pipeline.predict(list(texts))
        return out.tolist() if hasattr(out, "tolist") else list(out)

    def __call__(self, text: str):
        return self.predict_batch([text])[0]


def from_sklearn(pipeline: Any, name: str = "sklearn") -> SklearnAdapter:
    return SklearnAdapter(pipeline, name)


class LLMClassifier:
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

    def __call__(self, text: str) -> str:
        return self.parse(self.complete(self._prompt(text)))

    def predict_batch(self, texts: Sequence[str]) -> List[str]:
        return [self(t) for t in texts]


def llm_classifier(complete: Callable[[str], str], labels: Sequence[str], **kw) -> LLMClassifier:
    return LLMClassifier(complete, labels, **kw)


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
