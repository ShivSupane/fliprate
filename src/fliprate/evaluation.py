"""Robustness testing: how often does a model change its answer on safe variants?"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import random
import time
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Sequence, Tuple

from .events import Event, drain, make_event
from .models import Predictor, model_name
from .perturbation import KINDS, perturb
from .rules import DEFAULT_RULES, RuleSet

Model = Callable[[str], Any]
Row = Tuple[str, Any]


# ----------------------------------------------------------------------------- data
def read_csv_text(content: str) -> List[Row]:
    """Parse CSV text with header columns `text` and `label`."""
    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames or "text" not in reader.fieldnames or "label" not in reader.fieldnames:
        raise ValueError("CSV needs a header row with columns: text,label")
    return [(r["text"], r["label"]) for r in reader if (r.get("text") or "").strip()]


def load_dataset(source: Any) -> List[Row]:
    """Accept a CSV path, or a list of (text, label) pairs, or a list of {text, label} dicts."""
    if isinstance(source, (str, os.PathLike)):
        with open(source, encoding="utf-8", newline="") as f:
            rows = read_csv_text(f.read())
    else:
        rows = []
        for r in source:
            rows.append((r["text"], r["label"]) if isinstance(r, dict) else (r[0], r[1]))
    if not rows:
        raise ValueError("Dataset is empty")
    return rows


# --------------------------------------------------------------------------- metrics
def accuracy(y_true: Sequence, y_pred: Sequence) -> float:
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0


def macro_f1(y_true: Sequence, y_pred: Sequence) -> float:
    labels = sorted(set(y_true) | set(y_pred), key=str)
    if not labels:
        return 0.0
    scores = []
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(y_true, y_pred))
        fp = sum(t != lab and p == lab for t, p in zip(y_true, y_pred))
        fn = sum(t == lab and p != lab for t, p in zip(y_true, y_pred))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom else 0.0)
    return sum(scores) / len(scores)


def _bootstrap_ci(summaries: List[Tuple[int, int, int, int]], stat: Callable, boots: int, seed: int):
    n = len(summaries)
    if n < 2:
        v = stat(summaries)
        return (v, v)
    rng = random.Random(seed)
    vals = sorted(stat([summaries[rng.randrange(n)] for _ in range(n)]) for _ in range(boots))
    return (vals[int(0.025 * boots)], vals[max(int(0.975 * boots) - 1, 0)])


# summaries are per message: (orig_correct, n_variants, n_flips, n_variants_correct)
def _flip_stat(s):
    nv = sum(x[1] for x in s)
    return sum(x[2] for x in s) / nv if nv else 0.0


def _drop_stat(s):
    nv = sum(x[1] for x in s)
    acc_o = sum(x[0] for x in s) / len(s)
    return acc_o - (sum(x[3] for x in s) / nv if nv else acc_o)


# ---------------------------------------------------------------------------- report
@dataclass
class Report:
    n_messages: int
    n_variants: int
    accuracy_original: float
    macro_f1_original: float
    accuracy_variants: float
    macro_f1_variants: float
    flip_rate: float
    flip_rate_ci: Tuple[float, float]
    accuracy_drop: float
    accuracy_drop_ci: Tuple[float, float]
    by_kind: Dict[str, Dict[str, float]]
    flips: List[Dict[str, Any]] = field(default_factory=list)
    seed: int = 0
    transitions: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    message_stats: List[List[int]] = field(default_factory=list)  # per message: [orig_correct, n_variants, n_flips, n_variants_correct]

    # convenient aliases so `report.accuracy` / `report.macro_f1` just work
    @property
    def accuracy(self) -> float:
        return self.accuracy_original

    @property
    def macro_f1(self) -> float:
        return self.macro_f1_original

    def summary(self) -> str:
        worst = max(self.by_kind.items(), key=lambda kv: kv[1]["flip_rate"], default=None)
        lines = [
            f"Messages: {self.n_messages} | safe variants tested: {self.n_variants}",
            f"Accuracy on originals:      {self.accuracy_original:.1%}  (macro-F1 {self.macro_f1_original:.2f})",
            f"Accuracy on safe variants:  {self.accuracy_variants:.1%}  (macro-F1 {self.macro_f1_variants:.2f})",
            f"Accuracy drop:              {self.accuracy_drop:+.1%}  95% CI [{self.accuracy_drop_ci[0]:+.1%}, {self.accuracy_drop_ci[1]:+.1%}]",
            f"Flip rate:                  {self.flip_rate:.1%}  95% CI [{self.flip_rate_ci[0]:.1%}, {self.flip_rate_ci[1]:.1%}]",
        ]
        if worst:
            lines.append(f"Most fragile change type:   {worst[0]} (flip rate {worst[1]['flip_rate']:.1%})")
        lines.append("Note: flip rate shows instability, not correctness. A model can flip to the right answer.")
        return "\n".join(lines)

    def to_dict(self, max_flips: Optional[int] = 100) -> Dict[str, Any]:
        return {
            "n_messages": self.n_messages,
            "n_variants": self.n_variants,
            "accuracy_original": self.accuracy_original,
            "macro_f1_original": self.macro_f1_original,
            "accuracy_variants": self.accuracy_variants,
            "macro_f1_variants": self.macro_f1_variants,
            "flip_rate": self.flip_rate,
            "flip_rate_ci": list(self.flip_rate_ci),
            "accuracy_drop": self.accuracy_drop,
            "accuracy_drop_ci": list(self.accuracy_drop_ci),
            "by_kind": self.by_kind,
            "flips": self.flips if max_flips is None else self.flips[:max_flips],
            "seed": self.seed,
            "transitions": self.transitions,
            "metadata": self.metadata,
            "message_stats": self.message_stats,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Report":
        d = dict(d)
        d["flip_rate_ci"] = tuple(d["flip_rate_ci"])
        d["accuracy_drop_ci"] = tuple(d["accuracy_drop_ci"])
        return cls(**d)

    def save(self, path, max_flips: Optional[int] = 1000) -> None:
        """Write the report (with run metadata) as JSON so the experiment can be reproduced and audited."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(max_flips), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "Report":
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_markdown(self, title: str = "FlipRate report") -> str:
        lines = [f"### {title}", "", "| Metric | Value |", "|---|---|",
                 f"| Messages / safe variants | {self.n_messages} / {self.n_variants} |",
                 f"| Accuracy (originals) | {self.accuracy_original:.1%} |",
                 f"| Accuracy (safe variants) | {self.accuracy_variants:.1%} |",
                 f"| Macro-F1 (variants) | {self.macro_f1_variants:.3f} |",
                 f"| Flip rate | {self.flip_rate:.1%} (95% CI {self.flip_rate_ci[0]:.1%} to {self.flip_rate_ci[1]:.1%}) |",
                 "", "| Change type | Variants | Flip rate | Accuracy |", "|---|---|---|---|"]
        for k, v in sorted(self.by_kind.items()):
            lines.append(f"| {k} | {int(v['variants'])} | {v['flip_rate']:.1%} | {v['accuracy']:.1%} |")
        return "\n".join(lines)


def paired_difference(a: "Report", b: "Report", stat: str = "flip_rate", boots: int = 2000, seed: int = 0) -> Dict[str, Any]:
    """Paired comparison of two runs on the SAME messages (e.g. raw vs protected): value(b) - value(a),
    with a 95% bootstrap interval that resamples messages jointly. stat: flip_rate | accuracy_original | accuracy_variants.
    Much tighter and more honest than comparing two separate intervals."""
    if not a.message_stats or len(a.message_stats) != len(b.message_stats):
        raise ValueError("Reports must come from the same dataset and carry message_stats")
    if a.metadata.get("dataset_sha256_16") != b.metadata.get("dataset_sha256_16") or a.seed != b.seed:
        raise ValueError("Reports were produced on different datasets or seeds")

    def value(rows):
        n = len(rows)
        nv = sum(r[1] for r in rows)
        if stat == "flip_rate":
            return sum(r[2] for r in rows) / nv if nv else 0.0
        if stat == "accuracy_original":
            return sum(r[0] for r in rows) / n
        if stat == "accuracy_variants":
            return sum(r[3] for r in rows) / nv if nv else 0.0
        raise ValueError("stat must be flip_rate, accuracy_original or accuracy_variants")

    n = len(a.message_stats)
    point = value(b.message_stats) - value(a.message_stats)
    rng = random.Random(seed)
    diffs = []
    for _ in range(boots):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(value([b.message_stats[i] for i in idx]) - value([a.message_stats[i] for i in idx]))
    diffs.sort()
    lo, hi = diffs[int(0.025 * boots)], diffs[max(int(0.975 * boots) - 1, 0)]
    return {"stat": stat, "difference": point, "ci": (lo, hi), "significant": not (lo <= 0 <= hi)}


# ------------------------------------------------------------------------ the engine
def _dataset_sha(rows: List[Row]) -> str:
    h = hashlib.sha256()
    for t, l in rows:
        h.update(f"{t}\t{l}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def iter_test(
    model: Model,
    dataset: Any,
    n_variants: int = 5,
    kinds: Optional[Sequence[str]] = None,
    seed: int = 0,
    bootstrap: int = 1000,
    rules: Optional[RuleSet] = None,
    cache: bool = True,
    batch_messages: int = 8,
) -> Generator[Event, None, Report]:
    """Run a robustness test and yield progress events; the generator's return value is the Report.

    Messages are processed `batch_messages` at a time so batch-capable models (see models.py) stay fast.
    """
    rules = rules or DEFAULT_RULES
    rows = load_dataset(dataset)
    predictor = Predictor(model, cache=cache)
    t0 = time.time()
    yield make_event("started", total=len(rows), n_variants=n_variants)

    y_true, y_orig = [], []
    v_true, v_pred = [], []
    summaries: List[Tuple[int, int, int, int]] = []
    flips: List[Dict[str, Any]] = []
    transitions: Counter = Counter()
    kind_tot: Dict[str, List[int]] = {}  # kind -> [variants, flips, correct]
    total_v = total_f = 0

    for start in range(0, len(rows), max(1, batch_messages)):
        plans, flat = [], []
        for j, (text, label) in enumerate(rows[start:start + max(1, batch_messages)]):
            i = start + j
            vs = perturb(text, n_variants, kinds, seed + i, rules=rules)
            plans.append((i, text, label, vs, len(flat)))
            flat.append(text)
            flat.extend(v.text for v in vs)
        preds = predictor.predict(flat)

        for i, text, label, vs, off in plans:
            base = preds[off]
            y_true.append(label)
            y_orig.append(base)
            nflip = ncorrect = 0
            for k, v in enumerate(vs):
                pred = preds[off + 1 + k]
                v_true.append(label)
                v_pred.append(pred)
                stat = kind_tot.setdefault(v.kind, [0, 0, 0])
                stat[0] += 1
                if pred == label:
                    ncorrect += 1
                    stat[2] += 1
                if pred != base:
                    nflip += 1
                    stat[1] += 1
                    transitions[f"{base} -> {pred}"] += 1
                    flips.append({"original": text, "variant": v.text, "kind": v.kind,
                                  "original_pred": base, "variant_pred": pred, "label": label})
            summaries.append((int(base == label), len(vs), nflip, ncorrect))
            total_v += len(vs)
            total_f += nflip
            yield make_event(
                "progress", done=i + 1, total=len(rows),
                flip_rate=(total_f / total_v) if total_v else 0.0,
                accuracy=accuracy(y_true, y_orig),
                last={"text": text, "variants": len(vs), "flips": nflip},
            )

    by_kind = {
        k: {"variants": s_[0], "flip_rate": s_[1] / s_[0] if s_[0] else 0.0, "accuracy": s_[2] / s_[0] if s_[0] else 0.0}
        for k, s_ in kind_tot.items()
    }
    acc_o = accuracy(y_true, y_orig)
    acc_v = accuracy(v_true, v_pred) if v_pred else acc_o
    from . import __version__
    report = Report(
        n_messages=len(rows),
        n_variants=total_v,
        accuracy_original=acc_o,
        macro_f1_original=macro_f1(y_true, y_orig),
        accuracy_variants=acc_v,
        macro_f1_variants=macro_f1(v_true, v_pred) if v_pred else macro_f1(y_true, y_orig),
        flip_rate=_flip_stat(summaries),
        flip_rate_ci=_bootstrap_ci(summaries, _flip_stat, bootstrap, seed),
        accuracy_drop=acc_o - acc_v,
        accuracy_drop_ci=_bootstrap_ci(summaries, _drop_stat, bootstrap, seed),
        by_kind=by_kind,
        flips=flips,
        seed=seed,
        transitions=dict(transitions.most_common(20)),
        message_stats=[list(x) for x in summaries],
        metadata={
            "fliprate_version": __version__,
            "python": platform.python_version(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model_name(model),
            "dataset_sha256_16": _dataset_sha(rows),
            "rules_fingerprint": rules.fingerprint(),
            "variants_per_message": n_variants,
            "kinds": list(kinds) if kinds else list(KINDS),
            "bootstrap_resamples": bootstrap,
            "model_calls": predictor.calls,
            "cache_hits": predictor.cache_hits,
            "seconds": round(time.time() - t0, 2),
        },
    )
    yield make_event("done", report=report.to_dict())
    return report


def test(model: Model, dataset: Any, n_variants: int = 5, kinds: Optional[Sequence[str]] = None,
         seed: int = 0, bootstrap: int = 1000, on_event: Optional[Callable[[Event], None]] = None,
         rules: Optional[RuleSet] = None, cache: bool = True, batch_messages: int = 8) -> Report:
    """Test `model` (any function text -> label) on `dataset` and return a Report."""
    return drain(iter_test(model, dataset, n_variants, kinds, seed, bootstrap, rules, cache, batch_messages), on_event)


test.__test__ = False  # stop pytest from mistaking this function for a test
