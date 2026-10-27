"""Benchmarking and Quality Dashboard for NLP model robustness.

Provides multi-model comparisons, baseline tracking, before/after protection metrics,
and exportable markdown and HTML dashboards.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .dataset import check_leakage, find_near_duplicates, load_dataset
from .evaluation import Report, test
from .gating import detect_script
from .models import protect
from .rules import DEFAULT_RULES


@dataclass
class ModelBenchmarkScore:
    """Benchmark metrics for a single model."""
    name: str
    accuracy_original: float
    accuracy_variants: float
    macro_f1_original: float
    macro_f1_variants: float
    flip_rate: float
    flip_rate_ci: Tuple[float, float]
    protected_flip_rate: Optional[float] = None
    protected_accuracy: Optional[float] = None
    flip_rate_reduction: Optional[float] = None
    is_baseline: bool = False
    score_vs_baseline: Optional[float] = None  # flip rate difference vs baseline
    by_kind: Dict[str, Dict[str, float]] = field(default_factory=dict)
    model_calls: int = 0
    seconds: float = 0.0


def audit_dataset(
    dataset: Any,
    train_dataset: Optional[Any] = None,
    rules: Optional[Any] = None,
) -> Dict[str, Any]:
    """Audit evaluation and training datasets for duplicates, leakage, and script distribution."""
    eval_rows = load_dataset(dataset)
    labels = [l for _, l in eval_rows]
    label_counts = Counter(labels)

    dup_report = find_near_duplicates(eval_rows, rules=rules)
    scripts = Counter(detect_script(t) for t, _ in eval_rows)

    leak_report = None
    if train_dataset is not None:
        leak_report = check_leakage(train_dataset, eval_rows, rules=rules)

    return {
        "n_messages": len(eval_rows),
        "class_distribution": dict(label_counts),
        "script_distribution": dict(scripts),
        "exact_duplicates": len(dup_report.exact_duplicates),
        "near_duplicates": len(dup_report.near_duplicates),
        "label_conflicts": len(dup_report.conflicting_pairs),
        "train_test_exact_leaks": len(leak_report.exact_leaks) if leak_report else 0,
        "train_test_near_leaks": len(leak_report.near_leaks) if leak_report else 0,
        "leakage_detected": leak_report.has_leakage if leak_report else False,
    }


@dataclass
class BenchmarkSuite:
    """Collection of model benchmark results with dashboard rendering."""
    scores: Dict[str, ModelBenchmarkScore] = field(default_factory=dict)
    baseline_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dataset_audit: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "FlipRate — Model Robustness Benchmark Summary",
            "=" * 70,
        ]
        for name, s in self.scores.items():
            base_tag = " [BASELINE]" if s.is_baseline else ""
            lines.append(f"\nModel: {name}{base_tag}")
            lines.append(f"  Accuracy (Originals): {s.accuracy_original:.1%}")
            lines.append(f"  Accuracy (Variants):  {s.accuracy_variants:.1%}")
            lines.append(f"  Flip Rate:            {s.flip_rate:.1%} (95% CI [{s.flip_rate_ci[0]:.1%}, {s.flip_rate_ci[1]:.1%}])")
            if s.protected_flip_rate is not None:
                lines.append(f"  Protected Flip Rate:  {s.protected_flip_rate:.1%} (Reduction: {s.flip_rate_reduction:+.1%})")
            if s.score_vs_baseline is not None and not s.is_baseline:
                lines.append(f"  Flip Rate vs Baseline: {s.score_vs_baseline:+.1%}")
        lines.append("\n" + "=" * 70)
        return "\n".join(lines)

    def to_markdown(self) -> str:
        headers = [
            "| Model | Raw Flip Rate | Protected Flip Rate | Acc (Originals) | Acc (Variants) | vs Baseline |",
            "|---|---|---|---|---|---|",
        ]
        rows = []
        for name, s in self.scores.items():
            prot_str = f"{s.protected_flip_rate:.1%}" if s.protected_flip_rate is not None else "N/A"
            if s.is_baseline:
                vs_base = "Baseline"
            elif s.score_vs_baseline is not None:
                vs_base = f"{s.score_vs_baseline:+.1%}"
            else:
                vs_base = "-"
            model_label = f"**{name}**" if s.is_baseline else name
            rows.append(
                f"| {model_label} | {s.flip_rate:.1%} | {prot_str} | "
                f"{s.accuracy_original:.1%} | {s.accuracy_variants:.1%} | {vs_base} |"
            )
        return "\n".join(["### Robustness Benchmark Results", ""] + headers + rows)

    def to_html(self) -> str:
        """Render a standalone, modern HTML quality dashboard."""
        rows_html = []
        for name, s in self.scores.items():
            badge = "<span style='background:#2563eb;color:#fff;padding:2px 8px;border-radius:12px;font-size:12px;'>Baseline</span>" if s.is_baseline else ""
            prot_val = f"{s.protected_flip_rate:.1%}" if s.protected_flip_rate is not None else "—"
            flip_color = "#16a34a" if s.flip_rate < 0.05 else ("#d97706" if s.flip_rate < 0.15 else "#dc2626")
            rows_html.append(f"""
            <tr>
              <td style="padding:12px;font-weight:600;">{name} {badge}</td>
              <td style="padding:12px;color:{flip_color};font-weight:700;">{s.flip_rate:.1%}</td>
              <td style="padding:12px;color:#16a34a;font-weight:600;">{prot_val}</td>
              <td style="padding:12px;">{s.accuracy_original:.1%}</td>
              <td style="padding:12px;">{s.accuracy_variants:.1%}</td>
            </tr>
            """)

        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>FlipRate Quality Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; padding: 30px; }}
    .card {{ background: #1e293b; border-radius: 12px; padding: 24px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); max-width: 900px; margin: 0 auto; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 16px; }}
    th {{ text-align: left; padding: 12px; border-bottom: 2px solid #334155; color: #94a3b8; font-size: 13px; text-transform: uppercase; }}
    tr:not(:last-child) td {{ border-bottom: 1px solid #334155; }}
  </style>
</head>
<body>
  <div class="card">
    <h2 style="margin-top:0;">FlipRate Robustness Dashboard</h2>
    <p style="color:#94a3b8;font-size:14px;">Comparative robustness evaluation across models and safety normalization.</p>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          <th>Raw Flip Rate</th>
          <th>Protected Flip Rate</th>
          <th>Acc (Originals)</th>
          <th>Acc (Variants)</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows_html)}
      </tbody>
    </table>
  </div>
</body>
</html>"""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline_name,
            "metadata": self.metadata,
            "dataset_audit": self.dataset_audit,
            "scores": {
                name: {
                    "accuracy_original": s.accuracy_original,
                    "accuracy_variants": s.accuracy_variants,
                    "macro_f1_original": s.macro_f1_original,
                    "macro_f1_variants": s.macro_f1_variants,
                    "flip_rate": s.flip_rate,
                    "flip_rate_ci": s.flip_rate_ci,
                    "protected_flip_rate": s.protected_flip_rate,
                    "protected_accuracy": s.protected_accuracy,
                    "flip_rate_reduction": s.flip_rate_reduction,
                    "is_baseline": s.is_baseline,
                    "score_vs_baseline": s.score_vs_baseline,
                    "by_kind": s.by_kind,
                    "model_calls": s.model_calls,
                    "seconds": s.seconds,
                }
                for name, s in self.scores.items()
            }
        }

    def save(self, path: str | Path) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix.lower()
        if ext == ".html":
            out_path.write_text(self.to_html(), encoding="utf-8")
        elif ext == ".md":
            out_path.write_text(self.to_markdown(), encoding="utf-8")
        else:
            out_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return out_path


def benchmark(
    models: Dict[str, Any],
    dataset: Any,
    baseline: Optional[str] = None,
    with_protection: bool = True,
    n_variants: int = 5,
    bootstrap: int = 200,
    train_dataset: Optional[Any] = None,
    audit: bool = True,
    protected_terms: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> BenchmarkSuite:
    """Benchmark multiple models side-by-side on a test dataset."""
    rows = load_dataset(dataset)
    audit_data = audit_dataset(rows, train_dataset=train_dataset) if audit else {}
    scores: Dict[str, ModelBenchmarkScore] = {}

    baseline_model_name = baseline if (baseline and baseline in models) else (list(models.keys())[0] if models else None)
    baseline_flip_rate: Optional[float] = None

    for name, model in models.items():
        # Evaluate raw model
        rep = test(
            model,
            rows,
            n_variants=n_variants,
            bootstrap=bootstrap,
            protected_terms=protected_terms,
            **kwargs,
        )

        prot_flip: Optional[float] = None
        prot_acc: Optional[float] = None
        reduction: Optional[float] = None

        if with_protection:
            safe_model = protect(model)
            rep_prot = test(
                safe_model,
                rows,
                n_variants=n_variants,
                bootstrap=bootstrap,
                protected_terms=protected_terms,
                **kwargs,
            )
            prot_flip = rep_prot.flip_rate
            prot_acc = rep_prot.accuracy_variants
            reduction = rep.flip_rate - rep_prot.flip_rate

        is_base = (name == baseline_model_name)
        if is_base:
            baseline_flip_rate = rep.flip_rate

        scores[name] = ModelBenchmarkScore(
            name=name,
            accuracy_original=rep.accuracy_original,
            accuracy_variants=rep.accuracy_variants,
            macro_f1_original=rep.macro_f1_original,
            macro_f1_variants=rep.macro_f1_variants,
            flip_rate=rep.flip_rate,
            flip_rate_ci=rep.flip_rate_ci,
            protected_flip_rate=prot_flip,
            protected_accuracy=prot_acc,
            flip_rate_reduction=reduction,
            is_baseline=is_base,
            score_vs_baseline=None,
            by_kind=rep.by_kind,
            model_calls=rep.metadata.get("model_calls", 0),
            seconds=rep.metadata.get("seconds", 0.0),
        )

    # Compute score vs baseline
    if baseline_flip_rate is not None:
        for name, score in scores.items():
            score.score_vs_baseline = score.flip_rate - baseline_flip_rate

    return BenchmarkSuite(
        scores=scores,
        baseline_name=baseline_model_name,
        metadata={"n_models": len(models), "n_messages": len(rows), "n_variants": n_variants},
        dataset_audit=audit_data,
    )


def run_benchmark_protocol(
    models: Dict[str, Any],
    dataset: Any,
    train_dataset: Optional[Any] = None,
    baseline: Optional[str] = None,
    with_protection: bool = True,
    protected_terms: Optional[Sequence[str]] = None,
    **kwargs: Any,
) -> BenchmarkSuite:
    """Execute a rigorous, reproducible benchmark protocol on held-out natural text.

    Audits train/test leakage, near duplicates, and script distribution, and compares
    models raw vs protected with 95% Wilson confidence intervals.
    """
    return benchmark(
        models=models,
        dataset=dataset,
        baseline=baseline,
        with_protection=with_protection,
        train_dataset=train_dataset,
        audit=True,
        protected_terms=protected_terms,
        **kwargs,
    )


def compare_models(model_a: Any, model_b: Any, dataset: Any,
                   name_a: str = "model_a", name_b: str = "model_b", **kwargs) -> BenchmarkSuite:
    """Convenience helper to benchmark two models side-by-side."""
    return benchmark({name_a: model_a, name_b: model_b}, dataset, baseline=name_a, **kwargs)
