"""One-call experiment: several models x (raw, protected), on the same data, with the same seed."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .evaluation import Report, load_dataset, paired_difference, test
from .models import protect
from .rules import RuleSet


@dataclass
class ExperimentResult:
    reports: Dict[str, Dict[str, Report]]
    config: Dict[str, Any] = field(default_factory=dict)

    def markdown(self) -> str:
        head = ["| Model | Mode | Acc (originals) | Acc (variants) | Macro-F1 (variants) | Flip rate (95% CI) |",
                "|---|---|---|---|---|---|"]
        rows = []
        for name, modes in self.reports.items():
            for mode, r in modes.items():
                rows.append(f"| {name} | {mode} | {r.accuracy_original:.1%} | {r.accuracy_variants:.1%} | "
                            f"{r.macro_f1_variants:.3f} | {r.flip_rate:.1%} ({r.flip_rate_ci[0]:.1%} to {r.flip_rate_ci[1]:.1%}) |")
        out = ["### Robustness by model", ""] + head + rows
        prot = [(n, m["raw"], m["protected"]) for n, m in self.reports.items() if "protected" in m]
        if prot:
            out += ["", "### Effect of protection (normalization in front of the model), paired on the same messages", "",
                    "| Model | Flip rate raw -> protected | Change in flip rate (paired 95% CI) | Change in accuracy, originals (paired 95% CI) |",
                    "|---|---|---|---|"]
            for n, raw, p in prot:
                fl = paired_difference(raw, p, "flip_rate")
                ac = paired_difference(raw, p, "accuracy_original")
                out.append(f"| {n} | {raw.flip_rate:.1%} -> {p.flip_rate:.1%} | {fl['difference']*100:+.2f} pts [{fl['ci'][0]*100:+.2f}, {fl['ci'][1]*100:+.2f}] "
                           f"| {ac['difference']*100:+.2f} pts [{ac['ci'][0]*100:+.2f}, {ac['ci'][1]*100:+.2f}] |")
        c = self.config
        out += ["", f"messages={c.get('n_messages')} variants/message={c.get('n_variants')} seed={c.get('seed')} "
                    f"rules={c.get('rules_fingerprint')} dataset={c.get('dataset_sha256_16')}"]
        return "\n".join(out)

    def save(self, out_dir) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, modes in self.reports.items():
            for mode, r in modes.items():
                safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
                r.save(out / f"{safe}__{mode}.json")
        (out / "summary.md").write_text(self.markdown(), encoding="utf-8")
        (out / "config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        return out


def run_experiment(models: Dict[str, Callable], dataset: Any, n_variants: int = 5, seed: int = 0,
                   rules: Optional[RuleSet] = None, with_protection: bool = True, bootstrap: int = 1000,
                   on_progress: Optional[Callable[[str, str, Dict[str, Any]], None]] = None) -> ExperimentResult:
    """Run every model on identical data and seeds, raw and (optionally) wrapped with `protect()`."""
    rows = load_dataset(dataset)
    reports: Dict[str, Dict[str, Report]] = {}
    for name, model in models.items():
        reports[name] = {}
        variants = [("raw", model)] + ([("protected", protect(model, rules=rules))] if with_protection else [])
        for mode, m in variants:
            cb = (lambda ev, n=name, md=mode: on_progress(n, md, ev)) if on_progress else None
            reports[name][mode] = test(m, rows, n_variants=n_variants, seed=seed, bootstrap=bootstrap,
                                       rules=rules, on_event=cb)
    any_report = next(iter(next(iter(reports.values())).values()))
    cfg = {"n_messages": len(rows), "n_variants": n_variants, "seed": seed,
           "rules_fingerprint": any_report.metadata.get("rules_fingerprint"),
           "dataset_sha256_16": any_report.metadata.get("dataset_sha256_16"),
           "models": list(models)}
    return ExperimentResult(reports, cfg)
