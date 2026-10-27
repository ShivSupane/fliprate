"""Command line: `fliprate perturb|normalize|gate|demo|test|check-data|split|mine|audit|audit-summary`."""
from __future__ import annotations

import argparse
import sys

from . import gate, normalize, perturb, protect, test
from .dataset import (check_dataset, make_audit_sample, mine_variants, split_dataset, summarize_audit,
                      write_candidates, write_csv)
from .demo import load_model, sample_dataset
from .evaluation import load_dataset
from .rules import DEFAULT_RULES, RuleSet


import os


def _rules(args):
    return RuleSet.from_csv(args.rules) if getattr(args, "rules", None) else DEFAULT_RULES


def _load_model(spec: str | None = None):
    """Load model from 'module:function' string or sklearn pickle/joblib file."""
    if not spec:
        return load_model(None)
    if os.path.exists(spec) or spec.endswith((".pkl", ".pickle", ".joblib")):
        try:
            import joblib
            clf = joblib.load(spec)
        except Exception:
            import pickle
            with open(spec, "rb") as f:
                clf = pickle.load(f)
        from .adapters import from_sklearn
        return from_sklearn(clf)
    return load_model(spec)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fliprate", description="Robustness testing for romanized Indian text")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, help_, text=False, model=False, rules=True):
        sp = sub.add_parser(name, help=help_)
        if text:
            sp.add_argument("text")
        if model:
            sp.add_argument("--model", default=None, help="module:function or sklearn pickle path (default: toy keyword model)")
        if rules:
            sp.add_argument("--rules", default=None, help="CSV of reviewed spelling groups (see `fliprate mine`)")
        return sp

    add("perturb", "show safe variants of a message", text=True).add_argument("-n", type=int, default=8)
    add("normalize", "show the normalized form of a message", text=True)
    add("gate", "run the PASS/TRANSFORM/REVIEW gate", text=True, model=True)

    sd = add("demo", "robustness test on the bundled toy sample", model=True)
    sd.add_argument("-n", type=int, default=5)

    st = add("test", "robustness test on your labeled CSV", model=True)
    st.add_argument("--data", required=True, help="CSV with columns text,label")
    st.add_argument("-n", type=int, default=5, help="variants per message")
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--protect", action="store_true", help="put normalization in front of the model")
    st.add_argument("--save", default=None, help="write the full report (JSON) here")
    st.add_argument("--max-flip-rate", type=float, default=None, help="CI threshold: maximum allowable flip rate (e.g. 0.05); exit 1 if exceeded")
    st.add_argument("--min-accuracy", type=float, default=None, help="CI threshold: minimum allowable variant accuracy (e.g. 0.80); exit 1 if below")
    st.add_argument("--max-accuracy-drop", type=float, default=None, help="CI threshold: maximum allowable accuracy drop (e.g. 0.05); exit 1 if exceeded")

    srep = add("report", "evaluate model on data and write a Markdown report", model=True)
    srep.add_argument("--data", required=True, help="CSV with columns text,label")
    srep.add_argument("--out", required=True, help="path to write report.md")
    srep.add_argument("-n", type=int, default=5, help="variants per message")
    srep.add_argument("--seed", type=int, default=0)
    srep.add_argument("--protect", action="store_true", help="put normalization in front of the model")

    cd = add("check-data", "find duplicates, label conflicts and tiny classes in a CSV")
    cd.add_argument("data")
    cd.add_argument("--against", default=None, help="training CSV: also report train/eval overlap (leakage)")

    sp_ = add("split", "split a CSV into dev/test without leakage between spelling variants")
    sp_.add_argument("data")
    sp_.add_argument("--out-dir", default=".")
    sp_.add_argument("--test-frac", type=float, default=0.6)
    sp_.add_argument("--seed", type=int, default=0)

    mn = add("mine", "propose spelling groups from real messages, for a human to approve")
    mn.add_argument("data")
    mn.add_argument("--out", default="candidates.csv")
    mn.add_argument("--min-count", type=int, default=2)

    au = add("audit", "sample variants for a fluent speaker to judge (meaning preserved: yes/no)")
    au.add_argument("data")
    au.add_argument("--out", default="audit.csv")
    au.add_argument("-n", type=int, default=100)
    au.add_argument("--seed", type=int, default=0)

    sa = add("audit-summary", "precision of the safe-variant claim from a judged audit CSV", rules=False)
    sa.add_argument("audit_csv")

    add("explain", "explain why a gate decision occurred in plain English", text=True, model=True)

    sbm = add("benchmark", "benchmark candidate models on a dataset and optionally export HTML dashboard")
    sbm.add_argument("--data", required=True, help="CSV with columns text,label")
    sbm.add_argument("--models", required=True, help="comma-separated name=spec pairs, e.g. base=mod:fn1,cand=mod:fn2")
    sbm.add_argument("--baseline", default=None, help="name of the baseline model")
    sbm.add_argument("--out", default=None, help="path to write dashboard (.html or .md)")
    sbm.add_argument("-n", type=int, default=5, help="variants per message")

    args = p.parse_args(argv)
    rules = _rules(args)

    if args.cmd == "perturb":
        for v in perturb(args.text, n=args.n, rules=rules):
            print(f"[{v.kind:11}] {v.text}")
    elif args.cmd == "normalize":
        r = normalize(args.text, rules=rules)
        print(r.normalized)
        for old, new in r.changes:
            print(f"  {old} -> {new}")
    elif args.cmd == "gate":
        r = gate(_load_model(args.model), args.text, rules=rules)
        print(f"{r.decision}: {r.reason}")
        print(f"prediction={r.prediction!r} raw_stability={r.raw_stability} normalized_stability={r.normalized_stability}")
        for v in r.flipped_variants[:5]:
            print("  flipped on:", v)
        for w in r.warnings:
            print("warning:", w)
    elif args.cmd in ("demo", "test"):
        try:
            model = _load_model(args.model)
            if args.cmd == "test" and args.protect:
                model = protect(model, rules=rules)
            rows = sample_dataset() if args.cmd == "demo" else load_dataset(args.data)
        except Exception as exc:
            print(f"Error loading model or dataset: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

        def show(ev):
            if ev["stage"] == "progress" and ev["done"] % 25 == 0:
                print(f"  {ev['done']}/{ev['total']}  flip rate so far {ev['flip_rate']:.1%}", file=sys.stderr)

        report = test(model, rows, n_variants=args.n, seed=getattr(args, "seed", 0), on_event=show, rules=rules)
        print(report.summary())

        # CI Threshold evaluation
        violations = []
        applied_thresholds = {}
        max_fr = getattr(args, "max_flip_rate", None)
        min_acc = getattr(args, "min_accuracy", None)
        max_drop = getattr(args, "max_accuracy_drop", None)

        if max_fr is not None:
            applied_thresholds["max_flip_rate"] = max_fr
            if report.flip_rate > max_fr:
                violations.append(f"flip rate {report.flip_rate:.1%} exceeds threshold {max_fr:.1%}")

        if min_acc is not None:
            applied_thresholds["min_accuracy"] = min_acc
            if report.accuracy_variants < min_acc:
                violations.append(f"accuracy on variants {report.accuracy_variants:.1%} below threshold {min_acc:.1%}")

        if max_drop is not None:
            applied_thresholds["max_accuracy_drop"] = max_drop
            if report.accuracy_drop > max_drop:
                violations.append(f"accuracy drop {report.accuracy_drop:+.1%} exceeds threshold {max_drop:.1%}")

        if applied_thresholds:
            report.metadata["ci_thresholds"] = applied_thresholds
            report.metadata["ci_gate_passed"] = (len(violations) == 0)
            report.metadata["ci_violations"] = violations

        if getattr(args, "save", None):
            report.save(args.save)
            print(f"saved: {args.save}")

        if violations:
            print("\n[CI GATE FAILED] Threshold violations:", file=sys.stderr)
            for v in violations:
                print(f"  ✗ {v}", file=sys.stderr)
            return 1

        if applied_thresholds:
            print(f"\n[CI GATE PASSED] All {len(applied_thresholds)} threshold checks passed.", file=sys.stderr)
        return 0
    elif args.cmd == "report":
        model = _load_model(args.model)
        if args.protect:
            model = protect(model, rules=rules)
        rows = load_dataset(args.data)
        rep = test(model, rows, n_variants=args.n, seed=args.seed, rules=rules)
        md = rep.to_markdown(f"FlipRate Report: {args.data}")
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Report written to {args.out}")
    elif args.cmd == "check-data":
        chk = check_dataset(args.data, rules=rules)
        print(chk.summary())
        if args.against:
            from .dataset import cross_split_overlap
            ov = cross_split_overlap(args.against, args.data, rules=rules)
            print(f"Overlap with {args.against}: {ov['overlapping']} of {ov['eval_rows']} rows "
                  f"({ov['overlap_fraction']:.1%}); {ov['overlapping_with_different_label']} with a different label")
        return 0 if chk.ok else 1
    elif args.cmd == "split":
        dev, tst = split_dataset(args.data, args.test_frac, args.seed, rules=rules)
        import os
        os.makedirs(args.out_dir, exist_ok=True)
        write_csv(dev, os.path.join(args.out_dir, "dev.csv"))
        write_csv(tst, os.path.join(args.out_dir, "test.csv"))
        print(f"dev={len(dev)} test={len(tst)} written to {args.out_dir}/ (variants of one message stay in one split)")
    elif args.cmd == "audit":
        sample = make_audit_sample(args.data, args.n, args.seed, rules=rules, path=args.out)
        print(f"{len(sample)} variants written to {args.out}. Ask a fluent speaker to fill `meaning_preserved` (yes/no).")
    elif args.cmd == "audit-summary":
        for kind, st in sorted(summarize_audit(args.audit_csv).items()):
            print(f"{kind:12} judged={int(st['judged']):4d} meaning preserved {st['precision']:.1%} "
                  f"(95% CI {st['ci_low']:.1%} to {st['ci_high']:.1%})")
    elif args.cmd == "mine":
        cands = mine_variants([t for t, _ in load_dataset(args.data)], min_count=args.min_count, rules=rules)
        write_candidates(cands, args.out)
        print(f"{len(cands)} candidate groups written to {args.out}. Fill the `status` column with approve/reject, "
              f"then use --rules {args.out}")
    elif args.cmd == "explain":
        from .explain import explain_gate_decision
        expl = explain_gate_decision(args.text, _load_model(args.model), rules=rules)
        print(f"Decision: {expl['decision']}")
        print(f"Summary:  {expl['summary']}")
        print(f"Advice:   {expl['recommendation']}")
        print(f"Script:   {expl['script']} | Raw Stability: {expl['raw_stability']:.1%}")
    elif args.cmd == "benchmark":
        from .benchmark import benchmark
        model_dict = {}
        for pair in args.models.split(","):
            if "=" in pair:
                m_name, m_spec = pair.split("=", 1)
                model_dict[m_name.strip()] = _load_model(m_spec.strip())
            else:
                model_dict[pair.strip()] = _load_model(pair.strip())
        suite = benchmark(model_dict, args.data, baseline=args.baseline, n_variants=args.n, rules=rules)
        print(suite.summary())
        if args.out:
            suite.save(args.out)
            print(f"Report saved to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
