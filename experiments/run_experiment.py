"""The experiment that decides whether FlipRate is useful: real models, raw vs protected, same data and seeds.

Example (sentiment on Hinglish, 3 classes):

    python experiments/run_experiment.py --train data/train.csv --test data/test.csv \
        --sklearn --hf cardiffnlp/twitter-xlm-roberta-base-sentiment --hf-lowercase --hf-twitter \
        --rules data/rules_approved.csv -n 5 --out experiments/results/run1

Needs: scikit-learn for --sklearn; transformers + torch for --hf. Results (JSON per model and mode, summary.md)
are written to --out. Every number comes from this run; nothing is pre-filled.
"""
import argparse
import sys

import fliprate as fr
from fliprate.dataset import check_dataset, cross_split_overlap, read_sentimix
from fliprate.demo import sample_dataset_path
from fliprate.demo import load_model
from fliprate.rules import DEFAULT_RULES, RuleSet


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True, help="held-out data (CSV text,label; or SentiMix format with --sentimix)")
    ap.add_argument("--train", help="training data for the scikit-learn baselines")
    ap.add_argument("--sentimix", action="store_true", help="--train/--test files are in SemEval SentiMix format")
    ap.add_argument("--sklearn", action="store_true", help="train word and char n-gram TF-IDF baselines on --train")
    ap.add_argument("--hf", nargs="*", default=[], help="Hugging Face text-classification model names")
    ap.add_argument("--hf-lowercase", action="store_true", help="lowercase model labels (Positive -> positive)")
    ap.add_argument("--hf-twitter", action="store_true", help="apply the @user/http placeholders Twitter models expect")
    ap.add_argument("--hf-device", default=None, help="e.g. cuda, mps, or 0")
    ap.add_argument("--models", nargs="*", default=[], help="your own models as module:function")
    ap.add_argument("--rules", help="CSV of human-approved spelling groups")
    ap.add_argument("--no-lexicon", action="store_true",
                    help="drop the built-in customer-support word list (use this for any non-support dataset)")
    ap.add_argument("-n", type=int, default=5, help="safe variants per message")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-protection", action="store_true")
    ap.add_argument("--out", default="experiments/results/run")
    a = ap.parse_args()

    load = read_sentimix if a.sentimix else fr.load_dataset
    test_rows = load(a.test)
    base = RuleSet(lexicon=[]) if a.no_lexicon else DEFAULT_RULES
    rules = RuleSet.from_csv(a.rules, base=base) if a.rules else base
    header = []
    toy_sha = fr.evaluation._dataset_sha(fr.load_dataset(sample_dataset_path()))
    is_toy = fr.evaluation._dataset_sha(test_rows) == toy_sha
    if is_toy:
        header.append("**TOY RESULTS: bundled demo sample. This is a pipeline check, NOT evidence.**")
    chk = check_dataset(test_rows, rules=rules).summary()
    header.append("```\n" + chk + "\n```")
    print(chk, file=sys.stderr)
    if a.train:
        ov = cross_split_overlap(load(a.train), test_rows, rules=rules)
        line = (f"Train/test overlap: {ov['overlapping']} of {ov['eval_rows']} test rows ({ov['overlap_fraction']:.1%}) "
                f"also appear in train; {ov['overlapping_with_different_label']} with a different label.")
        header.append(line)
        print(line, file=sys.stderr)

    models = {}
    if a.sklearn:
        if not a.train:
            ap.error("--sklearn needs --train")
        from fliprate.baselines import train_tfidf
        train_rows = load(a.train)
        models["tfidf-word"] = train_tfidf(train_rows, "word")
        models["tfidf-char"] = train_tfidf(train_rows, "char_wb")
    for name in a.hf:
        device = int(a.hf_device) if (a.hf_device or "").isdigit() else a.hf_device
        models[name] = fr.HFTextClassifier(name, device=device, lowercase_labels=a.hf_lowercase,
                                           preprocess=fr.twitter_preprocess if a.hf_twitter else None)
    for spec in a.models:
        models[spec] = load_model(spec)
    if not models:
        ap.error("choose at least one model: --sklearn, --hf NAME, or --models module:function")

    def progress(name, mode, ev):
        if ev["stage"] == "progress" and ev["done"] % 50 == 0:
            print(f"  [{name} / {mode}] {ev['done']}/{ev['total']}  flip rate so far {ev['flip_rate']:.1%}", file=sys.stderr)

    result = fr.run_experiment(models, test_rows, n_variants=a.n, seed=a.seed, rules=rules,
                               with_protection=not a.no_protection, on_progress=progress)
    out = result.save(a.out)
    if a.sentimix and not is_toy:
        for name, modes in result.reports.items():
            if modes["raw"].macro_f1_original > 0.9:
                w = (f"WARNING: {name} macro-F1 {modes['raw'].macro_f1_original:.2f} on SentiMix is far above the best "
                     f"published system (~0.75 F1). Check for train/test leakage before trusting anything.")
                header.append(w)
                print(w, file=sys.stderr)
    (out / "RESULTS.md").write_text("\n\n".join(header) + "\n\n" + result.markdown() + "\n", encoding="utf-8")
    print(result.markdown())
    print(f"\nSaved to {out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
