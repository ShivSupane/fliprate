"""Turn the participant-processed SentiMix copy (FinalTrainingOnly/ValidationOnly/FinalTest/Ty) into clean CSVs.

    python experiments/prepare_sentimix_processed.py --train FinalTrainingOnly.tsv --valid ValidationOnly.tsv \
        --test FinalTest.tsv --test-labels Ty.txt --out-dir data/sentimix_processed

Problems found in this copy (all verified on the files), and what this script does about them:
  * 43.7% of TRAIN rows end with a stray "nan" token (a stringified missing value); valid/test have none.
        -> the trailing "nan" is removed from every split.
  * emoji/non-ASCII characters appear in 22% of train/valid rows but only 2% of test rows.
        -> non-ASCII characters are removed from every split so the splits look alike.
  * the training file contains 2,555 of the 3,000 validation tweets (same ids, same labels).
        -> validation is NOT used as held-out data; only the test split is.
  * text is lowercased and Porter-stemmed with stopwords removed (e.g. "polic", "occupi", "movi").
        -> cannot be repaired; this is why results on this copy are not evidence about natural user typing.
Label mapping 0=negative, 1=neutral, 2=positive was checked against the published validation counts.
"""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

LABELS = {"0": "negative", "1": "neutral", "2": "positive"}
NAN_TAIL = re.compile(r"(?:^|\s)nan$")


def clean(text: str) -> str:
    text = NAN_TAIL.sub("", text.strip())
    text = text.encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", text).strip()


def read_tsv(path, names):
    return pd.read_csv(path, sep="\t", header=None, names=names, quoting=3, dtype=str, keep_default_na=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", required=True)
    ap.add_argument("--valid", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--test-labels", required=True, help="Ty.txt: CSV with columns Uid,Sentiment")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    tr = read_tsv(a.train, ["id", "text", "label"])
    va = read_tsv(a.valid, ["id", "text", "label"])
    te = read_tsv(a.test, ["id", "text"])
    ty = pd.read_csv(a.test_labels, dtype=str, keep_default_na=False).rename(columns={"Uid": "id", "Sentiment": "label"})
    te = te.merge(ty, on="id", how="left")
    assert te.label.notna().all() and len(te) == len(ty), "test ids and Ty.txt ids do not match"
    for d in (tr, va):
        d["label"] = d["label"].map(LABELS)
        assert d.label.notna().all(), "unexpected label value"

    report = {"rows_raw": {"train": len(tr), "valid": len(va), "test": len(te)},
              "train_rows_ending_nan": float(tr.text.str.contains(NAN_TAIL).mean()),
              "valid_ids_inside_train": len(set(tr.id) & set(va.id)),
              "published_valid_counts": {"positive": 982, "negative": 890, "neutral": 1128},
              "valid_counts": va.label.value_counts().to_dict(),
              "test_counts": te.label.value_counts().to_dict(),
              "published_test_counts": {"positive": 1000, "negative": 900, "neutral": 1100}}
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, d in (("train", tr), ("valid", va), ("test", te)):
        d = d.assign(text=d.text.map(clean))
        d = d[d.text != ""][["text", "label"]]
        report.setdefault("rows_clean", {})[name] = len(d)
        d.to_csv(out / f"{name}.csv", index=False)
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
