# The experiment: is FlipRate useful, or just a nice demo?

Goal: measure how much real models change their answer on spelling variants of the same romanized message,
and whether putting normalization in front of them (`fr.protect`) helps without hurting clean accuracy.

**Test first, protect second.** Step 6 tells you whether protection is worth keeping. Do not claim a benefit before it.

## Data

Use a real public benchmark, not the bundled toy sample. **SentiMix (SemEval-2020 Task 9, Hinglish)** is a good start:
~20,000 Hindi-English tweets labeled positive / negative / neutral (14,000 train, 3,000 validation, 3,000 test as
described in the task paper). Get it from the official task page (CodaLab competition 20654) and read its license and
terms before use. I could not verify where the files are hosted today or whether test labels are public. If they are
not, use the validation set as your held-out test set and carve a dev set out of train.

`fliprate.dataset.read_sentimix` reads the token-per-line format, but it was tested only on a synthetic file in that
format. Print a few rows and compare them to your copy before trusting it.

## Steps

1. **Check the data.** `fliprate check-data test.csv` (duplicates, label conflicts, tiny classes).
2. **Keep splits clean.** Rules and thresholds are developed on *train/dev only*. The test set is run **once**, at the end.
   `fliprate split data.csv` keeps spelling variants of one message in the same split.
3. **Mine rules from train/dev, not test.** `fliprate mine train.csv --out candidates.csv`. A fluent speaker marks each row
   `approve` or `reject` in the `status` column (short words such as "mala"/"mila" collide; negations need extra care).
   Use the result with `--rules`.
4. **Audit the variants.** `fliprate audit test.csv -n 100`, have a fluent speaker fill `meaning_preserved` (yes/no), then
   `fliprate audit-summary audit.csv`. Report this precision with its interval. If it is not close to 100%, your
   flip rates are partly measuring meaning changes, so fix the rules first.
5. **Run the models.** Example:
   ```bash
   python experiments/run_experiment.py --sentimix --train train.txt --test test.txt \
       --sklearn --hf cardiffnlp/twitter-xlm-roberta-base-sentiment --hf-lowercase --hf-twitter \
       --rules approved_rules.csv --no-lexicon -n 5 --out experiments/results/run1
   ```
   `--no-lexicon` removes the built-in customer-support word list so that the protection result is not flattered by
   vocabulary written for a different task. Add an LLM with `fr.llm_classifier(...)` in a short script of your own.
6. **Read the protection table.** Protection is worth keeping only if flip rate drops *and* accuracy on originals does
   not fall. If neither holds, say so. That is a finding too.
7. **Look at the failures.** Open the JSON `flips` and `transitions`: which change types and which label pairs flip?
   Are the flips toward the right answer or the wrong one? Flip rate measures instability, not correctness.

## What to report

- Dataset, split, and sizes; the rules file used and its fingerprint (stored in every report).
- Audit precision of the safe-variant claim, per change type.
- Table from `summary.md`: accuracy on originals and variants, macro-F1, flip rate with 95% interval, raw vs protected.
- Which change type is most fragile, and the top label transitions.
- Limits: variants are rule-generated, not from real users; rules cover only reviewed words; intervals are wide on small sets.

## Do not

- Quote numbers from the toy sample or toy model.
- Tune rules on the test set.
- Claim novelty before checking existing robustness work (TextAttack, CheckList, nlpaug, and romanized-Indic benchmarks).

## If you use the participant-processed SentiMix copy

Some GitHub copies (FinalTrainingOnly.tsv, ValidationOnly.tsv, FinalTest.tsv, Ty.txt) are not the official release. In one such copy
we verified: a stray trailing "nan" token in 43.7% of training rows, 2,555 of the 3,000 validation tweets inside the training file,
emoji only in train/valid, and lowercased, stopword-stripped, Porter-stemmed text. `experiments/prepare_sentimix_processed.py`
fixes what can be fixed and writes a report of what it found. Use only the test split as held-out data, and treat results as
evidence about processed text, not about natural user typing.
