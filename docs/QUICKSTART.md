# FlipRate — Quick Start & Run Commands

**Author: Shiv Supane**

---

## 1. Install

```bash
# Core library (zero dependencies)
pip install fliprate

# With the web console + API
pip install "fliprate[api]"

# With the Hugging Face model support
pip install "fliprate[api,hf]"

# With scikit-learn baselines
pip install "fliprate[api,hf,sklearn]"

# Development (from this folder)
pip install -e ".[dev,api,hf,sklearn]"
```

---

## 2. Start the web console

```bash
# Default (toy keyword model — instant, no download)
uvicorn fliprate.api:app --reload

# With the real Hugging Face model
FLIPRATE_MODEL=fliprate.hf_models:xlmt_sentiment uvicorn fliprate.api:app --reload

# With your own model
FLIPRATE_MODEL=mypackage.mymodule:my_classifier uvicorn fliprate.api:app --reload
```

Open **http://localhost:8000** — the console loads automatically.

---

## 3. CLI commands

All commands use `fliprate <subcommand>`.

### Gate — single message decision

```bash
fliprate gate "mala product avadlaaa nahi"
# TRANSFORM: Stable after approved normalization
# prediction='product_feedback' raw_stability=0.625

fliprate gate "delivery nahi aaya" --model mypackage.mymodule:clf
```

### Perturb — see safe spelling variants

```bash
fliprate perturb "mala delivery nahi milali"
# [spelling    ] mala delivery nahee milali
# [repetition  ] mala deliveeery nahi milali
# [punctuation ] mala delivery nahi milali!
# [case        ] MALA DELIVERY NAHI MILALI

fliprate perturb "vanakkam nalla irukku" -n 10
```

### Normalize — clean a message

```bash
fliprate normalize "mala product avadlaaa nhi"
# mala product avadla nahi
#   avadlaaa -> avadla
#   nhi -> nahi

fliprate normalize "romba nallaa iruku"
```

### Robustness test — your labeled CSV

```bash
# Quick test on the bundled 40-message sample
fliprate demo

# Test on your own data
fliprate test --data mydata.csv
fliprate test --data mydata.csv -n 10 --protect --save report.json
fliprate test --data mydata.csv --model mypackage.mymodule:clf --rules approved.csv

# Save the full report
fliprate test --data mydata.csv --save results/report.json
```

### Report — export Markdown report

```bash
fliprate report --data mydata.csv --model mypackage.mymodule:clf --out report.md
fliprate report --data mydata.csv --model model.pkl --rules approved.csv --out report.md
```

### Data tools

```bash
# Check your dataset for duplicates and label conflicts
fliprate check-data mydata.csv
fliprate check-data testdata.csv --against traindata.csv  # leakage check

# Leak-safe train/test split
fliprate split mydata.csv --out-dir data/ --test-frac 0.3

# Mine spelling variant candidates from your data
fliprate mine mydata.csv --out candidates.csv
# → open candidates.csv in Excel, fill 'status' column with approve/reject
# → use the reviewed file:
fliprate test --data mydata.csv --rules candidates.csv

# Human audit of variants
fliprate audit mydata.csv -n 100 --out audit_sample.csv
# → send to a fluent speaker, ask them to fill 'meaning_preserved' (yes/no)
# → analyse results:
fliprate audit-summary audit_sample.csv
```

---

## 4. Python API — quick examples

```python
import fliprate as fr

# ── Gate (real-time quality check) ────────────────────────────────────
result = fr.gate(my_model, "mala product avadlaaa nahi")
print(result.decision)     # PASS | TRANSFORM | REVIEW
print(result.prediction)   # model's answer on the cleaned text
print(result.reason)       # why this decision was made

# ── Robustness test ────────────────────────────────────────────────────
report = fr.test(my_model, "data/test.csv")
print(report.summary())
# Messages: 500 | safe variants tested: 2500
# Accuracy on originals: 68.4%
# Flip rate: 6.2% (95% CI [5.1%, 7.4%])

# ── Protect a model with the normalizer ───────────────────────────────
safe_model = fr.protect(my_model)
protected_report = fr.test(safe_model, "data/test.csv")
# Compare:
print(fr.paired_difference(report, protected_report))

# ── Normalize text ────────────────────────────────────────────────────
r = fr.normalize("mala product avadlaaa nhi")
print(r.normalized)    # "mala product avadla nahi"
print(r.changes)       # [('avadlaaa', 'avadla'), ('nhi', 'nahi')]

# ── Perturb text ──────────────────────────────────────────────────────
variants = fr.perturb("delivery nahi aaya", n=5)
for v in variants:
    print(v.kind, v.text)
```

---

## 5. Multi-language rules (Dakshina + Aksharantar datasets)

These require the datasets to be present locally (not included in the package).

```bash
# Set paths (optional — auto-detected if in the project folder)
export FLIPRATE_DAKSHINA_PATH=/path/to/dakshina_dataset_v1.0
export FLIPRATE_AKSHARANTAR_PATH=/path/to/Aksharantar_All_Languages
```

```python
import fliprate as fr

# Dakshina: 225K+ groups, 12 languages (use 2-letter codes)
rules_ta = fr.dakshina_rules("ta")          # Tamil
rules_hi = fr.dakshina_rules("hi", "mr")   # Hindi + Marathi

# Aksharantar: 428K+ groups, 20 languages (use 3-letter codes)
rules_hin = fr.aksharantar_rules("hin")         # Hindi
rules_all = fr.aksharantar_rules("tam", "tel", "kan", "mal")  # South India

# Combined — best coverage
groups = fr.load_dakshina_groups("ta") + fr.load_aksharantar_groups("tam")
from fliprate.rules import RuleSet, DEFAULT_RULES
rules = RuleSet(
    groups=list(DEFAULT_RULES.groups) + groups,
    lexicon=list(DEFAULT_RULES.lexicon),
    negations=DEFAULT_RULES.negations,
    ambiguous=dict(DEFAULT_RULES.ambiguous),
)

# Use any rules object in any API call
report = fr.test(my_model, "data/test.csv", rules=rules)
result = fr.gate(my_model, "vanakkam nalla irukku", rules=rules)
```

---

## 6. Environment variables

| Variable | Default | Effect |
|---|---|---|
| `FLIPRATE_MODEL` | toy keyword model | `module:function` to load your classifier |
| `FLIPRATE_API_KEY` | none (open) | Require this key on POST endpoints |
| `FLIPRATE_CORS` | `*` | Comma-separated allowed origins for production |
| `FLIPRATE_RATE_LIMIT` | `60` | Max requests per minute per IP/key |
| `FLIPRATE_MAX_CONCURRENT` | `2` | Max simultaneous robustness test jobs |
| `FLIPRATE_DAKSHINA_PATH` | auto-detect | Path to dakshina_dataset_v1.0 |
| `FLIPRATE_AKSHARANTAR_PATH` | auto-detect | Path to Aksharantar_All_Languages |

---

## 7. Run tests

```bash
python -m pytest tests/ -q
# 86 passed
```

---

## 8. Docker

```bash
docker build -t fliprate .
docker run -p 7860:7860 fliprate
# Open http://localhost:7860

# With real HF model
docker run -p 7860:7860 -e FLIPRATE_MODEL=fliprate.hf_models:xlmt_sentiment fliprate
```

---

## 9. Regenerate rules from datasets

```bash
# From Dakshina (12 languages)
python scripts/build_rules_from_dakshina.py

# From Aksharantar (20 languages)
python scripts/build_rules_from_aksharantar.py

# Specific languages only
python scripts/build_rules_from_aksharantar.py --langs hin tam tel kan

# Output goes to data/rules_from_dakshina/ and data/rules_from_aksharantar/
```
