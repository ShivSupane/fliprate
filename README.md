# FlipRate

> **Does your NLP model flip its prediction when a user types the same message differently?**  
> FlipRate measures that, quantifies it with confidence intervals, and gives you tools to fix it — in **one pip install**, zero required dependencies.

[![PyPI](https://img.shields.io/pypi/v/fliprate?color=4f46e5&label=PyPI)](https://pypi.org/project/fliprate/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-3b82f6)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)
[![Tests](https://github.com/ShivSupane/fliprate/actions/workflows/ci.yml/badge.svg)](https://github.com/ShivSupane/fliprate/actions)
[![Downloads](https://img.shields.io/pypi/dm/fliprate)](https://pypi.org/project/fliprate/)

Built by **[Shiv Supane](https://github.com/ShivSupane)**

---

## The problem

People who text in Hindi, Marathi, Bengali, Tamil, and other Indian languages using the
Roman alphabet have no standard spelling. The same word gets typed dozens of ways:

| Hindi word | Romanized variants |
|---|---|
| नही (no) | `nahi` · `nhi` · `nahin` · `nahee` |
| बहुत (very) | `bahut` · `bohot` · `bhut` · `boht` |
| आवडला (liked) | `avadla` · `avdla` · `avadlaaa` |

Most NLP models treat these as completely different inputs and change their prediction —
even though the meaning is identical. **FlipRate quantifies that brittleness and fixes it.**

```
User types:   "mala product avadlaaa nahi"  →  model says: negative ✓
Same meaning: "mala product avdla nhi"      →  model says: positive ✗  ← FLIP
```

---

## Quick start

```bash
pip install fliprate
```

```python
import fliprate as fr

# 1 — Does my model flip on this sentence?
result = fr.gate(my_model, "delivery nahi aayi")
print(result.decision)        # PASS | TRANSFORM | REVIEW
print(result.flip_rate)       # e.g. 0.33

# 2 — Normalize noisy input before prediction
clean = fr.normalize_text("mala product avadlaaa nhi")
# → "mala product avadla nahi"

# 3 — Measure flip rate across your whole test set
report = fr.check_model(my_model, "data/test.csv")
print(report.summary())
# flip_rate: 0.27  [95% CI 0.21–0.33]  accuracy: 0.84  flips: 54/200

# 4 — Protected prediction: normalize → predict → check → route
pred = fr.safe_predict(my_model, "bohot accha product hai")
```

---

## Showcase & Colab notebook

FlipRate ships two living, executable references for the whole public API:

**1. Local showcase** — `scratch/fliprate_showcase.py`

A single Python file that walks every public function (23 sections) with deep
WHAT / WHY / HOW / WHEN explanations. Runs with only the dev extras:

```bash
pip install "fliprate[dev,sklearn]"
python scratch/fliprate_showcase.py
```

**2. Google Colab notebook** — [`examples/fliprate_colab_showcase.ipynb`](examples/fliprate_colab_showcase.ipynb)

A one-click, browser-runnable version of the showcase that downloads the **real
datasets** (Dakshina + Aksharantar) and runs **live models** — a Hugging Face
multilingual sentiment model plus LLM classification through your own Gemini /
OpenAI API key — so you can see FlipRate's robustness checks against actual
models and real transliteration data.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShivSupane/fliprate/blob/main/examples/fliprate_colab_showcase.ipynb)

Inside the notebook you set two Colab secrets (`GEMINI_API_KEY`, `OPENAI_API_KEY`);
sections degrade gracefully to a local fallback when a key is absent.

---

## Why FlipRate?

| What you need | FlipRate answer |
|---|---|
| Measure model brittleness on Indian-language text | `fr.gate()`, `fr.check_model()`, `fr.test()` |
| Fix noisy input before it reaches your model | `fr.normalize_text()`, `fr.protect()` |
| Compare multiple models side-by-side | `fr.benchmark()`, `fr.run_benchmark_protocol()` |
| Audit your evaluation dataset for quality issues | `fr.audit_dataset()` |
| Explain *why* a model flipped | `fr.show_why_it_flipped()`, `fr.explain_gate_decision()` |
| Integrate into CI/CD | `fliprate test --data test.csv --max-flip-rate 0.05` |
| Use Hugging Face / sklearn / any callable | `fr.HFTextClassifier`, `fr.from_sklearn`, `fr.CallableAdapter` |

---

## Installation

```bash
# Core (zero required runtime dependencies)
pip install fliprate

# With REST API / web console
pip install "fliprate[api]"

# With Hugging Face support
pip install "fliprate[hf]"

# With scikit-learn adapter
pip install "fliprate[sklearn]"

# Everything
pip install "fliprate[all]"
```

---

## Core concepts

### 1 · Gate — real-time quality check

```python
result = fr.gate(my_model, "delivery nahi aayi")

result.decision      # "PASS" | "TRANSFORM" | "REVIEW"
result.flip_rate     # fraction of variants that flipped
result.prediction    # the model's prediction on the (normalized) input
result.variants      # list of Variant objects with each perturbation
```

Decisions:
- **PASS** — model is stable; safe to use the prediction.
- **TRANSFORM** — model is unstable but we successfully normalized the text; use the prediction on the normalized form.
- **REVIEW** — model is unstable and normalization did not help; flag for human review.

### 2 · Normalize — clean messy text

```python
# Clean a single string
clean = fr.normalize_text("mala product avadlaaa nhi bohotttt")
# → "mala product avadla nahi bahut"

# Normalize + protect brand names / codes from being changed
clean = fr.normalize_text("mujhe ORDID-123 avdla nhi", protected_terms=["ORDID-123"])
```

### 3 · Test — batch evaluation on a dataset

```python
# CSV with columns: text, label
report = fr.test(my_model, "data/test.csv")

report.flip_rate          # overall flip rate
report.accuracy           # accuracy on original text
report.confidence_interval  # (lower, upper) 95% Wilson CI
report.summary()          # pretty-printed one-liner
```

### 4 · Benchmark — compare models

```python
suite = fr.benchmark(
    models={"tfidf": model_a, "bert": model_b},
    dataset="data/test.csv",
    baseline="tfidf",
)
suite.summary()           # terminal scorecard
suite.save("report.html") # standalone interactive HTML dashboard
suite.save("report.md")   # markdown table
```

### 5 · Dataset Audit — catch quality issues before you report numbers

```python
audit = fr.audit_dataset(
    "data/test.csv",
    train_dataset="data/train.csv",   # optional: check for leakage
)
audit["n_messages"]              # total rows
audit["class_distribution"]      # label → count
audit["script_distribution"]     # latin / devanagari / mixed → count
audit["exact_duplicates"]        # exact duplicate pairs
audit["near_duplicates"]         # spelling-variant pairs
audit["label_conflicts"]         # same text, different labels
audit["leakage_detected"]        # True if train/test overlap found
```

### 6 · Reproducible benchmark protocol

```python
suite = fr.run_benchmark_protocol(
    models={"word_tfidf": model_a, "char_tfidf": model_b},
    dataset="data/test.csv",
    train_dataset="data/train.csv",
    baseline="word_tfidf",
    n_variants=5,
    seed=0,
)
suite.save("results/dashboard.html")
suite.dataset_audit   # full audit dict attached to the suite
```

---

## Plug in any model

FlipRate works with **any callable that takes a string and returns a label**.

```python
# ── Any Python function ──────────────────────────────────────────
def my_model(text: str) -> str:
    return "positive" if "accha" in text else "negative"

result = fr.gate(my_model, "bohot accha product hai")

# ── Hugging Face pipeline ────────────────────────────────────────
clf = fr.HFTextClassifier(
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    lowercase_labels=True,
)
result = fr.gate(clf, "mujhe ye product pasand nahi")

# ── scikit-learn pipeline ────────────────────────────────────────
from sklearn.pipeline import Pipeline
clf = fr.from_sklearn(sklearn_pipeline)

# ── REST / HTTP service ──────────────────────────────────────────
clf = fr.HttpServiceAdapter("https://api.example.com/classify", timeout=5.0)

# ── LLM (OpenAI / Anthropic / Gemini) ───────────────────────────
clf = fr.LLMClassifier(
    client=openai_client,
    labels=["positive", "negative", "neutral"],
)
```

---

## CLI

Use FlipRate in CI/CD pipelines or shell scripts without writing Python.

```bash
# ── Real-time gate ───────────────────────────────────────────────
fliprate gate "delivery nahi aayi"

# ── Normalize text ───────────────────────────────────────────────
fliprate normalize "mala product avadlaaa nhi"

# ── Show all perturbations ───────────────────────────────────────
fliprate perturb "delivery nahi aaya"

# ── Explain a gate decision ──────────────────────────────────────
fliprate explain --model "mypackage:my_model" --text "delivery nahi aayi"

# ── Batch evaluation (CI gate) ───────────────────────────────────
fliprate test --data test.csv --max-flip-rate 0.05      # exit 1 if threshold exceeded
fliprate test --data test.csv -n 100 --protect --save report.json

# ── Multi-model benchmark ────────────────────────────────────────
fliprate benchmark --models models.json --data test.csv --train-data train.csv --out scorecard.html

# ── Full report ──────────────────────────────────────────────────
fliprate report --data test.csv --model "mypackage:my_model" --out report.md

# ── Dataset tools ────────────────────────────────────────────────
fliprate check-data test.csv --against train.csv   # check leakage
fliprate split mydata.csv --out-dir data/          # train/test split
fliprate mine mydata.csv --out candidates.csv      # mine spelling rules
```

**Exit codes:**

| Code | Meaning |
|---|---|
| `0` | All checks passed |
| `1` | Threshold violated (flip rate / accuracy) |
| `2` | Configuration or data error (bad file, model not found) |

---

## Devanagari & multilingual support

FlipRate supports Devanagari (Hindi, Marathi) and mixed-script text natively:

```python
# Script detection
fr.detect_script("मला प्रॉडक्ट आवडला")        # "devanagari"
fr.detect_script("product bohot accha hai")   # "latin"
fr.detect_script("ye product बहुत accha hai") # "mixed"

# Gate on Devanagari text
result = fr.gate(my_model, "मला प्रॉडक्ट आवडला नाही")
```

Perturbations include:
- Unicode normalization (precomposed vs decomposed nuktas)
- Digit swap: ASCII (`0–9`) ↔ Devanagari (`०–९`)
- Danda (`।`) / double-danda (`॥`) ↔ Western punctuation
- Meaning-safe nukta variations (`ज़रा`/`जरा`, `फ़ोन`/`फोन`)
- Hindi/Marathi negation variants (`नहीं`, `नाही`, `नको`)

---

## Web console

```bash
pip install "fliprate[api]"
uvicorn fliprate.api:app --reload
# → open http://localhost:8000
```

Two tabs:
- **Playground** — paste any message, see the gate decision live
- **Robustness Lab** — upload a labeled CSV, get a live flip-rate chart

---

## CI/CD integration

Add FlipRate to your GitHub Actions pipeline to catch regressions automatically:

```yaml
# .github/workflows/robustness.yml
- name: Install FlipRate
  run: pip install fliprate

- name: Run robustness gate
  run: |
    fliprate test \
      --data tests/data/test.csv \
      --model "mypackage:my_model" \
      --max-flip-rate 0.05 \
      --save robustness_report.json
  # exits 1 if flip rate exceeds 5%

- name: Upload report
  uses: actions/upload-artifact@v4
  with:
    name: robustness-report
    path: robustness_report.json
```

---

## Explainability & debugging

```python
# Why did the gate return REVIEW?
explanation = fr.explain_gate_decision(my_model, "delivery nahi aayi")
print(explanation["summary"])
print(explanation["recommendation"])

# Which exact variants flipped, and what changed?
flips = fr.show_why_it_flipped(my_model, "bohot accha product")
for flip in flips:
    print(f"Variant:  {flip['variant_text']}")
    print(f"Flipped:  {flip['original_prediction']} → {flip['variant_prediction']}")
    print(f"Diff:     {flip['diffs']}")

# Preview what normalization does to noisy text
variants = fr.preview_normalized_variants("bohotttt achaaa product")

# Report-level explainability (privacy-safe by default)
summary = fr.explain_report_flips(report)
# opt_in_examples=True to include raw text in output
```

---

## Rule management

```python
# Inspect built-in rules
fr.describe_rule_group("nahi")

# Add a project-specific spelling group
fr.add_custom_rule(["thnx", "thanks", "thankyou", "thx"])

# Protect brand names / codes from being changed
fr.normalize_text("please check ORDER-999 nahi aaya", protected_terms=["ORDER-999"])

# Preview what variants a rule generates
fr.preview_variants("delivery nahi aayi")

# Mine new rule candidates from your own data
candidates = fr.learn_rules_from_texts(my_texts)
```

---

## Security & privacy

- **Privacy by default** — `explain_report_flips(report)` never includes raw user text in output. Opt in with `opt_in_examples=True`.
- **Zero network calls** — core FlipRate runs entirely in-process; no telemetry, no external APIs.
- **Zero required dependencies** — safe to add to any project.
- **Vulnerability disclosure** — email `shiv.supane777@gmail.com` (see [`SECURITY.md`](SECURITY.md)).

---

## API quick-reference

| Function | One-liner description |
|---|---|
| `fr.gate(model, text)` | Real-time robustness check → `GateResult` |
| `fr.quick_check(model, text)` | Fastest 1-line check → `QuickCheckResult` |
| `fr.safe_predict(model, text)` | Normalize + gate + predict |
| `fr.normalize_text(text)` | Clean noisy romanized text |
| `fr.test(model, dataset)` | Batch evaluation → `Report` |
| `fr.check_model(model, dataset)` | Alias for `test` with summary |
| `fr.benchmark(models, dataset)` | Multi-model comparison → `BenchmarkSuite` |
| `fr.audit_dataset(dataset)` | Data quality audit → dict |
| `fr.run_benchmark_protocol(...)` | Rigorous reproducible benchmark |
| `fr.show_why_it_flipped(model, text)` | Explain each flip with word diffs |
| `fr.explain_gate_decision(model, text)` | Human-readable gate explanation |
| `fr.detect_script(text)` | `"latin"` / `"devanagari"` / `"mixed"` |
| `fr.from_sklearn(pipeline)` | Wrap any sklearn model |
| `fr.HFTextClassifier(model_name)` | Wrap a Hugging Face model |
| `fr.LLMClassifier(client, labels)` | Wrap an LLM API |

Full API reference → [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md)

---

## Project layout

```
fliprate/
├── src/fliprate/
│   ├── __init__.py        # all public exports
│   ├── engine.py          # FlipRate class
│   ├── gating.py          # gate / iter_gate / agate
│   ├── pipeline.py        # quick_check / safe_predict / check_model
│   ├── normalization.py   # normalize / normalize_text
│   ├── perturbation.py    # perturb / KINDS / Variant
│   ├── rules.py           # RuleSet / DEFAULT_RULES
│   ├── evaluation.py      # test / Report / iter_test
│   ├── benchmark.py       # benchmark / BenchmarkSuite / audit_dataset
│   ├── explain.py         # show_why_it_flipped / explain_gate_decision
│   ├── adapters.py        # HFTextClassifier / LLMClassifier / from_sklearn
│   ├── dataset.py         # find_near_duplicates / check_leakage / mine_variants
│   ├── augmentation.py    # augment / compare_augmentation
│   ├── cli.py             # fliprate CLI entry point
│   └── api.py             # FastAPI web console
├── tests/                 # 414 tests across all versions
├── examples/
│   ├── 01_basic_usage.py … 07_benchmarking_dashboard.py
│   └── fliprate_colab_showcase.ipynb   # Colab notebook — real datasets + live models
├── scratch/
│   └── fliprate_showcase.py            # local executable API showcase (23 sections)
├── docs/
│   ├── API_REFERENCE.md
│   ├── DEPLOYMENT_GUIDE.md
│   └── FLIPRATE_MASTER_ROADMAP.md
├── CHANGELOG.md
├── SECURITY.md
└── README.md
```

---

## Versioning

FlipRate follows [Semantic Versioning](https://semver.org/).  
Public API is frozen at 1.0.0 — future releases will be backward compatible.

| Version | Highlight |
|---|---|
| **1.0.2** | Google Colab showcase notebook + docs refresh (real datasets, live HF/LLM models) |
| **1.0.1** | Ergonomics properties, `protected_terms` pass-through, full executable showcase |
| **1.0.0** | Stable release — API frozen, 414 tests passing |
| 0.9.0 | Dataset audit, benchmark protocol, security policy, CLI exit codes |
| 0.8.0 | Protected terms, rule governance, non-causal diagnosis |
| 0.7.0 | Training augmentation, near-duplicate detection, leakage checking |
| 0.6.0 | Cost modes, partial evaluation, async gate |
| 0.5.0 | Multi-model benchmarking, HTML dashboards, explainability |
| 0.4.0 | Devanagari support, Dakshina/Aksharantar loaders |

Full history → [`CHANGELOG.md`](CHANGELOG.md)

---

## Contributing

1. Fork the repository and create a feature branch.
2. Add tests for every new behavior (`tests/` directory).
3. Ensure all 414 tests pass: `python -m pytest -q`
4. Open a PR — the CI checks Python 3.9 → 3.13 on every push.

No new required runtime dependencies, please.

---

## License

MIT — see [`LICENSE`](LICENSE)

---

*FlipRate — built to make Indian-language NLP robust by default.*
