# FlipRate

**Does your NLP model change its answer when the same message is typed differently?**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-80%20passed-brightgreen)](.github/workflows/ci.yml)

Built by **Shiv Supane**

---

## The problem

People who text in Hindi, Marathi, Bengali, Tamil, Telugu, and other Indian languages
using Roman (English) script have no standard spelling. The same word gets typed dozens of ways:

| Word | Romanized variants |
|---|---|
| नही (no) | `nahi` · `nhi` · `nahin` · `nahee` |
| बहुत (very) | `bahut` · `bohot` · `bhut` · `boht` |
| आवडलाsss | `avadlaaa` · `avdla` · `avadla` |

Most NLP models treat these as completely different inputs and give different answers —
even though the meaning is identical. **FlipRate measures that problem, quantifies it,
and gives you tools to fix it.**

```
User types:  "mala product avadlaaa nahi"   →  model says: product_feedback ✓
Same meaning: "mala product avdla nhi"      →  model says: human_support    ✗  ← FLIP
```

A **flip rate of 7%** means your model changes its answer on 7% of meaning-preserving
spelling variants. After FlipRate's normalizer: **3.5%** — without changing accuracy.

---

## Install

```bash
pip install fliprate                    # core (zero dependencies)
pip install "fliprate[api]"             # + FastAPI web console
pip install "fliprate[api,hf]"          # + Hugging Face model support
pip install "fliprate[api,hf,sklearn]"  # + scikit-learn baselines
```

From source:
```bash
git clone https://github.com/ShivSupane/fliprate.git
cd fliprate
pip install -e ".[dev,api,hf,sklearn]"
```

---

## Quick start

### Test your model in 3 lines

```python
import fliprate as fr

report = fr.test(my_model, "data/test.csv")   # my_model: any function text → label
print(report.summary())
```

```
Messages: 500 | safe variants tested: 2500
Accuracy on originals:      68.4%  (macro-F1 0.672)
Accuracy on safe variants:  67.1%  (macro-F1 0.658)
Accuracy drop:              -1.3%  95% CI [-2.1%, -0.5%]
Flip rate:                   6.2%  95% CI [5.1%, 7.4%]
Most fragile change type:   repetition (flip rate 9.8%)
```

### Fix it with one line

```python
safe_model = fr.protect(my_model)           # puts the normalizer in front
report2 = fr.test(safe_model, "data/test.csv")

diff = fr.paired_difference(report, report2)
print(f"Flip rate change: {diff['difference']:+.1%}")   # e.g. -3.8%
print(f"95% CI: {diff['ci']}")                          # excludes zero → significant
```

### Real-time gate

```python
result = fr.gate(my_model, "mala product avadlaaa nahi")
print(result.decision)     # PASS / TRANSFORM / REVIEW
print(result.prediction)   # the model's answer
print(result.reason)       # why this decision
```

---

## Web Console

Start the server and open `http://localhost:8000` in your browser:

```bash
# Toy model (instant, no download)
uvicorn fliprate.api:app --reload

# Real Hugging Face model
FLIPRATE_MODEL=fliprate.hf_models:xlmt_sentiment uvicorn fliprate.api:app --reload

# Your own model
FLIPRATE_MODEL=mypackage.mymodule:my_clf uvicorn fliprate.api:app --reload
```

The console has two tabs:
- **Playground** — paste any message, watch the gate decision stream live
- **Robustness Lab** — upload your CSV or use the bundled sample, get a live flip-rate chart

The console is zero-dependency vanilla HTML/JS — served directly by FastAPI, no Streamlit needed.

---

## CLI

```bash
# Test on your data
fliprate test --data mydata.csv
fliprate test --data mydata.csv -n 10 --protect --save report.json

# Single message gate
fliprate gate "mala product avadlaaa nahi"

# See spelling variants
fliprate perturb "delivery nahi aaya"

# Normalize a message
fliprate normalize "mala product avadlaaa nhi"

# Dataset health check
fliprate check-data testdata.csv --against traindata.csv

# Leak-safe train/test split
fliprate split mydata.csv --out-dir data/

# Mine spelling groups from your data
fliprate mine mydata.csv --out candidates.csv
```

---

## Plug in any model

```python
# Hugging Face
clf = fr.HFTextClassifier(
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    lowercase_labels=True,
    preprocess=fr.twitter_preprocess,
)

# scikit-learn
from fliprate.baselines import train_tfidf
clf = train_tfidf(train_rows, analyzer="word")   # or "char_wb"

# Any LLM
llm = fr.llm_classifier(
    my_openai_complete_fn,
    labels=["positive", "neutral", "negative"],
)
llm_safe = fr.with_retries(llm)   # add exponential backoff
```

Models with a `predict_batch(texts) → labels` method are called in batches,
and identical texts are deduplicated and cached.

---

## Multi-language rules

FlipRate ships with 73 core HIGH-confidence spelling groups for immediate use.
For serious work, load from two large human-annotated datasets:

```python
# Dakshina (Google Research, CC BY-SA 4.0) — 225K groups, 12 languages
# Language codes: hi, ur, mr, bn, ta, te, kn, ml, gu, pa
rules = fr.dakshina_rules("ta")          # Tamil: 19K+ groups
rules = fr.dakshina_rules("hi", "mr")   # Hindi + Marathi: 31K+ groups

# Aksharantar (AI4Bharat, CC BY 4.0) — 428K groups, 21 languages
# Language codes: hin, guj, tam, tel, kan, mal, mar, pan, ben, urd, ori, asm, ...
rules = fr.aksharantar_rules("hin")              # Hindi: 58K groups
rules = fr.aksharantar_rules("tam", "tel", "kan") # South India: 83K+ groups

# Use in any call
report = fr.test(my_model, "data/test.csv", rules=rules)
result = fr.gate(my_model, "vanakkam nalla irukku", rules=rules)
```

Combined, the two datasets cover **24 million** human-annotated word pairs across
**21 Indian languages**.

---

## How it works

### 1. Safe variants

A variant of a message is "safe" if and only if it has:
- The same canonical words (up to known-equivalent spellings)
- The same negation count (never adds or removes words like `nahi`, `nako`, `illa`)

This strict definition prevents false positives. Adding `nahi` to a message is **not** a spelling variant — it's a different meaning.

### 2. Flip rate

```
flip rate = (variants where model changed answer) / (total safe variants)
```

Confidence intervals are computed by 1000-resample bootstrap over messages.

### 3. Gate decisions

| Decision | Meaning |
|---|---|
| **PASS** | Model gives the same answer on 80%+ of safe variants. |
| **TRANSFORM** | Model is unstable, but normalization fixes it. Send normalized text to model; keep original for audit. |
| **REVIEW** | Unstable and normalization doesn't help. Route to a human. |

### 4. Normalization strategy

Conservative — only repairs things we are certain about:
- Known spelling variants → canonical form (`nhi` → `nahi`, `bohot` → `bahut`)
- Consonant runs of 3+ → 1 (`avadlaaa` → `avadla`)
- Vowel runs of 3+ → 2 (`sooooo` → `soo`)
- Repeated punctuation collapsed (`!!!` → `!`)
- Ambiguous tokens (`nai`, `he`, `na`) are **never** rewritten — they stay as-is with a warning

---

## Results (v0.2)

On SentiMix Hinglish sentiment (2,997 test messages, 5 variants/message):

| Model | Mode | Accuracy | Flip rate (95% CI) |
|---|---|---|---|
| xlm-roberta-base-sentiment | raw | 59.2% | 7.3% (6.7–7.9%) |
| xlm-roberta-base-sentiment | **protected** | 59.1% | **3.5%** (3.1–3.9%) |
| TF-IDF word n-gram | raw | 65.9% | 4.2% (3.8–4.6%) |
| TF-IDF word n-gram | **protected** | 65.6% | **3.2%** (2.9–3.6%) |

Paired effect of normalization: flip rate **−3.80 pts** [−4.22, −3.39].
Accuracy on originals: −0.03 pts [−0.23, +0.17] — **no significant change**.

> The case flip rate for the transformer was 11.4% raw → 0.3% protected.
> Spelling swap flip rate dropped to 0.0% after protection (normalizer covers its own rules).

Full evidence: [`test and other evalutions/evidence/EVIDENCE.md`](test%20and%20other%20evalutions/evidence/EVIDENCE.md)

---

## Full experiment

```bash
python experiments/run_experiment.py \
    --sentimix \
    --train train.txt \
    --test test.txt \
    --sklearn \
    --hf cardiffnlp/twitter-xlm-roberta-base-sentiment --hf-lowercase --hf-twitter \
    --rules approved_rules.csv \
    -n 5 \
    --out experiments/results/run1
```

See [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) for the full protocol.

---

## Production API

```bash
pip install "fliprate[api]"
FLIPRATE_MODEL=mypackage:my_clf uvicorn fliprate.api:app
```

| Endpoint | Auth | Description |
|---|---|---|
| `POST /jobs/analyze` | X-API-Key | Start gate analysis, get `job_id` |
| `POST /jobs/test` | X-API-Key | Start robustness test |
| `GET /jobs/{id}/stream` | none (id is a secret token) | Server-Sent Events stream |
| `GET /health` | none | `{"status":"ok"}` |
| `GET /ready` | none | 503 until model loaded |
| `GET /metrics` | none | Prometheus counters |
| `GET /` | none | Web console |

Environment variables:

| Variable | Default | Effect |
|---|---|---|
| `FLIPRATE_MODEL` | toy model | `module:function` path to your classifier |
| `FLIPRATE_API_KEY` | none | Require key on POST endpoints |
| `FLIPRATE_CORS` | `*` | Allowed origins (set in production) |
| `FLIPRATE_RATE_LIMIT` | `60` | Requests per minute per IP/key |
| `FLIPRATE_MAX_CONCURRENT` | `2` | Max simultaneous jobs |

Docker:
```bash
docker build -t fliprate .
docker run -p 7860:7860 -e FLIPRATE_MODEL=fliprate.hf_models:xlmt_sentiment fliprate
```

---

## Documentation

| Document | Contents |
|---|---|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | All CLI commands, install steps, environment variables |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Every function with parameters and examples |
| [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | Architecture, design decisions, full module map |
| [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) | Experiment protocol (data sourcing, metrics, caveats) |
| [`docs/RELEASE_AND_DEPLOY.md`](docs/RELEASE_AND_DEPLOY.md) | GitHub → PyPI → HuggingFace deployment |
| [`CHANGELOG.md`](CHANGELOG.md) | Version history |

---

## Limitations

- Romanized (Latin-script) text only. Devanagari and other native scripts go to REVIEW.
- Variants are rule-generated, not sampled from real users. Real typing noise is wider than 4 change types.
- Built-in lexicon is customer-support focused. Use `--no-lexicon` for other domains.
- The Dakshina/Aksharantar rules come from Wikipedia text — social-media vocabulary differs.
- Do not publish numbers from the bundled 40-message demo sample.

---

## Datasets used

If you publish work using the multi-language rules, please cite:

> Roark et al. (2020). "Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset."
> LREC 2020. https://aclanthology.org/2020.lrec-1.294

> Mittal et al. (2022). "Aksharantar: Towards Building Open Transliteration Tools for the Next Billion Users."
> ArXiv. https://arxiv.org/abs/2205.03018

---

## License

MIT — see [LICENSE](LICENSE).

The Dakshina dataset is CC BY-SA 4.0. The Aksharantar dataset is CC BY 4.0.
Neither dataset is included in this package — download separately.
