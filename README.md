# FlipRate

**Does your NLP model change its answer when the same message is typed differently?**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://github.com/ShivSupane/fliprate/actions/workflows/ci.yml/badge.svg)](https://github.com/ShivSupane/fliprate/actions/workflows/ci.yml)

Built by **Shiv Supane**

---

## The problem

People who text in Hindi, Marathi, Bengali, Tamil, Telugu, and other Indian languages
using Roman (English) script have no standard spelling. The same word gets typed dozens of ways:

| Word | Romanized variants |
|---|---|
| नही (no) | `nahi` · `nhi` · `nahin` · `nahee` |
| बहुत (very) | `bahut` · `bohot` · `bhut` · `boht` |
| आवडला (liked) | `avadla` · `avdla` · `avadlaaa` |

Most NLP models treat these as completely different inputs and give different answers —
even though the meaning is identical. **FlipRate measures that problem, quantifies it
with confidence intervals, and gives you tools to fix it.**

```
User types:   "mala product avadlaaa nahi"  →  model says: product_feedback ✓
Same meaning: "mala product avdla nhi"      →  model says: human_support    ✗  ← FLIP
```

---

## Install

```bash
pip install fliprate                    # core library (zero dependencies)
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

### The Happy-Path API (Beginner Friendly)

```python
import fliprate as fr

# 1-line quality check
verdict = fr.quick_check(my_model, "mala product avadlaaa nahi")
print(verdict.is_safe)       # True / False
print(verdict.prediction)    # Model's answer on the canonical text
print(verdict.reason)        # Plain-English explanation

# Clean messy text directly
clean = fr.normalize_text("Mala product avadlaaa nhi", lowercase=False)
# -> "Mala product avadla nahi"

# Predict safely with automatic spelling normalization fallback
pred = fr.safe_predict(my_model, "ye delivery nhi aayi")

# Audit model robustness on a dataset
report = fr.check_model(my_model, "data/test.csv")
print(report.summary())
```

### Full Robustness Evaluation (Detailed Metrics)

```python
import fliprate as fr

report = fr.test(my_model, "data/test.csv")   # my_model: any function text -> label
print(report.summary())
```

Example output *(illustrative — your numbers will differ)*:
```
Messages: 500 | safe variants tested: 2500
Accuracy on originals:      68.4%  (macro-F1 0.672)
Accuracy on safe variants:  67.1%  (macro-F1 0.658)
Accuracy drop:              -1.3%  95% CI [-2.1%, -0.5%]
Flip rate:                   6.2%  95% CI [5.1%, 7.4%]
Most fragile change type:   repetition (flip rate 9.8%)
```

### Protect your model

```python
safe_model = fr.protect(my_model)   # normalizer in front
report2 = fr.test(safe_model, "data/test.csv")

diff = fr.paired_difference(report, report2)
print(f"Flip rate change: {diff['difference']:+.1%}")
print(f"95% CI: {diff['ci']}")
```

### Real-time gate

```python
result = fr.gate(my_model, "mala product avadlaaa nahi")
print(result.decision)          # PASS / TRANSFORM / REVIEW
print(result.prediction)        # model's answer on the (possibly normalized) text
print(result.reason)            # human-readable explanation
print(result.flipped_variants)  # variants that caused model flip (if unstable)
```

### Production-friendly convenience wrappers

```python
import fliprate as fr

# One-line runtime quality gate
res = fr.guard(my_model, "mala order avadlaaa nahi")

# One-line model audit
report = fr.audit(my_model, "data/test.csv")

# Mine spelling rule candidates from raw text
cands = fr.learn_rules_from_texts([
    "mala avadla nahi",
    "mala avdla nahi",
    "mala avadlaa nahi",
])
```

### Devanagari & Script-Aware Gating (v0.4.0)

FlipRate natively supports Devanagari (Hindi, Marathi, etc.) and mixed-script text:

```python
# Detect script
fr.detect_script("मला प्रॉडक्ट आवडला")      # "devanagari"
fr.detect_script("product bohot accha hai")  # "latin"
fr.detect_script("ye product बहुत accha hai") # "mixed"

# Gate Devanagari and mixed text with safe orthographic perturbations
result = fr.gate(my_model, "मला प्रॉडक्ट आवडला नाही")
print(result.decision)   # PASS / TRANSFORM / REVIEW
```

Devanagari perturbations include:
- **Unicode normalization**: precomposed vs decomposed nuktas (e.g., U+0958..U+095F vs base + U+093C)
- **Digits**: bidirectional swap between ASCII (`0-9`) and Devanagari (`०-९`) digits
- **Punctuation**: danda (`।`) and double danda (`॥`) interchange with Western punctuation
- **Curated spelling**: meaning-safe nukta variations (`ज़रा`/`जरा`, `फ़ोन`/`फोन`) and Hindi/Marathi negations (`नहीं`, `नाही`, `नको`)
- **Opt-in cross-script**: optional `cross_script=True` for high-confidence 1-to-1 transliterated pairs

---

## Web Console

```bash
uvicorn fliprate.api:app --reload          # toy model, instant start
# then open http://localhost:8000
```

Two tabs:
- **Playground** — paste a message, watch the gate decision stream live
- **Robustness Lab** — upload a labeled CSV, get a live flip-rate chart

Zero-dependency vanilla HTML/JS. No Streamlit, no npm, no build step.

---

## CLI

```bash
fliprate gate "mala product avadlaaa nahi"
fliprate perturb "delivery nahi aaya"
fliprate normalize "mala product avadlaaa nhi"
fliprate test --data mydata.csv
fliprate test --data mydata.csv -n 10 --protect --save report.json
fliprate check-data testdata.csv --against traindata.csv
fliprate split mydata.csv --out-dir data/
fliprate mine mydata.csv --out candidates.csv
```

---

## Plug in any model

```python
# Hugging Face (lazy-loaded, batched)
clf = fr.HFTextClassifier(
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    lowercase_labels=True,
    preprocess=fr.twitter_preprocess,
)

# scikit-learn
from fliprate.baselines import train_tfidf
clf = train_tfidf(train_rows, analyzer="word")   # or "char_wb"

# Any LLM
llm = fr.with_retries(fr.llm_classifier(
    my_openai_fn, labels=["positive", "neutral", "negative"]
))
```

---

## Multi-language rules

FlipRate ships with 73 core HIGH-confidence groups for immediate use.
For broader coverage, load from two research datasets:

```python
# Built-in per-language groups (12 Indian languages + 1 shared cross-lingual group)
# hindi, urdu, marathi, bengali, tamil, telugu, kannada, malayalam, gujarati, punjabi, odia, assamese, shared
tamil_rules = fr.DEFAULT_RULES.extended(groups=fr.LANGUAGE_GROUPS["tamil"])

# Dakshina (Google Research, CC BY-SA 4.0) — 225K groups, 12 languages
# Codes: hi ur mr bn ta te kn ml gu pa sd si
rules = fr.dakshina_rules("ta")           # Tamil
rules = fr.dakshina_rules("hi", "mr")    # Hindi + Marathi

# Aksharantar (AI4Bharat, CC BY 4.0) — 428K groups, 20 languages
# Codes: hin guj tam tel kan mal mar pan ben urd ori asm kas kok mai mni nep san brx sid
rules = fr.aksharantar_rules("hin")               # Hindi: 58K groups
rules = fr.aksharantar_rules("tam", "tel", "kan") # South India

# Use in any call
report = fr.test(my_model, "data/test.csv", rules=rules)
```

---

## How it works

**Safe variant:** a rewriting of a message that keeps the same canonical words
(up to reviewed equivalent spellings) and the same negation count. Adding or
removing a negation word (`nahi`, `nako`, `illa`, …) is never allowed.

**Flip rate:** `variants_where_answer_changed / total_safe_variants`.
Bootstrap 95% CIs are computed over messages, not individual variants.

**Gate decisions:**

| Decision | When |
|---|---|
| **PASS** | ≥ 80% of safe variants get the same answer. |
| **TRANSFORM** | Unstable, but the built-in normalizer fixes it. Sends cleaned text to model; keeps original. |
| **REVIEW** | Unstable and normalization does not help. Route to a human. |

---

## Experiment results (v0.2)

Tested on a processed copy of SentiMix Hinglish sentiment (2,997 held-out messages,
5 rule-generated variants per message). **Read the caveats below before citing these numbers.**

| Model | Mode | Accuracy | Flip rate (95% CI) |
|---|---|---|---|
| xlm-roberta-base-sentiment | raw | 59.2% | 7.3% (6.7–7.9%) |
| xlm-roberta-base-sentiment | **protected** | 59.1% | **3.5%** (3.1–3.9%) |
| TF-IDF word n-gram | raw | 65.9% | 4.2% (3.8–4.6%) |
| TF-IDF word n-gram | **protected** | 65.6% | **3.2%** (2.9–3.6%) |

Paired flip-rate change: **−3.80 pts** [−4.22, −3.39]. Accuracy: −0.03 pts [−0.23, +0.17].

> **Important caveats (from [`evidence/EVIDENCE.md`](evidence/EVIDENCE.md)):**
> - The SentiMix copy used here is lowercased, stopword-stripped and Porter-stemmed — not natural user typing.
> - The spelling-swap flip rate drops to 0 % after protection **by construction**: the variants and the normalizer use the same rule list, so normalization simply undoes its own perturbations. This shows the normalizer is internally consistent, not that it fixes real-world spelling noise.
> - The spelling rules have not been reviewed by a fluent speaker.
> - Only one transformer was tested, zero-shot.
>
> These numbers are a proof-of-concept baseline. Treat them as lower bounds on what a fair evaluation on natural, unprocessed text would show.

Full evidence and raw result files: [`evidence/`](evidence/)

---

## Production deployment

```bash
# With API key and restricted CORS
FLIPRATE_API_KEY=secret FLIPRATE_CORS=https://yoursite.com \
  uvicorn fliprate.api:app --host 0.0.0.0 --port 8000 \
  --proxy-headers --forwarded-allow-ips="*"
```

**Environment variables:**

| Variable | Default | Effect |
|---|---|---|
| `FLIPRATE_MODEL` | toy model | `module:function` path to your classifier |
| `FLIPRATE_API_KEY` | none (open) | Require key on POST endpoints |
| `FLIPRATE_CORS` | `*` | Allowed origins — always restrict in production |
| `FLIPRATE_RATE_LIMIT` | `60` | Requests per minute per IP/key |
| `FLIPRATE_MAX_CONCURRENT` | `2` | Max simultaneous jobs |
| `FLIPRATE_MAX_TEXT_CHARS` | `2000` | Max chars per analyze request |
| `FLIPRATE_MAX_ROWS` | `2000` | Max rows per test request |

Docker:
```bash
docker build -t fliprate .
docker run -p 7860:7860 \
  -e FLIPRATE_MODEL=fliprate.hf_models:xlmt_sentiment \
  -e FLIPRATE_API_KEY=secret \
  fliprate
```

**Production notes:**
- `normalize(text, lowercase=False)` accurately preserves word casing (`Title`, `UPPER`, `lower`).
- Single-instance deployments run entirely in-process with zero external database dependencies.
- In distributed environments behind load balancers, note that in-memory rate limiting and task queues are local to each process.

---

## Documentation

| File | Contents |
|---|---|
| [`docs/COOKBOOK.md`](docs/COOKBOOK.md) | Practical real-world recipes (sentiment, chatbot, ticket routing, custom slang) |
| [`docs/DEPLOYMENT_GUIDE.md`](docs/DEPLOYMENT_GUIDE.md) | Production architecture, when to use it, distributed scaling, PII & security |
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Every command, env var, and code example |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Full function reference with parameters |
| [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | Architecture, design decisions, module map |
| [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) | Experiment protocol and caveats |
| [`docs/RELEASE_AND_DEPLOY.md`](docs/RELEASE_AND_DEPLOY.md) | PyPI and deployment guide |
| [`evidence/`](evidence/) | Raw result files and EVIDENCE.md |

---

## Limitations

- Supported scripts: Roman (Latin), Devanagari, and mixed code-switched text. Other Indic scripts (Tamil, Telugu, Bengali script) route to `REVIEW` with an informative reason.
- Rule-generated variants only — real typing noise is wider than 6 change types.
- Spelling rules have not been reviewed by a fluent speaker for any language.
- Customer-support domain lexicon; use `--no-lexicon` for other domains.
- Dakshina data is from Wikipedia. Aksharantar draws on Wikipedia, IndicCorp, Samanantar, and other corpora — but not all pairs are manually annotated.
- Do not publish numbers from the bundled 40-message demo sample.

---

## Datasets

If you use the multi-language rules in published work, cite:

> Roark et al. (2020). "Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset." LREC 2020. https://aclanthology.org/2020.lrec-1.294

> Mittal et al. (2022). "Aksharantar: Towards Building Open Transliteration Tools for the Next Billion Users." https://arxiv.org/abs/2205.03018

---

## License

MIT — see [LICENSE](LICENSE). © 2025 Shiv Supane

The Dakshina dataset is CC BY-SA 4.0. The Aksharantar dataset is CC BY 4.0.
Neither is included in this package — download separately.
