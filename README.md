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

### Test your model in 3 lines

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
print(result.decision)     # PASS / TRANSFORM / REVIEW
print(result.prediction)   # model's answer on the (possibly normalized) text
print(result.reason)       # human-readable explanation
```

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

---

## Documentation

| File | Contents |
|---|---|
| [`docs/QUICKSTART.md`](docs/QUICKSTART.md) | Every command, env var, and code example |
| [`docs/API_REFERENCE.md`](docs/API_REFERENCE.md) | Full function reference with parameters |
| [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md) | Architecture, design decisions, module map |
| [`docs/EXPERIMENT.md`](docs/EXPERIMENT.md) | Experiment protocol and caveats |
| [`docs/RELEASE_AND_DEPLOY.md`](docs/RELEASE_AND_DEPLOY.md) | PyPI and deployment guide |
| [`evidence/`](evidence/) | Raw result files and EVIDENCE.md |

---

## Limitations

- Romanized Latin-script text only. Devanagari and other native scripts go to REVIEW.
- Rule-generated variants only — real typing noise is wider than 4 change types.
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
