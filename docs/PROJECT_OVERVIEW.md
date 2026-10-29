# FlipRate — Complete Project Overview

**Author: Shiv Supane**  
**Version: 1.0.3**  
**Status: Production/Stable**  

> See the runnable references: `scratch/fliprate_showcase.py` (local) and
> [`examples/fliprate_colab_showcase.ipynb`](../examples/fliprate_colab_showcase.ipynb)
> (Google Colab — downloads real Dakshina + Aksharantar datasets and runs live HF + LLM models).

---

## What is FlipRate?

FlipRate is a Python library that answers one question:

> **Does your NLP model change its answer when the same message is typed differently?**

People who text in Hindi, Marathi, Bengali, Tamil, Telugu, and other Indian languages
using Roman (English) script have no standard spelling. The same word gets typed
dozens of ways:

- "nahi" / "nhi" / "nahin" / "nahee" — all mean "no"
- "bahut" / "bohot" / "bhut" / "boht" — all mean "very"
- "avadlaaa" — stretched version of "avadla" (liked, in Marathi)

Most NLP models treat these as completely different inputs and give different answers —
even though the meaning is identical. FlipRate **measures** this problem, **quantifies**
it with confidence intervals, and gives you a normalizer to **fix** it.

---

## The Core Problem

```
User types: "mala product avadlaaa nahi"
Model says: product_feedback

Same user types: "mala product avadla nhi"
Model says: human_support   ← WRONG, same message!
```

This is called a **flip**. The flip rate is the fraction of meaning-preserving
variants for which the model gives a different answer than on the original.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FlipRate Library                         │
│                                                                   │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐  │
│  │  rules.py   │    │perturbation.py│   │  normalization.py   │  │
│  │             │    │              │    │                     │  │
│  │ SAFE_GROUPS │───▶│ perturb()    │    │ normalize()         │  │
│  │ NEGATIONS   │    │ 4 strategies │    │ run-collapse +      │  │
│  │ AMBIGUOUS   │    │ spelling     │    │ canonical lookup    │  │
│  │ RuleSet     │    │ repetition   │    │                     │  │
│  │             │    │ punctuation  │    │ NormalizeResult     │  │
│  │ 73 core     │    │ case         │    │ .original           │  │
│  │ groups      │    │              │    │ .normalized         │  │
│  │             │    │ Variant(     │    │ .changes            │  │
│  │ Dakshina:   │    │  text, kind) │    └─────────────────────┘  │
│  │ 225K groups │    └──────────────┘             │               │
│  │             │             │                   │               │
│  │ Aksharantar:│             ▼                   ▼               │
│  │ 428K groups │    ┌──────────────────────────────────────────┐  │
│  └─────────────┘    │              evaluation.py               │  │
│                     │                                          │  │
│                     │  iter_test() → yields progress events    │  │
│                     │  test()      → returns Report            │  │
│                     │                                          │  │
│                     │  Report:                                 │  │
│                     │    .flip_rate          (with 95% CI)     │  │
│                     │    .accuracy_original                    │  │
│                     │    .accuracy_variants                    │  │
│                     │    .macro_f1_variants                    │  │
│                     │    .accuracy_drop                        │  │
│                     │    .by_kind            (per change type) │  │
│                     │    .flips              (examples)        │  │
│                     │    .metadata           (reproducibility) │  │
│                     └──────────────────────────────────────────┘  │
│                                        │                          │
│                     ┌──────────────────┴──────────────────────┐   │
│                     │               gating.py                  │   │
│                     │                                          │   │
│                     │  iter_gate() → yields events             │   │
│                     │  gate()      → returns GateResult        │   │
│                     │                                          │   │
│                     │  Decision logic:                         │   │
│                     │    100% stable → PASS                    │   │
│                     │    unstable + norm helps → TRANSFORM     │   │
│                     │    unstable + norm fails → REVIEW        │   │
│                     │    non-Latin / empty → REVIEW            │   │
│                     └──────────────────────────────────────────┘   │
│                                                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────┐  │
│  │  models.py   │   │  adapters.py │   │    experiment.py       │  │
│  │              │   │              │   │                        │  │
│  │ Predictor    │   │ HFTextClass- │   │ run_experiment()       │  │
│  │ - batching   │   │   ifier      │   │ - raw vs protected     │  │
│  │ - caching    │   │ SklearnAdapt │   │ - paired bootstrap     │  │
│  │ - dedup      │   │ LLMClassifier│   │ - markdown report      │  │
│  │              │   │ with_retries │   │                        │  │
│  │ ProtectedModel│  └──────────────┘   └────────────────────────┘  │
│  │ protect()    │                                                   │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Web Service Layer                           │
│                                                                   │
│  api.py (FastAPI)           service.py                           │
│  ├── POST /jobs/analyze  →  JobStore (one-time tokens)           │
│  ├── POST /jobs/test     →  RateLimiter (60 req/min/IP)          │
│  ├── GET  /jobs/{id}/stream → Server-Sent Events stream          │
│  ├── GET  /health                                                 │
│  ├── GET  /ready         →  503 until model warmed up            │
│  ├── GET  /metrics       →  Prometheus text format               │
│  └── GET  /              →  web/index.html console               │
│                                                                   │
│  web/index.html (zero JS dependencies)                           │
│  ├── Playground tab  — live gate analysis                        │
│  └── Robustness Lab  — live progress chart + flip table          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Language Data Layer                     │
│                                                                   │
│  rules.py CORE_GROUPS (73 groups, 224 keys)                      │
│    └── 12 Indian languages, HIGH-confidence, hand-curated        │
│                                                                   │
│  dakshina.py → dakshina_rules("ta")                              │
│    └── Google Dakshina v1.0 (CC BY-SA 4.0)                       │
│        225K groups from Wikipedia, 12 languages                  │
│        Native-speaker annotated romanizations                    │
│                                                                   │
│  aksharantar.py → aksharantar_rules("hin","tam","tel")           │
│    └── AI4Bharat Aksharantar (CC BY 4.0)                         │
│        428K groups from 24M word pairs, 21 languages             │
│        Sources: IndicCorp, Samanantar, Dakshina, AK-Freq         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

| Module | Purpose | Key exports |
|---|---|---|
| `rules.py` | Language rules — the heart of FlipRate | `RuleSet`, `DEFAULT_RULES`, `LANGUAGE_GROUPS`, `CORE_GROUPS` |
| `perturbation.py` | Generate safe spelling variants | `perturb()`, `Variant`, `KINDS` |
| `normalization.py` | Conservative text normalizer | `normalize()`, `NormalizeResult` |
| `gating.py` | Real-time PASS/TRANSFORM/REVIEW decision | `gate()`, `iter_gate()`, `GateResult` |
| `evaluation.py` | Batch robustness measurement | `test()`, `iter_test()`, `Report`, `paired_difference()` |
| `models.py` | Model wrapper with batching + cache | `Predictor`, `protect()`, `ProtectedModel` |
| `adapters.py` | Ready-made adapters for HF/sklearn/LLM | `HFTextClassifier`, `from_sklearn()`, `LLMClassifier` |
| `experiment.py` | Run raw vs protected comparison | `run_experiment()`, `ExperimentResult` |
| `dataset.py` | Dataset utilities | `check_dataset()`, `split_dataset()`, `mine_variants()` |
| `dakshina.py` | Dakshina dataset loader | `dakshina_rules()`, `load_dakshina_groups()` |
| `aksharantar.py` | Aksharantar dataset loader | `aksharantar_rules()`, `load_aksharantar_groups()` |
| `api.py` | FastAPI web service | `app` (uvicorn entry point) |
| `service.py` | Business logic (testable without FastAPI) | `JobStore`, `RateLimiter`, `Metrics` |
| `cli.py` | `fliprate` command-line tool | `main()` |
| `demo.py` | Toy model + sample data | `keyword_classifier`, `sample_dataset()` |
| `events.py` | Server-Sent Events helpers | `make_event()`, `to_sse()`, `drain()` |

---

## The RuleSet — Central Design Decision

Everything in FlipRate depends on one concept: **safe variants**.

A variant of a message is "safe" if and only if:
1. It has exactly the same canonical words (up to known-equivalent spellings)
2. It has exactly the same number of negation words

This is enforced by `RuleSet.is_safe_variant(original, variant)`.

The `RuleSet` knows:
- **SAFE_GROUPS** — which romanized spellings are the same word (e.g. `nahi/nhi/nahin`)
- **NEGATIONS** — tokens that reverse meaning (never add or remove these)
- **AMBIGUOUS** — tokens that must never be rewritten (e.g. `nai` = "no" or "new" depending on context)

```
SAFE_GROUPS drives:
  ├── perturbation.py  → only generates variants within safe groups
  ├── normalization.py → maps variants to canonical form
  ├── gating.py        → checks if normalized text is a safe variant
  └── evaluation.py    → determines if predictions changed on a safe variant
```

---

## Perturbation Strategies

| Strategy | What it does | Example |
|---|---|---|
| `spelling` | Swap a word for another member of its safe group | `nahi` → `nhi` |
| `repetition` | Stretch a vowel (realistic emphasis) | `nahi` → `naahii` |
| `punctuation` | Strip or double trailing punctuation | `!` → `!!` or removed |
| `case` | Lowercase / uppercase / capitalize the whole message | `MALA PRODUCT NAHI` |

Every variant is checked with `is_safe_variant` before being returned.
Variants that would add or remove a negation word are rejected.

---

## Gate Decision Logic

```
Input text
    │
    ├─ Empty / mostly non-Latin? ──────────────────────────► REVIEW
    │
    ├─ Generate N safe variants
    ├─ Run model on original + all variants
    ├─ raw_stability = fraction with same answer as original
    │
    ├─ raw_stability == 1.0? ──────────────────────────────► PASS (perfect)
    │
    ├─ Apply normalization
    ├─ Is normalized text itself a safe variant?
    ├─ norm_stability = fraction of variants stable after normalization
    │
    ├─ norm_stability >= threshold AND norm_stability > raw?
    │       │
    │       └─ YES ────────────────────────────────────────► TRANSFORM
    │              (send normalized text to model)
    │
    ├─ raw_stability >= threshold? ────────────────────────► PASS (with warning)
    │
    └─ otherwise ──────────────────────────────────────────► REVIEW
```

---

## Evaluation Metrics

| Metric | What it measures |
|---|---|
| **Flip rate** | % of safe variants where the model changed its answer. Measures instability. |
| **Accuracy (originals)** | Standard accuracy on the original messages. |
| **Accuracy (variants)** | Accuracy on spelling variants — shows generalization. |
| **Accuracy drop** | Originals accuracy minus variants accuracy. Shows degradation. |
| **Macro-F1** | Class-balanced F1 on variants. |
| **By-kind flip rate** | Per change type (spelling, repetition, punctuation, case). |

All rates include **95% bootstrap confidence intervals** over messages.
`paired_difference()` computes a paired CI for raw vs protected comparison.

---

## Experiment Results (v0.2)

Run on SentiMix Hinglish sentiment test set (2,997 messages), 5 variants per message.

| Model | Mode | Accuracy | Flip rate (95% CI) |
|---|---|---|---|
| cardiffnlp/twitter-xlm-roberta-base-sentiment | raw | 59.2% | 7.3% (6.7–7.9%) |
| cardiffnlp/twitter-xlm-roberta-base-sentiment | protected | 59.1% | **3.5%** (3.1–3.9%) |
| TF-IDF word n-gram | raw | 65.9% | 4.2% (3.8–4.6%) |
| TF-IDF word n-gram | protected | 65.6% | 3.2% (2.9–3.6%) |

Paired effect of normalizer: flip rate **−3.80 pts** [−4.22, −3.39]. Accuracy: −0.03 pts [−0.23, +0.17] (no significant change).

---

## Multi-Language Coverage

| Language | Dakshina groups | Aksharantar groups | Combined keys |
|---|---|---|---|
| Hindi | 13,589 | 76,816 | ~150K |
| Tamil | 20,865 | 33,518 | ~78K |
| Telugu | 19,076 | 33,454 | ~70K |
| Kannada | 12,283 | 23,041 | ~45K |
| Malayalam | 15,207 | 23,480 | ~51K |
| Marathi | 18,029 | 24,537 | ~54K |
| Bengali | 25,150 | 53,543 | ~130K |
| Gujarati | 24,003 | 35,421 | ~65K |
| Punjabi | 20,724 | 29,811 | ~63K |
| Urdu | 25,957 | 25,826 | ~76K |
| + 11 more | — | Odia, Assamese, Kashmiri, Konkani, Maithili, Manipuri, Nepali, Sanskrit, Bodo, Sindhi, Sinhala |

All groups extracted from real human-annotated transliteration data.
The DEFAULT_RULES contains 73 conservative HIGH-confidence groups for immediate use without any dataset download.

---

## File Structure

```
fliprate/
├── src/fliprate/
│   ├── __init__.py          public API
│   ├── rules.py             language rules (RuleSet, SAFE_GROUPS, etc.)
│   ├── perturbation.py      variant generation
│   ├── normalization.py     text normalizer
│   ├── gating.py            PASS/TRANSFORM/REVIEW gate
│   ├── evaluation.py        robustness test + Report
│   ├── models.py            model wrapper (batching, caching, protection)
│   ├── adapters.py          HF / sklearn / LLM adapters
│   ├── experiment.py        raw vs protected comparison
│   ├── dataset.py           data tools (split, mine, audit, check)
│   ├── dakshina.py          Dakshina dataset loader
│   ├── aksharantar.py       Aksharantar dataset loader
│   ├── api.py               FastAPI web service
│   ├── service.py           service layer (JobStore, RateLimiter, Metrics)
│   ├── cli.py               fliprate command-line tool
│   ├── demo.py              toy model + sample data
│   ├── events.py            SSE helpers
│   ├── hf_models.py         ready-made HF model definition
│   ├── baselines.py         TF-IDF sklearn baselines
│   └── web/index.html       zero-dependency web console
│
├── tests/                   80 unit tests
├── scripts/
│   ├── build_rules_from_dakshina.py     extract groups from Dakshina
│   ├── build_rules_from_aksharantar.py  extract groups from Aksharantar
│   └── find_todos.py                    find remaining placeholders
├── experiments/
│   ├── run_experiment.py                full experiment runner
│   └── prepare_sentimix_processed.py   SentiMix data prep
├── docs/
│   ├── QUICKSTART.md        commands + code examples
│   ├── PROJECT_OVERVIEW.md  this file
│   ├── API_REFERENCE.md     full API docs
│   ├── EXPERIMENT.md        experiment protocol
│   └── RELEASE_AND_DEPLOY.md deployment guide
├── data/                    generated rules (gitignored)
├── dakshina_dataset_v1.0/   dataset (gitignored)
├── Aksharantar_All_Languages/ dataset (gitignored)
├── pyproject.toml
├── Dockerfile
├── CHANGELOG.md
└── README.md
```

---

## What was built — chronological

1. **Core algorithm** (v0.1): RuleSet, perturbation, normalization, gating, evaluation
2. **Web service** (v0.1): FastAPI + SSE streaming + web console
3. **Model adapters** (v0.2): HuggingFace, sklearn, LLM, retries
4. **Experiment runner** (v0.2): raw vs protected, paired bootstrap
5. **Data tools** (v0.2): mine, audit, split, check-data
6. **Hardened API** (v0.2): API key, rate limit, concurrency cap, /ready, /metrics
7. **Multi-language rules** (v0.2.1): 12 languages in LANGUAGE_GROUPS
8. **Dakshina integration** (v0.2.1): 225K groups from Google Dakshina dataset
9. **Aksharantar integration** (v0.2.1): 428K groups from AI4Bharat Aksharantar (21 languages)
10. **Improved normalization** (v0.2.1): consonant/vowel run distinction
11. **New web console** (v0.2.1): replaced Streamlit with built-in zero-dependency console
12. **Devanagari & Script-Aware Gating** (v0.4.0): native Devanagari support, script detection, unicode/digit/danda/curated spelling perturbations, negation invariance, canonical form gate stability pass, opt-in cross-script transliteration

---

## Known Limitations

- Supported scripts: Roman (Latin), Devanagari, and mixed code-switched text. Other Indic scripts (Tamil, Telugu, Bengali native scripts) route to `REVIEW` with an informative reason.
- Variants are rule-generated, not sampled from real users. Real typing noise is wider than four change types.
- The built-in customer-support lexicon is domain-specific. Use `--no-lexicon` for other domains.
- The Dakshina and Aksharantar rules are from Wikipedia text — social-media vocabulary is different.
- The bundled 40-message sample and toy model are for demos only. Do not publish numbers from them.
- CI tests across Python 3.9–3.13.

---

## Datasets Used

| Dataset | Source | Licence | Used for |
|---|---|---|---|
| SentiMix (processed copy) | SemEval 2020 Task 9 | Research use | Experiment validation |
| Dakshina v1.0 | Google Research | CC BY-SA 4.0 | Spelling variant groups (225K) |
| Aksharantar | AI4Bharat | CC BY 4.0 | Spelling variant groups (428K) |

Cite the datasets if you use the multi-language rules in published work.
See `docs/RELEASE_AND_DEPLOY.md` for citation details.
