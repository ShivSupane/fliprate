# FlipRate — API Reference

**Author: Shiv Supane** | Version 0.4.0

All public functions are available as `fliprate.<name>` after `import fliprate as fr`.

---

## Gate functions

### `fr.gate(model, text, ...)`

Run the PASS / TRANSFORM / REVIEW quality gate on a single message. Supports Roman (Latin), Devanagari, and mixed code-switched text.

```python
result = fr.gate(my_model, "mala product avadlaaa nahi")

result.decision          # "PASS", "TRANSFORM", or "REVIEW"
result.reason            # human-readable explanation
result.prediction        # model's answer on the (possibly normalized) text
result.text_used         # the text actually sent to the model
result.original_text     # the original input
result.raw_stability     # 0.0–1.0, fraction of variants matching original
result.normalized_stability  # stability after normalization (or None)
result.n_variants        # how many safe variants were tested
result.warnings          # list of warning strings (ambiguous tokens etc.)
result.flipped_variants  # list of variant texts that changed the answer

result.to_dict()         # dict representation for logging
result.to_json()         # JSON string representation
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `callable` | required | Any function `text → label` |
| `text` | `str` | required | The message to analyse |
| `n_variants` | `int` | `8` | Number of safe variants to test |
| `threshold` | `float` | `0.8` | Minimum stability for PASS/TRANSFORM |
| `min_variants` | `int` | `3` | Below this count, stability is flagged as weak |
| `seed` | `int` | `0` | For reproducibility |
| `kinds` | `list[str]` | all 4 | Restrict to `["spelling","repetition","punctuation","case"]` (Latin) or Devanagari kinds |
| `rules` | `RuleSet` | `DEFAULT_RULES` | Language rules to use |
| `on_event` | `callable` | `None` | Called with each event dict during streaming |
| `cache` | `bool` | `True` | Cache model predictions |
| `on_error` | `str` | `"raise"` | `"raise"` to re-raise model exceptions, or `"review"` to yield REVIEW |
| `script_aware` | `bool` | `True` | Enable script detection and native Devanagari/mixed script support (`False` for legacy Latin-only) |

> **Canonical Form Handling:** When an input is already in canonical normalized form (`normalize(text).normalized == text`) and raw variants cause the model to flip, `gate()` evaluates the stability of the normalized variants. If they are stable, it returns `PASS` with the warning `"input is already in canonical form; model is sensitive to other spellings (protect() recommended)"` rather than routing to `REVIEW`.

### `fr.agate(model, text, ...)` — async version

Asynchronous equivalent of `gate()`. Runs in a worker thread via `asyncio.to_thread` to avoid blocking the event loop:

```python
result = await fr.agate(my_model, "kya haal hai")
```

### `fr.iter_gate(model, text, ...)` — streaming version

Same parameters. Returns a generator that yields event dicts as the gate runs.
Use `fr.drain(gen, on_event)` to consume it and get the `GateResult`.

---

## Test functions

### `fr.test(model, dataset, ...)`

Run a full robustness test and return a `Report`.

```python
report = fr.test(my_model, "data/test.csv")
report = fr.test(my_model, "data/test.csv", n_variants=10, rules=fr.dakshina_rules("ta"))
print(report.summary())
report.save("results/report.json")
```

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `callable` | required | Any function `text → label` |
| `dataset` | `str / Path / list` | required | Path to CSV (text,label) or list of tuples |
| `n_variants` | `int` | `5` | Variants per message |
| `kinds` | `list[str]` | all 4 | Which change types to use |
| `seed` | `int` | `0` | Random seed for reproducibility |
| `bootstrap` | `int` | `1000` | Bootstrap resamples for CIs |
| `rules` | `RuleSet` | `DEFAULT_RULES` | Language rules |
| `on_event` | `callable` | `None` | Progress callback |
| `cache` | `bool` | `True` | Cache model predictions |
| `batch_messages` | `int` | `8` | Messages processed per batch |

### `Report` object

```python
report.n_messages          # number of messages tested
report.n_variants          # total safe variants tested
report.accuracy_original   # accuracy on original messages
report.accuracy_variants   # accuracy on safe variants
report.macro_f1_original   # macro-F1 on originals
report.macro_f1_variants   # macro-F1 on variants
report.flip_rate           # fraction of variants that changed the answer
report.flip_rate_ci        # (low, high) 95% confidence interval
report.accuracy_drop       # accuracy_original - accuracy_variants
report.accuracy_drop_ci    # (low, high) 95% CI
report.by_kind             # {kind: {flip_rate, accuracy, variants}}
report.flips               # list of {original, variant, kind, original_pred, variant_pred, label}
report.transitions         # top-20 {label_a → label_b: count} flip pairs
report.metadata            # version, timestamp, model, dataset hash, etc.

report.summary()           # printable summary string
report.to_markdown(title)  # markdown table
report.save(path)          # write JSON
report.to_dict()           # dict (for custom serialization)

# Load back
report2 = fr.Report.load("results/report.json")
```

### `fr.paired_difference(report_a, report_b, stat="flip_rate")`

Compare two reports on the **same dataset** (e.g. raw vs protected).
Returns a paired 95% bootstrap interval for `value(b) - value(a)`.

```python
raw_report = fr.test(my_model, dataset)
safe_model = fr.protect(my_model)
protected_report = fr.test(safe_model, dataset)

diff = fr.paired_difference(raw_report, protected_report)
print(diff["difference"])        # e.g. -0.038 (flip rate dropped 3.8 pts)
print(diff["ci"])                # e.g. (-0.042, -0.034)
print(diff["significant"])       # True if CI excludes 0
```

`stat` can be `"flip_rate"`, `"accuracy_original"`, or `"accuracy_variants"`.

---

## Script Detection

### `fr.detect_script(text)`

Identify the dominant script of an input string for script-aware routing.

```python
fr.detect_script("मला प्रॉडक्ट आवडला")       # "devanagari"
fr.detect_script("product bohot accha hai")   # "latin"
fr.detect_script("ye product बहुत accha hai")  # "mixed"
fr.detect_script("வணக்கம்")                   # "other"
```

- Returns: `"latin"`, `"devanagari"`, `"mixed"`, or `"other"`.
- Robust to punctuation, digits, symbols, emojis, and zero-width characters.

---

## Normalization

### `fr.normalize(text, lowercase=True, rules=None)`

Conservatively clean a message. Only repairs things we are sure about.

```python
r = fr.normalize("mala product avadlaaa nhi")
r.original      # "mala product avadlaaa nhi"
r.normalized    # "mala product avadla nahi"
r.changes       # [("avadlaaa", "avadla"), ("nhi", "nahi")]
str(r)          # same as r.normalized
```

What it does:
- Unicode NFC normalization and zero-width noise stripping (`\u200B` ZWSP, `\u200C` ZWNJ, `\u200D` ZWJ, `\uFEFF` BOM)
- Known spelling variants → canonical form (`nhi` → `nahi`)
- Consonant runs of 3+ → 1 (`avadlaaa` → `avadla`)
- Vowel runs of 3+ → 2 (`sooooo` → `soo`)
- Repeated punctuation → 1 (`!!!` → `!`)
- Extra spaces removed
- URLs, emails, hashtags, and mentions are preserved byte-for-byte

What it **never** does: add/remove/reorder words, touch ambiguous tokens like `nai`.

---

## Perturbation

### `fr.perturb(text, n=10, kinds=None, seed=0, rules=None, cross_script=False)`

Generate meaning-preserving spelling variants. Adapts automatically based on the detected script:
- **Latin text**: kinds include `"spelling"`, `"repetition"`, `"punctuation"`, `"case"`
- **Devanagari text**: kinds include `"unicode"` (precomposed/decomposed nuktas), `"digits"` (0-9 ↔ ०-९), `"punctuation"` (danda । ॥ ↔ western punctuation), and `"spelling"` (curated nukta variants)
- **Mixed text**: combines eligible Latin and Devanagari perturbation strategies
- `cross_script`: boolean (default `False`). When `True`, permits deterministic 1-to-1 transliterated word pairs.

```python
# Latin
variants = fr.perturb("mala delivery nahi milali", n=8)
for v in variants:
    print(v.kind, v.text)
# spelling    mala delivery nahee milali
# repetition  mala deliveeery nahi milali
# punctuation mala delivery nahi milali!
# case        MALA DELIVERY NAHI MILALI

# Devanagari
variants = fr.perturb("मला डिलिव्हरी नाही मिळाली", n=5)
for v in variants:
    print(v.kind, v.text)
```

Returns `list[Variant]`. Each `Variant` has `.text` and `.kind`.

Every variant satisfies `rules.is_safe_variant(original, variant)` — same canonical
words, same negation count.

---

## Protection

### `fr.protect(model, rules=None, lowercase=True)`

Wrap a model with the normalizer. Every input is normalized before the model sees it.

```python
safe_model = fr.protect(my_model)
safe_model("mala product avadlaaa nhi")   # internally normalized first
```

---

## Rules

### `fr.DEFAULT_RULES`

The built-in conservative rule set (73 groups, 224 canonical keys). Covers
Hindi, Urdu, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi,
as well as Devanagari safe nukta pairs and comprehensive Hindi/Marathi negations
(`नहीं`, `नही`, `नहिं`, `मत`, `नाही`, `नाहीं`, `नको`, `नकोस`).

### `fr.RuleSet(groups=None, lexicon=None, negations=None, ambiguous=None)`

Build a custom rule set.

```python
from fliprate.rules import LANGUAGE_GROUPS, CORE_GROUPS, RuleSet

# Add Tamil groups on top of defaults
rules = fr.DEFAULT_RULES.extended(groups=LANGUAGE_GROUPS["tamil"])

# Build from scratch
rules = RuleSet(
    groups=CORE_GROUPS + LANGUAGE_GROUPS["telugu"] + LANGUAGE_GROUPS["kannada"]
)

# Load from a reviewed CSV
rules = RuleSet.from_csv("approved_spellings.csv")
# CSV format: spellings column with pipe-separated variants: "nahi|nhi|nahin"
# Optional: status column with approve/reject

# Fingerprint (stored in reports for reproducibility)
rules.fingerprint()   # e.g. "7c63bddd679f"
```

### `fr.LANGUAGE_GROUPS`

Dict of hand-curated groups per language. Keys are language names.
`LANGUAGE_GROUPS["hindi"]`, `LANGUAGE_GROUPS["tamil"]`, etc.

Languages (12 languages + shared): hindi, urdu, marathi, bengali, tamil, telugu, kannada, malayalam, gujarati, punjabi, odia, assamese, and shared.

---

## Dataset loaders (Dakshina + Aksharantar)

### `fr.dakshina_rules(*languages, base=None, ...)`

Build a RuleSet with Dakshina groups. Uses 2-letter language codes.

```python
rules = fr.dakshina_rules("ta")           # Tamil
rules = fr.dakshina_rules("hi", "mr")    # Hindi + Marathi
rules = fr.dakshina_rules()              # all 12 languages
```

Language codes: `hi`, `ur`, `mr`, `bn`, `ta`, `te`, `kn`, `ml`, `gu`, `pa`, `sd`, `si`

### `fr.aksharantar_rules(*languages, base=None, ...)`

Build a RuleSet with Aksharantar groups. Uses 3-letter language codes.

```python
rules = fr.aksharantar_rules("hin")              # Hindi (58K groups)
rules = fr.aksharantar_rules("tam", "tel", "kan") # South India
rules = fr.aksharantar_rules()                    # all 20 languages
```

Language codes: `hin`, `guj`, `tam`, `tel`, `kan`, `mal`, `mar`, `pan`, `ben`, `urd`, `ori`, `asm`, `kas`, `kok`, `mai`, `mni`, `nep`, `san`, `brx`, `sid`

### `fr.load_dakshina_groups(*languages)` / `fr.load_aksharantar_groups(*languages)`

Return raw group lists (for inspection or manual building).

```python
groups = fr.load_dakshina_groups("ta")
print(groups[0])   # ('vanakkam', 'vanakam', 'vanakam')
```

---

## Model adapters

### `fr.HFTextClassifier(model_name, ...)`

Lazy-loaded Hugging Face text-classification pipeline.

```python
clf = fr.HFTextClassifier(
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    lowercase_labels=True,
    preprocess=fr.twitter_preprocess,
)
report = fr.test(clf, dataset)
```

Or via environment: `FLIPRATE_MODEL=fliprate.hf_models:xlmt_sentiment`

### `fr.from_sklearn(pipeline)`

Wrap a fitted sklearn Pipeline.

```python
from fliprate.baselines import train_tfidf
model = train_tfidf(train_rows, "word")   # word n-gram TF-IDF + LogReg
clf = fr.from_sklearn(model.pipeline)
```

### `fr.LLMClassifier(complete_fn, labels, ...)` / `fr.llm_classifier(...)`

For any LLM accessed via a completion function.

```python
llm = fr.llm_classifier(
    my_openai_complete,
    labels=["positive", "neutral", "negative"],
    prompt_template="Classify this text: {text}\nLabel:",
)
```

### `fr.with_retries(fn, retries=3, backoff=0.5)`

Exponential backoff decorator for flaky network calls.

```python
safe_llm = fr.with_retries(llm)
```

---

## Dataset tools

### `fr.read_csv_text(content)`

Parse CSV formatted text (a string, not a path) with headers into a list of `(text, label)` Row objects.

### `fr.load_dataset(source)`

Load a dataset from a CSV file path, a raw CSV string, or directly pass a list of tuples/dicts. Returns `list[tuple[str, str]]`.

---

## Streaming (advanced)

All gate and test functions have `iter_` variants that yield event dicts.
Use `fr.drain(gen, on_event)` to consume and get the return value.

```python
result = fr.drain(
    fr.iter_gate(model, text),
    on_event=lambda ev: print(ev["stage"])
)
# result is a GateResult

report = fr.drain(
    fr.iter_test(model, dataset, n_variants=5),
    on_event=lambda ev: print(ev["stage"], ev.get("done",""))
)
# report is a Report
```

Event stages for gate: `received`, `script_check`, `variants`, `prediction`,
`raw_stability`, `normalized`, `normalized_stability`, `gate`, `done`, `end`

Event stages for test: `started`, `progress`, `done`, `end`

---

## Experiment runner

### `fr.run_experiment(models, dataset, ...)`

Run every model × {raw, protected} on the same data and return an `ExperimentResult`.

```python
from fliprate.baselines import train_tfidf

result = fr.run_experiment(
    models={
        "word_tfidf": train_tfidf(train_rows, "word"),
        "char_tfidf": train_tfidf(train_rows, "char_wb"),
    },
    dataset=test_rows,
    n_variants=5,
    seed=0,
    with_protection=True,
)
print(result.markdown())
result.save("experiments/results/run1/")
```
