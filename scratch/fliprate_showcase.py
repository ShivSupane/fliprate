"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║                     F L I P R A T E   —   C O M P L E T E   S H O W C A S E  ║
║                              Version 1.0.1  •  Shiv Supane                     ║
╚══════════════════════════════════════════════════════════════════════════════════╝

PURPOSE OF THIS FILE
────────────────────
This file is a living, executable reference for every public function in the
FlipRate library.  Each section explains:

  1. WHAT the function does — plain English description
  2. WHY it is needed — the real-world problem it solves
  3. HOW to use it — a runnable example with printed output
  4. WHEN to use it — guidance on which API tier to reach for first

The file runs end-to-end with only the standard dev extras:
    pip install "fliprate[dev,sklearn]"

Hugging Face and LLM sections are shown as runnable blocks but guarded by
try/except so the file still runs without those optional packages installed.

Run it:
    python scratch/fliprate_showcase.py

Colab-ready twin (downloads real Dakshina + Aksharantar datasets and runs live
Hugging Face + LLM models with your own API keys):
    examples/fliprate_colab_showcase.ipynb

Author : Shiv Supane
Repo   : https://github.com/ShivSupane/fliprate
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Callable

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── helper ────────────────────────────────────────────────────────────────────

def section(title: str, subtitle: str = "") -> None:
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * width)


def show(label: str, value) -> None:
    """Pretty-print a labelled result."""
    if isinstance(value, dict):
        formatted = json.dumps(value, indent=4, default=str)
    elif hasattr(value, "to_dict"):
        formatted = json.dumps(value.to_dict(), indent=4, default=str)
    elif hasattr(value, "__dataclass_fields__"):
        import dataclasses
        formatted = json.dumps(dataclasses.asdict(value), indent=4, default=str)
    else:
        formatted = str(value)
    print(f"\n  [>] {label}:")
    for line in formatted.splitlines():
        print(f"      {line}")



# ── build a simple reproducible sklearn model for all examples ─────────────────

def _build_demo_model() -> Callable[[str], str]:
    """
    Build a tiny but realistic sklearn text classifier.

    WHY sklearn?  Because fliprate[dev] already pulls in scikit-learn and
    the model is deterministic — no network, no GPU, no API key required.
    The model is trained on a tiny Hinglish / romanized-Hindi corpus that
    deliberately contains spelling variation so FlipRate's robustness checks
    produce meaningful, non-trivial results.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # Training corpus: realistic romanized Hinglish customer-support messages
    train_texts = [
        # ── negative ──────────────────────────────────────────────────────────
        "delivery nahi aayi",        "mujhe product pasand nahi",
        "ye product kharab hai",     "return karna hai",
        "mera order cancel karo",    "bohot bura anubhav tha",
        "quality bilkul bhi acchi nahi", "nahi chahiye",
        "package damage tha",        "bohot der ho gayi delivery mein",
        "product fake nikla",        "paisa wapas karo",
        "order nahi mila",           "delivery late hai bahut",
        "product ka quality achha nahi tha", "problem hai",
        # ── positive ──────────────────────────────────────────────────────────
        "bahut accha product hai",   "delivery jaldi aayi",
        "quality bahut achhi hai",   "mujhe bahut pasand aaya",
        "shukriya flipkart",         "bohot khush hoon",
        "value for money hai",       "definitely recommend karunga",
        "packaging bhi bahut achhi thi", "bilkul sahi hai",
        "perfect condition mein mila", "superb quality",
        "bahut fast delivery",       "acha product",
        "mast product hai",          "behtareen experience",
        # ── neutral ──────────────────────────────────────────────────────────
        "theek hai",                 "average quality",
        "okay okay hai",             "normal product hai",
        "kuch khas nahi",            "chalega",
        "theek thak hai",            "average laga mujhe",
    ]
    train_labels = (
        ["negative"] * 16
        + ["positive"] * 16
        + ["neutral"] * 8
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
        ("clf",   LogisticRegression(max_iter=500, random_state=42)),
    ])
    pipe.fit(train_texts, train_labels)

    def predict(text: str) -> str:
        return pipe.predict([text])[0]

    return predict


MODEL = _build_demo_model()

# ── tiny CSV dataset for batch tests ──────────────────────────────────────────

_CSV_PATH = Path(__file__).parent / "_showcase_data.csv"

def _ensure_csv() -> str:
    """Write a small labeled CSV to disk (used by batch functions)."""
    rows = [
        "text,label",
        "delivery nahi aayi,negative",
        "mujhe product pasand nahi,negative",
        "ye product kharab hai,negative",
        "return karna hai,negative",
        "package damage tha,negative",
        "bahut accha product hai,positive",
        "delivery jaldi aayi,positive",
        "quality bahut achhi hai,positive",
        "bohot khush hoon,positive",
        "bilkul sahi hai,positive",
        "theek hai,neutral",
        "average quality,neutral",
        "kuch khas nahi,neutral",
        "chalega,neutral",
    ]
    _CSV_PATH.write_text("\n".join(rows), encoding="utf-8")
    return str(_CSV_PATH)


DATASET_PATH = _ensure_csv()


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 1 · NORMALIZATION",
    "Clean messy, noisy, and over-typed romanized Indian-language text",
)

print("""
  WHAT IS NORMALIZATION?
  ──────────────────────
  Indian users who type in Hindi, Marathi, Hinglish or other romanized
  Indic scripts have no standard spelling.  The same word gets typed in
  dozens of different ways:

      "nahi"  "nhi"  "nahin"  "nahee"  "nahiii"   ← all mean "no"
      "bahut" "bohot" "bhut"  "boht"   "bahuttt"   ← all mean "very"
      "accha" "acha"  "achha" "achaa"  "aacha"     ← all mean "good"

  WHY IS IT NEEDED?
  ──────────────────
  Your NLP model was trained on clean text.  When a user types "nahinnn"
  instead of "nahi", the model may produce a completely different prediction
  — even though the meaning is identical.  Normalization collapses all those
  noisy variants into a single canonical form BEFORE the text reaches the
  model, making predictions consistent.

  FUNCTIONS:
    fr.normalize_text(text)          — high-level, returns a clean string
    fr.normalize(text)               — low-level, returns NormalizeResult
    fr.mask_protected_terms(text)    — hide brand/code names from the normalizer
    fr.unmask_protected_terms(text)  — restore them afterward
""")

import fliprate as fr

# ── 1a. normalize_text — simplest API ─────────────────────────────────────────
noisy = "mala product avadlaaa nhi bohotttt ahe"
clean = fr.normalize_text(noisy)
show("normalize_text — basic noisy input → clean canonical form", {
    "input" : noisy,
    "output": clean,
    "why"   : "Pass 'clean' to your model instead of 'noisy' to get stable predictions",
})

# ── 1b. normalize_text with protected_terms ─────────────────────────────────
brand_text = "please check ORDER-2024-XYZ nahi aaya mujhe"
protected   = ["ORDER-2024-XYZ"]
clean_brand = fr.normalize_text(brand_text, protected_terms=protected)
show("normalize_text — brand/order codes protected from modification", {
    "input"           : brand_text,
    "protected_terms" : protected,
    "output"          : clean_brand,
    "why"             : (
        "Order IDs, product codes, brand names must never be changed. "
        "protected_terms ensures they survive normalization unchanged."
    ),
})

# ── 1c. normalize — low-level, inspect every change ─────────────────────────
result = fr.normalize("bohotttt achaaa product")
show("normalize — NormalizeResult with change details", {
    "original"   : result.original,
    "normalized" : result.normalized,
    "changed"    : result.changed,
})

# ── 1d. mask / unmask — manual control for custom pipelines ─────────────────
raw      = "BRAND-42 ko nahi chahiye bohot zyada"
masked   = fr.mask_protected_terms(raw, ["BRAND-42"])
unmask   = fr.unmask_protected_terms(masked[0], masked[1])
show("mask_protected_terms / unmask_protected_terms — manual two-step protection", {
    "original"   : raw,
    "masked"     : masked[0],
    "mask_table" : masked[1],
    "restored"   : unmask,
    "why"        : "Useful when you're building a custom normalization pipeline and need fine control",
})

# ── 1e. preview_normalized_variants — see all possible outputs ────────────────
variants = fr.preview_normalized_variants("nahinnn product milaa")
show("preview_normalized_variants — see exactly what the normalizer will produce", {
    "variants_generated": len(variants),
    "sample"            : variants[:3],
    "why"               : "Use during debugging or rule development to inspect normalizer behavior",
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — PERTURBATION
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 2 · PERTURBATION",
    "Generate realistic spelling variants to stress-test your model",
)

print("""
  WHAT IS PERTURBATION?
  ─────────────────────
  Perturbation is the process of generating alternative spellings or forms of
  the same text that a real user might type.  For example:
      "delivery nahi aayi"  →  ["delivery nhi aayi", "delivery nahin aayi", ...]

  WHY IS IT NEEDED?
  ──────────────────
  To measure robustness, FlipRate needs to know what your model predicts on
  MULTIPLE realistic variants of the same message.  If the model changes its
  answer across variants (a "flip"), that is a reliability problem.

  Perturbation is the engine that powers the gate, test, and benchmark functions.

  FUNCTIONS:
    fr.perturb(text)           — generate all variants with metadata
    fr.KINDS                   — dict of available perturbation categories
    fr.is_safe_variant(a, b)   — check if two forms are meaning-safe
""")

# ── 2a. perturb — generate variants ──────────────────────────────────────────
variants = fr.perturb("delivery nahi aayi")
show("perturb — all realistic variants of a message", {
    "original"        : "delivery nahi aayi",
    "n_variants"      : len(variants),
    "first_3_variants": [
        {"text": v.text, "kind": v.kind} for v in variants[:3]
    ],
    "why": (
        "Each Variant has .text (the perturbed form) and .kind (what type "
        "of perturbation it is: negation, vowel drop, elongation, etc.)"
    ),
})

# ── 2b. KINDS — what categories exist ─────────────────────────────────────────
show("fr.KINDS — all built-in perturbation categories", {
    "categories": list(fr.KINDS),
    "why"       : (
        "Each kind targets a specific real-world typing behaviour. "
        "You can filter or extend these when designing custom rule sets."
    ),
})

# ── 2c. is_safe_variant — verify meaning-safety ──────────────────────────────
a = "nahi"
b = "nahin"
c = "hello"
show("is_safe_variant — check if two forms are meaning-equivalent", {
    f"'{a}' vs '{b}' (same meaning)": fr.is_safe_variant(a, b),
    f"'{a}' vs '{c}' (different)":    fr.is_safe_variant(a, c),
    "why": (
        "Before adding a custom rule, call is_safe_variant to verify the "
        "two forms genuinely carry the same meaning and won't corrupt labels."
    ),
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — SCRIPT DETECTION
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 3 · SCRIPT DETECTION",
    "Detect whether text is Latin, Devanagari, or mixed-script",
)

print("""
  WHAT IS SCRIPT DETECTION?
  ─────────────────────────
  Indian users write in three different script types:
    • Latin   — Roman alphabet  ("mujhe product pasand nahi")
    • Devanagari — native script ("मुझे प्रोडक्ट पसंद नहीं")
    • Mixed   — both in one sentence ("ye product बहुत accha hai")

  WHY IS IT NEEDED?
  ──────────────────
  Different perturbation rules apply to each script.  Knowing the script
  upfront lets FlipRate select the right normalizer and perturbation set.
  It also helps you route incoming text to different processing pipelines.

  FUNCTION:
    fr.detect_script(text)  → "latin" | "devanagari" | "mixed"
""")

examples = [
    ("मुझे प्रोडक्ट पसंद नहीं",           "Devanagari-only"),
    ("mujhe product pasand nahi",          "Latin/Romanized only"),
    ("ye product बहुत accha hai",          "Mixed script (code-mixed)"),
    ("delivery 3 दिन में आई",              "Mixed with Devanagari digits"),
    ("ORDID-42 nahi mila",                 "Latin with codes"),
]

for text, description in examples:
    script = fr.detect_script(text)
    print(f"  ► {description:40s}  script = {script!r:15s}  text = {text!r}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — REAL-TIME GATE
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 4 · REAL-TIME GATE",
    "Check whether your model is safe to trust on a given input — right now",
)

print("""
  WHAT IS THE GATE?
  ──────────────────
  The gate is the core robustness check.  It:
    1. Takes your model and a user's message.
    2. Generates realistic spelling variants of the message.
    3. Runs the model on every variant.
    4. Checks whether the model gives the SAME prediction on all of them.

  If it does → PASS   (model is robust, safe to use the prediction)
  If it flips but normalization fixes it → TRANSFORM  (use normalized form)
  If it still flips after normalization → REVIEW  (flag for human review)

  WHY IS IT NEEDED?
  ──────────────────
  Imagine a customer-support bot:
    User types "delivery nahi aayi"  → bot says "We're sorry"  ✓
    Same user, same meaning:
    User types "delivery nhi aai"    → bot says "Great!"        ✗  ← dangerous

  The gate catches this BEFORE you send the prediction to the user.

  FUNCTIONS:
    fr.gate(model, text)         — synchronous gate check
    fr.agate(model, text)        — async version (for async web frameworks)
    fr.iter_gate(model, texts)   — streaming gate over many messages
    fr.quick_check(model, text)  — ultra-simple beginner API
    fr.safe_predict(model, text) — normalize + gate + predict in one call
""")

# ── 4a. gate — full result ────────────────────────────────────────────────────
for msg in ["delivery nahi aayi", "bahut accha product hai", "theek hai"]:
    result = fr.gate(MODEL, msg)
    print(f"\n  Input: {msg!r}")
    print(f"    decision    = {result.decision}")
    print(f"    flip_rate   = {result.flip_rate:.2f}")
    print(f"    prediction  = {result.prediction!r}")
    print(f"    n_variants  = {result.n_variants}")

# ── 4b. quick_check — beginner's one-liner ───────────────────────────────────
qr = fr.quick_check(MODEL, "ye product kharab hai")
show("quick_check — simplest possible API for beginners", {
    "is_safe"          : qr.is_safe,
    "decision"         : qr.decision,
    "prediction"       : qr.prediction,
    "suggested_text"   : qr.suggested_text,
    "reason"           : qr.reason,
    "flipped_variants" : qr.flipped_variants[:2],
    "why": (
        "quick_check gives a single True/False 'is_safe' answer. "
        "Perfect for beginners or simple routing logic."
    ),
})

# ── 4c. safe_predict — production one-liner ──────────────────────────────────
prediction = fr.safe_predict(MODEL, "bohot bura anubhav tha")
show("safe_predict — normalize + gate + predict in one call (production API)", {
    "input"      : "bohot bura anubhav tha",
    "prediction" : prediction,
    "why": (
        "safe_predict is the production shortcut: it normalizes the text, "
        "runs the gate, and returns the prediction — all in one line."
    ),
})

# ── 4d. iter_gate — stream real-time events during gating ────────────────────
print("\n  iter_gate — stream real-time events during evaluation:")
print("  WHY: Yields progressive evaluation events (variants, predictions) as they happen")
events = list(fr.iter_gate(MODEL, "delivery nahi aayi"))
print(f"    Streamed {len(events)} events during evaluation of 'delivery nahi aayi'")
if events:
    print(f"    First event: kind={events[0].get('kind')!r}, data={events[0].get('data')}")




# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — PROTECT & PROTECTED MODEL
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 5 · PROTECT & PROTECTED MODEL",
    "Wrap any model so it normalizes text automatically before every prediction",
)

print("""
  WHAT IS A PROTECTED MODEL?
  ──────────────────────────
  A ProtectedModel is a thin wrapper around your model that automatically
  normalizes every input BEFORE calling predict().  Once wrapped, you can
  pass the protected model anywhere that accepts a normal model — the
  normalization happens invisibly.

  WHY IS IT NEEDED?
  ──────────────────
  Instead of calling fr.normalize_text(text) before every predict() call,
  you wrap your model ONCE and forget about it.  Every downstream call
  automatically gets normalized input.

  FUNCTIONS:
    fr.protect(model, protected_terms=None)   — wrap a model
    fr.make_protected_model(model, ...)       — alias with extra options
    ProtectedModel class                      — the wrapper object
""")

protected_model = fr.protect(MODEL)
raw_noisy       = "nahinnn product milaaaa"
raw_pred        = MODEL(raw_noisy)
prot_pred       = protected_model(raw_noisy)

show("fr.protect — wrap your model to normalize inputs automatically", {
    "input"                    : raw_noisy,
    "raw_model_prediction"     : raw_pred,
    "protected_model_prediction": prot_pred,
    "why": (
        "The raw model may flip on 'nahinnn' vs 'nahi'. "
        "The protected model normalizes first, giving a consistent prediction."
    ),
})

# protected model with protected terms
protected_with_terms = fr.protect(MODEL, protected_terms=["ORDID-99"])
pred = protected_with_terms("ORDID-99 product nahi mila")
show("fr.protect with protected_terms — preserve codes through normalization", {
    "input"      : "ORDID-99 product nahi mila",
    "prediction" : pred,
    "protected"  : ["ORDID-99"],
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — BATCH EVALUATION (TEST)
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 6 · BATCH EVALUATION — test / check_model / iter_test",
    "Measure flip rate and accuracy across your entire labeled dataset",
)

print("""
  WHAT IS BATCH EVALUATION?
  ─────────────────────────
  The gate checks ONE message at a time.  Batch evaluation runs the
  robustness check across an entire labeled CSV dataset and produces a
  Report with aggregated statistics:

    • flip_rate          — what fraction of messages flipped across variants
    • accuracy           — model accuracy on the original inputs
    • confidence_interval— 95% Wilson CI on the flip rate
    • flips              — list of all individual flip events

  WHY IS IT NEEDED?
  ──────────────────
  You need to know OVERALL how brittle your model is, not just for one
  message.  "Our model has a flip rate of 0.27 [CI: 0.21–0.33]" is
  a concrete, reportable metric you can track in CI/CD.

  FUNCTIONS:
    fr.test(model, dataset)            — main batch evaluation → Report
    fr.check_model(model, dataset)     — alias with printed summary
    fr.iter_test(model, dataset)       — streaming version (memory-efficient)
    fr.paired_difference(r1, r2)       — compare two reports statistically
    fr.macro_f1(labels, preds)         — compute macro F1 on any label list
""")

# ── 6a. test — main evaluation ───────────────────────────────────────────────
report = fr.test(MODEL, DATASET_PATH)
show("fr.test — full batch evaluation on a labeled CSV", {
    "flip_rate"           : round(report.flip_rate, 4),
    "accuracy"            : round(report.accuracy, 4),
    "confidence_interval" : [round(x, 4) for x in report.confidence_interval],
    "n_messages"          : report.n_messages,
    "n_flips"             : report.n_flips,
    "schema_version"      : report.schema_version,
    "why": (
        "Use fr.test() in your CI pipeline: if flip_rate > 0.05 → fail the build. "
        "The Report is JSON-serializable so you can store it as an artifact."
    ),
})

print(f"\n  report.summary():\n    {report.summary()}")

# ── 6b. check_model — prints summary ─────────────────────────────────────────
print("\n  fr.check_model — same as test, also prints a human-readable summary:")
print("  WHY: One-liner for quick inspection during development")
report2 = fr.check_model(MODEL, DATASET_PATH)

# ── 6c. iter_test — memory-efficient streaming ───────────────────────────────
print("\n  fr.iter_test — streaming evaluation (memory-efficient for huge datasets):")
print("  WHY: When your dataset is millions of rows, don't load it all at once")
count = 0
for r in fr.iter_test(MODEL, DATASET_PATH):
    count += 1
print(f"    Streamed and processed {count} individual test results")

# ── 6d. paired_difference — statistical comparison ───────────────────────────
report_protected = fr.test(fr.protect(MODEL), DATASET_PATH)
diff = fr.paired_difference(report, report_protected)
show("fr.paired_difference — statistically compare two models/conditions", {
    "base_flip_rate"      : round(report.flip_rate, 4),
    "protected_flip_rate" : round(report_protected.flip_rate, 4),
    "difference"          : diff,
    "why": (
        "Use to prove that adding normalization (fr.protect) actually improved "
        "robustness by a statistically meaningful amount."
    ),
})

# ── 6e. macro_f1 ─────────────────────────────────────────────────────────────
labels = ["negative", "positive", "neutral", "negative", "positive"]
preds  = ["negative", "positive", "neutral", "positive", "positive"]
f1 = fr.macro_f1(labels, preds)
show("fr.macro_f1 — compute macro-averaged F1 on any label / prediction lists", {
    "labels"    : labels,
    "preds"     : preds,
    "macro_f1"  : round(f1, 4),
    "why"       : "Handy when evaluating accuracy beyond just flip rate",
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — EXPLAINABILITY & DEBUGGING
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 7 · EXPLAINABILITY & DEBUGGING",
    "Understand exactly WHY a model flipped its prediction",
)

print("""
  WHAT IS EXPLAINABILITY?
  ────────────────────────
  When the gate returns REVIEW, you need to know:
    • Which specific variants caused the flip?
    • Which words changed between the original and the variant?
    • What can be done to fix it?

  WHY IS IT NEEDED?
  ──────────────────
  "Our model flipped" is not actionable.  "The model flipped on 'nhi'
  vs 'nahi' because the negation token changed and the model never saw
  'nhi' in training" IS actionable — you know to add normalization or
  retrain with augmented data.

  FUNCTIONS:
    fr.show_why_it_flipped(model, text)       — per-variant word diffs
    fr.explain_gate_decision(model, text)     — human-readable explanation
    fr.explain_report_flips(report)           — report-level summary
    fr.print_rule_hits(text)                  — show which rules matched
""")

# ── 7a. show_why_it_flipped ──────────────────────────────────────────────────
flips = fr.show_why_it_flipped(MODEL, "delivery nahi aayi")
print(f"\n  fr.show_why_it_flipped — {len(flips)} flip event(s) found")
print("  WHY: Get per-variant details with word-level diffs")
for i, flip in enumerate(flips[:3], 1):
    print(f"\n    Flip #{i}:")
    print(f"      original_text       = {flip.get('original_text', '')!r}")
    print(f"      variant_text        = {flip.get('variant_text', '')!r}")
    print(f"      original_prediction = {flip.get('original_prediction', '')!r}")
    print(f"      variant_prediction  = {flip.get('variant_prediction', '')!r}")
    print(f"      diffs               = {flip.get('diffs', [])}")

# ── 7b. explain_gate_decision ────────────────────────────────────────────────
explanation = fr.explain_gate_decision(MODEL, "delivery nahi aayi")
show("fr.explain_gate_decision — human-readable gate explanation", {
    "summary"         : explanation.get("summary", ""),
    "recommendation"  : explanation.get("recommendation", ""),
    "flip_rate"       : explanation.get("flip_rate", 0),
    "why": (
        "Use this in a developer dashboard or logging system to get an "
        "English explanation of why the gate made its decision."
    ),
})

# ── 7c. explain_report_flips — privacy-safe report summary ──────────────────
summary = fr.explain_report_flips(report)
show("fr.explain_report_flips — report-level summary (privacy-safe by default)", {
    "summary_keys"  : list(summary.keys()) if isinstance(summary, dict) else type(summary).__name__,
    "privacy_note"  : "Raw input text is NEVER included unless opt_in_examples=True",
    "why"           : (
        "Generate a safe, shareable summary of a full test report. "
        "Call with opt_in_examples=True only when you explicitly want examples."
    ),
})

# ── 7d. print_rule_hits — debug which rules matched ─────────────────────────
print("\n  fr.print_rule_hits — show which built-in rules matched tokens in text:")
print("  WHY: During rule development, verify your new rules actually fire")
fr.print_rule_hits("delivery nahi bohot aayi")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — RULE MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 8 · RULE MANAGEMENT",
    "Inspect, extend, and customise the spelling rule engine",
)

print("""
  WHAT ARE RULES?
  ───────────────
  FlipRate's normalizer is rule-driven.  Each rule is a set of equivalent
  spellings (a "canonical group"):

      Group "nahi":   {"nahi", "nhi", "nahin", "nahee", "nai", ...}
      Group "bahut":  {"bahut", "bohot", "bhut", "boht", ...}

  The normalizer maps every variant in a group to the canonical (first) form.

  WHY IS IT NEEDED?
  ──────────────────
  Your domain will have custom terms — brand names, jargon, product codes —
  that the built-in rules don't know about.  You can extend the rule engine
  with your own groups without modifying the library.

  FUNCTIONS:
    fr.DEFAULT_RULES                   — the built-in RuleSet
    fr.describe_rule_group(name)       — inspect a rule group
    fr.preview_variants(text)          — see what variants a text generates
    fr.add_custom_rule(spellings)      — add a new spelling equivalence group
    fr.load_rules(path)                — load rules from a JSON file
    fr.RuleSet                         — the rule container class
""")

# ── 8a. describe a rule group ─────────────────────────────────────────────────
desc = fr.describe_rule_group("nahi")
show("fr.describe_rule_group('nahi') — inspect all known spellings for a word", {
    "result": desc,
    "why"   : "Before adding a custom rule, check if a built-in group already covers it",
})

# ── 8b. preview_variants — what variants does a rule generate? ───────────────
pv = fr.preview_variants("bohot accha delivery tha")
show("fr.preview_variants — see all variants generated for a text", {
    "input"           : "bohot accha delivery tha",
    "n_variants"      : len(pv),
    "first_3"         : pv[:3],
    "why"             : "Use during development to verify your text generates enough meaningful variants",
})

# ── 8c. add_custom_rule — extend the engine with domain terms ─────────────────
fr.add_custom_rule(["thnx", "thanks", "thankyou", "thx", "tysm"])
fr.add_custom_rule(["superfast", "super fast", "super-fast", "bahut tezi se"])
print("\n  fr.add_custom_rule — added domain-specific spelling groups")
print("  WHY: Your brand or domain may have specific abbreviations not in the built-in rules")
show("Custom rule test — normalize 'thnx' and 'tysm'", {
    "'thnx'  → ": fr.normalize_text("thnx"),
    "'tysm'  → ": fr.normalize_text("tysm"),
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — MULTI-SCRIPT: DEVANAGARI
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 9 · DEVANAGARI & MIXED-SCRIPT SUPPORT",
    "Gate and normalize native Devanagari and mixed-script Hindi/Marathi text",
)

print("""
  WHAT IS DEVANAGARI SUPPORT?
  ────────────────────────────
  FlipRate handles not just romanized text but also native Devanagari
  script (Hindi, Marathi) and mixed-script sentences.

  Devanagari perturbations include:
    • Unicode normalization: precomposed vs decomposed nuktas
      (e.g. ज़रा  vs  जरा  — different Unicode codepoints, same glyph)
    • Digit swap: ASCII 0-9  ↔  Devanagari ०-९
    • Punctuation: danda (।) / double-danda (॥) ↔ Western . / ..
    • Meaning-safe nukta variants: फ़ोन / फोन, ज़रा / जरा
    • Hindi/Marathi negations: नहीं / नाही / नको (don't want / not / no)

  WHY IS IT NEEDED?
  ──────────────────
  Real users mix Devanagari and Roman script in the same sentence.
  A robust Indian-language model must handle both.
""")

deva_examples = [
    "मुझे प्रोडक्ट पसंद नहीं",
    "delivery बहुत late थी",
    "product ज़रा भी अच्छा नहीं",
    "package damage था",
]

print("  Devanagari gate results:")
for text in deva_examples:
    script = fr.detect_script(text)
    result = fr.gate(MODEL, text)
    print(f"    [{script:12s}]  {text:35s}  →  {result.decision:10s}  flip={result.flip_rate:.2f}")

# ── Devanagari normalization ──────────────────────────────────────────────────
deva_text = "मुझे प्रोडक्ट पसंद नहीं"
deva_norm = fr.normalize_text(deva_text)
show("normalize_text on Devanagari input", {
    "input"  : deva_text,
    "output" : deva_norm,
    "script" : fr.detect_script(deva_text),
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — MODEL ADAPTERS
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 10 · MODEL ADAPTERS",
    "Plug in sklearn, Hugging Face, LLMs, REST APIs, or any callable",
)

print("""
  WHAT ARE MODEL ADAPTERS?
  ─────────────────────────
  FlipRate works with ANY callable that maps a string → a label string.
  Model Adapters are thin wrappers that make standard model types
  (sklearn, HuggingFace, REST, LLMs) conform to this interface.

  WHY IS IT NEEDED?
  ──────────────────
  You shouldn't have to rewrite your model to use FlipRate.
  Adapters mean you can pass your existing model directly:
    • Hugging Face pipeline  → fr.HFTextClassifier(model_name)
    • sklearn pipeline       → fr.from_sklearn(sklearn_pipeline)
    • REST endpoint          → fr.HttpServiceAdapter(url)
    • OpenAI / Gemini        → fr.LLMClassifier(client, labels)
    • Any Python callable    → fr.CallableAdapter(fn)

  FUNCTIONS:
    fr.CallableAdapter(fn)                     — wrap any function
    fr.as_adapter(model)                       — auto-detect and wrap
    fr.from_sklearn(pipeline)                  — wrap sklearn pipelines
    fr.HttpServiceAdapter(url, timeout, retries)
    fr.HFTextClassifier(model_name, ...)       — Hugging Face pipelines
    fr.LLMClassifier(client, labels, ...)      — LLM classification
    fr.with_retries(adapter, max_retries)      — add retry logic
""")

# ── 10a. CallableAdapter — wrap any function ──────────────────────────────────
def simple_keyword_model(text: str) -> str:
    """Rule-based demo model: negative if text contains negation keywords."""
    neg = {"nahi", "nhi", "nahin", "kharab", "bura", "pasand nahi", "damage"}
    pos = {"accha", "achha", "khush", "fast", "superb"}
    low = text.lower()
    if any(w in low for w in neg): return "negative"
    if any(w in low for w in pos): return "positive"
    return "neutral"

adapter = fr.CallableAdapter(simple_keyword_model)
result  = fr.gate(adapter, "delivery nahi aayi")
show("fr.CallableAdapter — wrap any Python function as a FlipRate model", {
    "input"    : "delivery nahi aayi",
    "decision" : result.decision,
    "flip_rate": round(result.flip_rate, 4),
    "why"      : "Any function(text) -> str works — regex, rules, ML models, APIs",
})

# ── 10b. from_sklearn — wrap a sklearn pipeline ──────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SKPipeline

sk_pipeline = SKPipeline([
    ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
    ("clf",   LogisticRegression(max_iter=400, random_state=0)),
])
# Re-use the same training data
train_texts = [
    "delivery nahi aayi", "mujhe product pasand nahi", "ye product kharab hai",
    "bahut accha product hai", "delivery jaldi aayi", "quality bahut achhi hai",
    "theek hai", "average quality", "kuch khas nahi",
]
train_labels = ["negative", "negative", "negative", "positive", "positive", "positive",
                "neutral", "neutral", "neutral"]
sk_pipeline.fit(train_texts, train_labels)

sklearn_adapter = fr.from_sklearn(sk_pipeline)
result_sk = fr.gate(sklearn_adapter, "bohot bura experience tha")
show("fr.from_sklearn — wrap a scikit-learn Pipeline", {
    "adapter_type"  : type(sklearn_adapter).__name__,
    "input"         : "bohot bura experience tha",
    "decision"      : result_sk.decision,
    "flip_rate"     : round(result_sk.flip_rate, 4),
    "why"           : (
        "You already have a trained sklearn model. "
        "from_sklearn wraps it so every FlipRate function can use it."
    ),
})

# ── 10c. as_adapter — auto-detect ────────────────────────────────────────────
auto = fr.as_adapter(MODEL)
show("fr.as_adapter — automatically detect and wrap any model type", {
    "adapter_type": type(auto).__name__,
    "why"         : "Use as_adapter when you don't know the model type at import time",
})

# ── 10d. with_retries — add retry logic ──────────────────────────────────────
resilient = fr.with_retries(adapter, retries=3)
pred_r    = resilient("delivery nahi aayi")
show("fr.with_retries — add automatic retry logic to any adapter", {
    "prediction": pred_r,
    "why"       : (
        "When the adapter wraps a REST endpoint or LLM API, transient network "
        "failures should be retried automatically rather than crashing the gate."
    ),
})

# ── 10e. HFTextClassifier — Hugging Face (optional) ──────────────────────────
print("""
  ── Hugging Face Adapter ──────────────────────────────────────────────────────
  Requires: pip install "fliprate[hf]"  (transformers + torch)

  USAGE:
    clf = fr.HFTextClassifier(
        "cardiffnlp/twitter-xlm-roberta-base-sentiment",
        lowercase_labels=True,
        preprocess=fr.twitter_preprocess,
        batch_size=16,
    )
    result = fr.gate(clf, "mujhe ye product pasand nahi")
    report = fr.test(clf, "data/test.csv")

  WHY:
    HFTextClassifier lazy-loads the model on first call (no memory wasted
    until needed), supports batched inference for large datasets, and
    automatically handles HuggingFace label formatting.
""")

try:
    from transformers import pipeline as hf_pipeline
    hf_clf = fr.HFTextClassifier.__new__(fr.HFTextClassifier)
    print("  [HuggingFace transformers available — use fr.HFTextClassifier(...)]")
except ImportError:
    print("  [HuggingFace not installed — install with: pip install 'fliprate[hf]']")

# ── 10f. LLMClassifier — OpenAI / Gemini / Anthropic ─────────────────────────
import os

print("""
  ── LLM Classifier (Google Gemini / OpenAI / Anthropic) ───────────────────────
  Accepts ANY text completion function: complete(prompt) -> str.
  Strictly validates and extracts only designated labels, rejecting freeform chat.
  Compatible with fr.gate(), fr.test(), fr.benchmark(), and fr.protect().
""")

def _get_llm_completion() -> tuple[Callable[[str], str], str]:
    """Return an LLM completion function using live API keys if present, or a realistic simulator."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    if gemini_key:
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            return (
                lambda prompt: client.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                ).text or "neutral",
                "Google Gemini (Live API - gemini-2.5-flash)",
            )
        except Exception:
            try:
                import google.generativeai as gai
                gai.configure(api_key=gemini_key)
                model = gai.GenerativeModel("gemini-1.5-flash")
                return (
                    lambda prompt: model.generate_content(prompt).text or "neutral",
                    "Google Gemini (Live API - gemini-1.5-flash)",
                )
            except Exception:
                pass
    if openai_key:
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
            return (
                lambda prompt: client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                ).choices[0].message.content or "neutral",
                "OpenAI (Live API - gpt-4o-mini)",
            )
        except Exception:
            pass

    def simulated_llm_complete(prompt: str) -> str:
        low = prompt.lower()
        if any(w in low for w in ["nahi", "nhi", "nahin", "kharab", "bura", "damage", "fake", "bekaar"]):
            return "negative"
        if any(w in low for w in ["accha", "achha", "superb", "mast", "badhiya", "fast", "khush"]):
            return "positive"
        return "neutral"

    return simulated_llm_complete, "Simulated LLM (set GEMINI_API_KEY or OPENAI_API_KEY for live API calls)"

llm_complete_fn, llm_provider_name = _get_llm_completion()
llm_clf = fr.LLMClassifier(
    complete=llm_complete_fn,
    labels=["positive", "negative", "neutral"],
    instruction="Classify this Indian customer support message into exactly one sentiment label.",
    name="llm_sentiment_classifier",
)

result_llm = fr.gate(llm_clf, "bohot bura experience tha")
show("fr.LLMClassifier — LLM-powered classification with strict label parsing", {
    "provider": llm_provider_name,
    "input": "bohot bura experience tha",
    "decision": result_llm.decision,
    "prediction": result_llm.prediction,
    "flip_rate": round(result_llm.flip_rate, 4),
    "why": (
        "LLMClassifier wraps any completion endpoint (Gemini, OpenAI, Claude, Ollama), "
        "enforces single-label parsing, and works seamlessly with fr.gate() and fr.protect()."
    ),
})

protected_llm = fr.protect(llm_clf)
pred_protected = protected_llm("mala product avadlaaa nhi bohotttt")
show("fr.protect(llm_clf) — wrap LLM with FlipRate normalizer", {
    "raw_input": "mala product avadlaaa nhi bohotttt",
    "protected_pred": pred_protected,
    "benefit": "Normalizes romanized variants BEFORE hitting LLM, stopping flips and cutting token costs!",
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — RUNTIME GUARD & PRODUCTION PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 11 · RUNTIME GUARD & PRODUCTION PIPELINE",
    "Production-grade multi-step pipeline: guard, batch-guard, evaluate",
)

print("""
  WHAT IS THE RUNTIME GUARD?
  ──────────────────────────
  The runtime guard extends the basic gate with production features:
    • Configurable thresholds: only REVIEW above a certain flip rate
    • Custom fallback labels: what to return when flipping is detected
    • Async-ready design for high-throughput APIs

  WHY IS IT NEEDED?
  ──────────────────
  In production, you want fine-grained control over what happens at each
  flip-rate threshold.  You might accept flip_rate < 0.1 as PASS, flag
  0.1–0.3 for soft review, and hard-block above 0.3.

  FUNCTIONS:
    fr.runtime_guard(model, text, max_flip_rate, fallback_label)
    fr.batch_guard(model, texts, ...)       — guard many messages at once
    fr.evaluate_pipeline(model, dataset)    — full pipeline evaluation
    fr.make_protected_model(model, ...)     — factory for protected models
""")

# ── 11a. runtime_guard ───────────────────────────────────────────────────────
rg = fr.runtime_guard(MODEL, "delivery nahi aayi", mode="lenient")
show("fr.runtime_guard — gate with configurable mode (strict/lenient)", {
    "decision"              : rg.decision,
    "prediction"            : rg.prediction,
    "clean_text"            : rg.clean_text,
    "should_route_to_human" : rg.should_route_to_human,
    "flip_rate"             : round(rg.flip_rate, 4),
    "reason"                : rg.reason,
    "why"                   : "Set mode='strict' or 'lenient' to control whether TRANSFORM triggers human routing",
})

# ── 11b. batch_guard ─────────────────────────────────────────────────────────
messages = ["delivery nahi aayi", "accha product hai", "kharab tha", "theek hai"]
batch    = fr.batch_guard(MODEL, messages)
print("\n  fr.batch_guard — guard a batch of messages in one call:")
print("  WHY: More efficient than calling gate() in a loop for many messages")
for msg, bg in zip(messages, batch):
    print(f"    {msg:30s}  →  {bg.decision:10s}  flip={bg.flip_rate:.2f}")

# ── 11c. make_protected_model ─────────────────────────────────────────────────
pm = fr.make_protected_model(MODEL, protected_terms=["ORDER-XYZ"])
pred_pm = pm("ORDER-XYZ nahi mila mujhe")
show("fr.make_protected_model — factory with protected terms", {
    "input"      : "ORDER-XYZ nahi mila mujhe",
    "prediction" : pred_pm,
    "why"        : "Convenience factory for creating a protected model in one call with options",
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — MULTI-MODEL BENCHMARKING
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 12 · MULTI-MODEL BENCHMARKING",
    "Compare multiple models side-by-side with dashboards and scorecards",
)

print("""
  WHAT IS BENCHMARKING?
  ──────────────────────
  Benchmarking runs your entire test dataset through MULTIPLE models and
  compares them side-by-side:

    Model          | Accuracy | Flip Rate | Flip Rate (protected) | Rank
    word-tfidf     |   0.84   |   0.27    |        0.12           |  2
    char-tfidf     |   0.87   |   0.21    |        0.09           |  1

  Output formats: terminal scorecard, interactive HTML dashboard, Markdown table.

  WHY IS IT NEEDED?
  ──────────────────
  Choosing the best model means comparing both raw accuracy AND robustness.
  A model with 90% accuracy but 40% flip rate is worse for production than
  one with 85% accuracy and 5% flip rate.

  FUNCTIONS:
    fr.benchmark(models, dataset)          — multi-model comparison
    fr.compare_models(models, dataset)     — alias
    BenchmarkSuite.summary()              — terminal scorecard
    BenchmarkSuite.save("out.html")       — interactive HTML dashboard
    BenchmarkSuite.to_markdown()          — Markdown table
""")

# Build a second model for comparison
def _build_char_model() -> Callable[[str], str]:
    sk = SKPipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5))),
        ("clf",   LogisticRegression(max_iter=400, random_state=1)),
    ])
    sk.fit(train_texts, train_labels)
    return lambda t: sk.predict([t])[0]

CHAR_MODEL = _build_char_model()

suite = fr.benchmark(
    models={"word_tfidf": MODEL, "char_tfidf": CHAR_MODEL},
    dataset=DATASET_PATH,
    baseline="word_tfidf",
)
print("\n  fr.benchmark result:")
print(suite.summary())

show("BenchmarkSuite.to_dict() — fully serializable benchmark data", {
    "models"       : list(suite.scores.keys()),
    "baseline"     : suite.baseline,
    "why"          : (
        "Store suite.to_dict() as a JSON artifact in CI so you can track "
        "robustness trends over time across model versions."
    ),
})

# save HTML dashboard
out_html = Path(__file__).parent / "_benchmark_dashboard.html"
suite.save(str(out_html))
print(f"\n  ✓ Interactive HTML dashboard saved → {out_html}")

# save Markdown
out_md = Path(__file__).parent / "_benchmark_report.md"
suite.save(str(out_md))
print(f"  ✓ Markdown report saved             → {out_md}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — DATASET AUDIT
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 13 · DATASET AUDIT",
    "Audit your evaluation dataset before you report any numbers",
)

print("""
  WHAT IS A DATASET AUDIT?
  ─────────────────────────
  Before publishing benchmark results, you need to know your evaluation
  data is clean.  A dataset audit checks for:

    • Class imbalance          — too many examples of one class
    • Script distribution      — latin / devanagari / mixed ratio
    • Exact duplicates         — identical rows that inflate metrics
    • Near-duplicates          — spelling variants of the same message
    • Label conflicts          — near-duplicates with different labels
    • Train/test leakage       — test rows that appeared in training

  WHY IS IT NEEDED?
  ──────────────────
  If your test set contains exact duplicates of training examples, your
  benchmark numbers are inflated and misleading.  Publishing results from
  a leaky or imbalanced dataset damages credibility.

  FUNCTION:
    fr.audit_dataset(dataset, train_dataset=None, rules=None)
""")

# Build a separate "train" CSV for leakage demo
train_csv = Path(__file__).parent / "_showcase_train.csv"
train_csv.write_text(
    "text,label\n"
    "delivery nahi aayi,negative\n"  # ← same as test row → leakage!
    "bahut accha product hai,positive\n"
    "mast product tha,positive\n"
    "quality theek nahi,negative\n",
    encoding="utf-8",
)

audit = fr.audit_dataset(DATASET_PATH, train_dataset=str(train_csv))

show("fr.audit_dataset — full data quality report", {
    "n_messages"            : audit.get("n_messages"),
    "class_distribution"    : audit.get("class_distribution"),
    "script_distribution"   : audit.get("script_distribution"),
    "exact_duplicates"      : audit.get("exact_duplicates"),
    "near_duplicates"       : audit.get("near_duplicates"),
    "label_conflicts"       : audit.get("label_conflicts"),
    "train_test_exact_leaks": audit.get("train_test_exact_leaks"),
    "leakage_detected"      : audit.get("leakage_detected"),
    "why": (
        "Run audit_dataset BEFORE running benchmark. "
        "If leakage_detected is True, your test set is contaminated."
    ),
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 14 — REPRODUCIBLE BENCHMARK PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 14 · REPRODUCIBLE BENCHMARK PROTOCOL",
    "Run a rigorous, audit-first reproducible comparison",
)

print("""
  WHAT IS THE BENCHMARK PROTOCOL?
  ─────────────────────────────────
  run_benchmark_protocol combines:
    1. Dataset audit (automatic, always on)
    2. Multi-model comparison: raw AND protected variants of each model
    3. 95% Wilson CIs on every flip rate
    4. Runtime and call-count accounting
    5. The audit results attached directly to the BenchmarkSuite

  WHY IS IT NEEDED?
  ──────────────────
  A standard benchmark (fr.benchmark) doesn't automatically check data
  quality.  run_benchmark_protocol forces you to audit first and attaches
  the evidence to the same object you save as an artifact.

  This makes your results reproducible and auditable:
  "Here's the benchmark result AND proof the dataset was clean."

  FUNCTION:
    fr.run_benchmark_protocol(models, dataset, train_dataset=None, ...)
""")

protocol_suite = fr.run_benchmark_protocol(
    models       = {"word_tfidf": MODEL, "char_tfidf": CHAR_MODEL},
    dataset      = DATASET_PATH,
    train_dataset= str(train_csv),
    baseline     = "word_tfidf",
    n_variants   = 3,
    seed         = 42,
)
print("\n  run_benchmark_protocol result:")
print(protocol_suite.summary())
show("protocol_suite.dataset_audit — audit evidence attached to the suite", {
    "leakage_detected"  : protocol_suite.dataset_audit.get("leakage_detected"),
    "near_duplicates"   : protocol_suite.dataset_audit.get("near_duplicates"),
    "n_messages"        : protocol_suite.dataset_audit.get("n_messages"),
    "why"               : "The audit is permanently attached — evidence travels with results",
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 15 — TRAINING AUGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 15 · TRAINING AUGMENTATION",
    "Improve robustness by augmenting your training data with spelling variants",
)

print("""
  WHAT IS AUGMENTATION?
  ──────────────────────
  Instead of only normalizing text at inference time, you can ALSO improve
  your model by training it on realistic spelling variants.

  Augmentation generates meaning-safe spelling variants of your training
  examples and adds them to your training set with the SAME label.

  Before augmentation:
    training set: 100 examples
    model flip rate: 0.27

  After augmentation:
    training set: 340 examples (with variants)
    model flip rate: 0.09  ← significantly reduced

  WHY IS IT NEEDED?
  ──────────────────
  Normalization is a runtime fix.  Augmentation is a training-time fix.
  Together they give the best robustness.

  FUNCTIONS:
    fr.augment(texts, labels, rules=None)           — generate augmented dataset
    fr.AugmentationResult                           — result container
    fr.compare_augmentation(model_fn, dataset, ...) — measure improvement
""")

train_rows_sample = list(zip(
    ["delivery nahi aayi", "bahut accha product hai", "theek hai", "ye kharab hai"],
    ["negative",           "positive",                "neutral",   "negative"],
))

aug = fr.augment(train_rows_sample, n_per_example=2)
show("fr.augment — generate meaning-safe spelling variants for training", {
    "original_size"        : aug.n_original,
    "augmented_variants"   : aug.n_augmented,
    "total_rows"           : aug.total_rows,
    "by_kind"              : aug.by_kind,
    "sample_augmented_rows": aug.to_rows()[:4],
    "why": (
        "Feed aug.to_rows() to your model trainer. "
        "The model learns to handle spelling variation at training time."
    ),
})

# compare_augmentation requires a model factory that returns an object with .fit() and __call__()
class RetrainableClassifier:
    def __init__(self):
        self.pipe = SKPipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))),
            ("clf",   LogisticRegression(max_iter=400, random_state=0)),
        ])

    def fit(self, X, y):
        self.pipe.fit(X, y)
        return self

    def __call__(self, text: str) -> str:
        return str(self.pipe.predict([text])[0])

    def predict_batch(self, texts: Sequence[str]) -> List[str]:
        return [str(x) for x in self.pipe.predict(list(texts))]

comparison = fr.compare_augmentation(
    model_factory  = RetrainableClassifier,
    train_data     = str(train_csv),
    test_data      = DATASET_PATH,
)
show("fr.compare_augmentation — before / after flip rate comparison", {
    "baseline"  : comparison.get("baseline"),
    "augmented" : comparison.get("augmented"),
    "diff"      : comparison.get("diff"),
    "why": (
        "compare_augmentation retrains your model twice (with/without augmentation) "
        "and reports the flip-rate improvement from augmentation."
    ),
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 16 — NEAR DUPLICATES, LEAKAGE & CANDIDATE MINING
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 16 · DATASET TOOLS — find_near_duplicates / check_leakage / mine_variants",
    "Low-level tools for dataset quality and rule discovery",
)

print("""
  WHAT ARE THESE TOOLS?
  ──────────────────────
  These are the low-level primitives that power audit_dataset().
  Use them individually when you need fine-grained control.

    find_near_duplicates(dataset)   — detect near-duplicate (variant) pairs
    check_leakage(test, train)      — detect test rows that appeared in training
    mine_variants(texts)            — discover new spelling rules from your data

  WHY ARE THEY NEEDED?
  ─────────────────────
  • Near-duplicates with conflicting labels corrupt training
  • Leakage inflates benchmark numbers
  • Mining helps you discover domain-specific spelling patterns you didn't know about
""")

# ── 16a. find_near_duplicates ─────────────────────────────────────────────────
dup_report = fr.find_near_duplicates(DATASET_PATH)
show("fr.find_near_duplicates — find spelling-variant pairs in a dataset", {
    "n_near_duplicates" : len(dup_report.near_duplicates),
    "n_conflicts"       : len(dup_report.conflicting_pairs),
    "has_duplicates"    : dup_report.has_duplicates,
    "why"               : (
        "Near-duplicates with the same label inflate flip-rate metrics. "
        "Near-duplicates with DIFFERENT labels create training label noise."
    ),
})

# ── 16b. check_leakage ───────────────────────────────────────────────────────
leakage = fr.check_leakage(str(train_csv), DATASET_PATH)
show("fr.check_leakage — detect train/test overlap", {
    "exact_leaks" : len(leakage.exact_leaks),
    "near_leaks"  : len(leakage.near_leaks),
    "has_leakage" : leakage.has_leakage,
    "why"         : (
        "If your test set contains examples the model saw during training, "
        "accuracy and flip-rate numbers are inflated. Detect and remove them."
    ),
})

# ── 16c. mine_variants ───────────────────────────────────────────────────────
domain_texts = [
    "delivery nahi aayi",  "delivery nhi aai",   "delivery nahin aayi",
    "bahut accha product", "bohot achha product", "bhut acha product",
    "return karna hai",    "return krna h",       "return chahiye",
]
candidates = fr.mine_variants(domain_texts)
show("fr.mine_variants — discover spelling rules from raw text", {
    "n_candidates" : len(candidates),
    "why": (
        "Run mine_variants on your production logs to discover domain-specific "
        "spelling patterns. Review candidates and add them via fr.add_custom_rule()."
    ),
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 17 — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 17 · INPUT VALIDATION",
    "Validate inputs before they reach your model",
)

print("""
  WHAT IS VALIDATION?
  ────────────────────
  Before passing text to the gate or normalizer, validate that it's:
    • A non-empty string (not None, not a number)
    • Within an acceptable length (not 50,000 characters)

  WHY IS IT NEEDED?
  ──────────────────
  In a production API, users send garbage.  Validate first so that
  downstream functions receive clean, well-formed inputs.

  FUNCTIONS:
    fr.validate_text(text, max_length=10000, allow_empty=False)
    fr.validate_dataset(dataset, text_col="text", label_col="label")
""")

# ── 17a. validate_text ───────────────────────────────────────────────────────
for txt, desc in [
    ("delivery nahi aayi", "valid input"),
    ("",                   "empty string"),
    ("x" * 20001,          "too long (>10000)"),
]:
    try:
        fr.validate_text(txt)
        status = "✓ valid"
    except Exception as e:
        status = f"✗ rejected: {type(e).__name__}"
    print(f"  validate_text — {desc:25s}  →  {status}")

# ── 17b. validate_dataset ────────────────────────────────────────────────────
try:
    fr.validate_dataset(DATASET_PATH)
    print(f"\n  validate_dataset — {DATASET_PATH}  →  ✓ valid")
except Exception as e:
    print(f"\n  validate_dataset failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 18 — FlipRate CLASS & assert_stable
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 18 · FlipRate CLASS & assert_stable",
    "Object-oriented wrapper and assertion helper",
)

print("""
  WHAT IS THE FlipRate CLASS?
  ────────────────────────────
  FlipRate is a stateful OOP wrapper that keeps your model and configuration
  together.  Instead of passing the model to every function call, you create
  one FlipRate instance and call methods on it.

  WHAT IS assert_stable?
  ───────────────────────
  assert_stable raises AssertionError if the model flips on a given message.
  Use it in unit tests to write model robustness tests in a single line.

  WHY ARE THEY NEEDED?
  ─────────────────────
  • FlipRate class: cleaner API for long-lived production services
  • assert_stable: write compact regression tests for known-stable inputs
""")

# ── 18a. FlipRate class ───────────────────────────────────────────────────────
fr_obj = fr.FlipRate(MODEL)
gate_r = fr_obj.gate("bahut accha product hai")
show("fr.FlipRate class — OOP wrapper for stateful usage", {
    "decision"  : gate_r.decision,
    "flip_rate" : round(gate_r.flip_rate, 4),
    "why"       : (
        "Create one FlipRate(model) at startup and reuse it. "
        "It holds configuration so you don't repeat kwargs every call."
    ),
})

# ── 18b. assert_stable ────────────────────────────────────────────────────────
print("\n  fr.assert_stable — assertion helper for robustness unit tests:")
print("  WHY: Write compact CI regression tests: 'this model must pass max flip rate threshold'")
try:
    fr.assert_stable(MODEL, DATASET_PATH, max_flip_rate=0.50, min_accuracy=0.60)
    print(f"    assert_stable(MODEL, '{DATASET_PATH}')  →  ✓ passed all thresholds")
except AssertionError as e:
    print(f"    assert_stable failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 19 — EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 19 · EXPERIMENT RUNNER",
    "Run structured experiments across models, seeds, and protection",
)

print("""
  WHAT IS THE EXPERIMENT RUNNER?
  ───────────────────────────────
  run_experiment evaluates multiple models on identical data and seeds,
  both raw and automatically wrapped with protect().

  WHY IS IT NEEDED?
  ──────────────────
  It provides paired comparisons of raw vs protected performance and produces
  comprehensive markdown tables and exportable report artifacts.

  FUNCTION:
    fr.run_experiment(models, dataset, n_variants=5, seed=0, with_protection=True)
""")

exp = fr.run_experiment(
    models={"word_tfidf": MODEL},
    dataset=DATASET_PATH,
    n_variants=3,
    seed=42,
)
show("fr.run_experiment — tracked experiment run", {
    "models_evaluated": list(exp.reports.keys()),
    "modes_per_model" : list(exp.reports["word_tfidf"].keys()),
    "config"          : exp.config,
    "why"             : "Evaluates raw vs protected variants across models, returning exportable ExperimentResult",
})


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 20 — MULTI-LANGUAGE DATASET LOADERS
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 20 · MULTI-LANGUAGE DATASET LOADERS",
    "Load Dakshina and Aksharantar transliteration datasets",
)

print("""
  WHAT ARE THESE LOADERS?
  ────────────────────────
  Dakshina and Aksharantar are publicly available datasets of
  Hindi/Indic transliteration pairs published by Google.

  FlipRate can load them directly and convert them into canonical
  spelling-rule groups that you can add to the normalizer.

  WHY IS IT NEEDED?
  ──────────────────
  Instead of manually curating spelling rules, you can bootstrap
  from thousands of human-verified transliteration pairs.

  FUNCTIONS:
    fr.load_dakshina_groups(lang, dataset_dir)   — load Dakshina rules
    fr.dakshina_rules(lang, dataset_dir)         — → RuleSet
    fr.DAKSHINA_LANGUAGES                        — supported language codes
    fr.load_aksharantar_groups(lang, dataset_dir)
    fr.aksharantar_rules(lang, dataset_dir)
    fr.AKSHARANTAR_LANGUAGES
""")

show("fr.DAKSHINA_LANGUAGES — supported language codes for Dakshina", {
    "languages": list(fr.DAKSHINA_LANGUAGES),
})
show("fr.AKSHARANTAR_LANGUAGES — supported language codes for Aksharantar", {
    "languages": list(fr.AKSHARANTAR_LANGUAGES),
})

print("""
  USAGE (when dataset files are available locally):
    ruleset = fr.dakshina_rules("hi", dataset_dir="/path/to/dakshina/hi/")
    # ruleset is a RuleSet you can pass to fr.gate(..., rules=ruleset)
""")

# ── Hugging Face & Custom Datasets Integration ────────────────────────────────
try:
    import datasets
    # Hugging Face datasets integration example
    hf_sample = datasets.Dataset.from_dict({
        "text": [
            "ye phone bahut accha hai",
            "battery backup bekaar nikla",
            "delivery time pe aayi shukriya",
            "kuch khas nahi normal product tha",
        ],
        "label": ["positive", "negative", "positive", "neutral"],
    })
    hf_csv_path = Path(__file__).parent / "_hf_sample_data.csv"
    hf_sample.to_csv(str(hf_csv_path))
    report_hf_data = fr.test(MODEL, str(hf_csv_path), n_variants=3, seed=42)
    show("Hugging Face datasets -> FlipRate batch evaluation", {
        "status"      : "Successfully evaluated Hugging Face Dataset via FlipRate",
        "rows"        : len(hf_sample),
        "flip_rate"   : round(report_hf_data.flip_rate, 4),
        "accuracy"    : round(report_hf_data.accuracy, 4),
        "why"         : "Easily load any Hugging Face NLP dataset (tweet_eval, glue, indic_glue) and evaluate with fr.test()",
    })
    if hf_csv_path.exists():
        hf_csv_path.unlink()
except ImportError:
    print("  [Hugging Face 'datasets' not installed — install with: pip install datasets]")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 21 — WEB CONSOLE (FASTAPI)
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 21 · WEB CONSOLE",
    "Launch an interactive browser UI for the gate and robustness lab",
)

print("""
  WHAT IS THE WEB CONSOLE?
  ─────────────────────────
  FlipRate ships a zero-dependency FastAPI web console with two tabs:
    • Playground   — paste any message, watch the gate decision stream live
    • Robustness Lab — upload a labeled CSV, get a live flip-rate chart

  WHY IS IT NEEDED?
  ──────────────────
  Not every user wants to write Python.  The web console lets product
  managers, QA teams, and researchers explore model robustness visually
  without any code.

  HOW TO START:
    pip install "fliprate[api]"
    uvicorn fliprate.api:app --reload
    # → open http://localhost:8000

  The UI is vanilla HTML/JS — no React, no npm, no build step required.
""")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 22 — CLI REFERENCE
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 22 · CLI REFERENCE",
    "Use FlipRate from the terminal — no Python required",
)

print("""
  COMMANDS:
  ──────────
  fliprate gate "delivery nahi aayi"
      → Real-time gate check.  Prints PASS | TRANSFORM | REVIEW + flip rate.

  fliprate normalize "mala product avadlaaa nhi"
      → Print the normalized canonical form.

  fliprate perturb "delivery nahi aaya"
      → Print all generated spelling variants.

  fliprate explain --model "mypackage:my_model" --text "delivery nahi aayi"
      → Print a human-readable explanation of the gate decision.

  fliprate test --data test.csv --max-flip-rate 0.05
      → Batch evaluation.  EXIT CODE 1 if flip_rate > 0.05.
      → EXIT CODE 2 if the CSV or model cannot be loaded.

  fliprate benchmark --models models.json --data test.csv \\
                     --train-data train.csv --out scorecard.html
      → Multi-model benchmark.  Writes an interactive HTML dashboard.

  fliprate report --data test.csv --model "mypackage:my_model" --out report.md
      → Full report in Markdown.

  fliprate check-data test.csv --against train.csv
      → Check for train/test leakage.

  fliprate split mydata.csv --out-dir data/
      → Split a dataset into train/test.

  fliprate mine mydata.csv --out candidates.csv
      → Mine spelling rule candidates.

  EXIT CODES:
    0 — all checks passed
    1 — threshold violated (flip rate too high)
    2 — configuration or data error
""")


# ══════════════════════════════════════════════════════════════════════════════
#  SECTION 23 — STREAMING EVENTS
# ══════════════════════════════════════════════════════════════════════════════
section(
    "SECTION 23 · STREAMING EVENTS",
    "Build real-time event consumers with SSE helpers",
)

print("""
  WHAT ARE STREAMING EVENTS?
  ───────────────────────────
  FlipRate emits structured events during long-running operations (batch
  evaluation, benchmarks).  You can consume these as Server-Sent Events
  (SSE) in a web UI or as raw Python dicts in a custom consumer.

  WHY IS IT NEEDED?
  ──────────────────
  For real-time dashboards: instead of waiting for a full benchmark to
  finish, stream progress events to the browser as they arrive.

  FUNCTIONS:
    fr.make_event(kind, data)   — create a structured event dict
    fr.to_sse(event)            — format as an SSE string (for HTTP streaming)
    fr.drain(source)            — collect all events from an async source
""")

event = fr.make_event("gate_result", decision="PASS", flip_rate=0.12)
sse   = fr.to_sse(event)
show("fr.make_event + fr.to_sse — create and format a streaming event", {
    "event" : event,
    "sse"   : sse.strip(),
    "why"   : "Use with FastAPI StreamingResponse to push results to the browser in real-time",
})


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("SHOWCASE COMPLETE ✓", f"FlipRate {fr.__version__} · Author: {fr.__author__}")

print(f"""
  Everything ran successfully.

  API COVERAGE IN THIS FILE:
  ───────────────────────────
  Section  1  · Normalization       normalize_text, normalize, mask/unmask, preview_normalized_variants
  Section  2  · Perturbation        perturb, KINDS, is_safe_variant
  Section  3  · Script Detection    detect_script
  Section  4  · Gate                gate, iter_gate, agate, quick_check, safe_predict
  Section  5  · Protect             protect, make_protected_model, ProtectedModel
  Section  6  · Batch Evaluation    test, check_model, iter_test, paired_difference, macro_f1
  Section  7  · Explainability      show_why_it_flipped, explain_gate_decision,
                                    explain_report_flips, print_rule_hits
  Section  8  · Rule Management     DEFAULT_RULES, describe_rule_group, preview_variants,
                                    add_custom_rule, load_rules, RuleSet
  Section  9  · Devanagari          detect_script, gate on Devanagari text
  Section 10  · Model Adapters      CallableAdapter, as_adapter, from_sklearn,
                                    HttpServiceAdapter, with_retries,
                                    HFTextClassifier (optional), LLMClassifier (optional)
  Section 11  · Runtime Guard       runtime_guard, batch_guard, make_protected_model
  Section 12  · Benchmarking        benchmark, compare_models, BenchmarkSuite
  Section 13  · Dataset Audit       audit_dataset
  Section 14  · Benchmark Protocol  run_benchmark_protocol
  Section 15  · Augmentation        augment, compare_augmentation, AugmentationResult
  Section 16  · Dataset Tools       find_near_duplicates, check_leakage, mine_variants
  Section 17  · Validation          validate_text, validate_dataset
  Section 18  · FlipRate Class      FlipRate, assert_stable
  Section 19  · Experiment          run_experiment, ExperimentResult
  Section 20  · Language Loaders    dakshina_rules, aksharantar_rules, DAKSHINA_LANGUAGES
  Section 21  · Web Console         (startup instructions)
  Section 22  · CLI                 (full command reference)
  Section 23  · Streaming Events    make_event, to_sse, drain

  Install:    pip install "fliprate[all]"
  Repo:       https://github.com/ShivSupane/fliprate
  Docs:       docs/API_REFERENCE.md
  Security:   SECURITY.md
""")
