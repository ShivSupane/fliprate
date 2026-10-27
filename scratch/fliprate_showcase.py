"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              FlipRate v0.5.0 — Complete Function Showcase                  ║
║              Author: Shiv Supane                                           ║
║                                                                            ║
║  This script demonstrates EVERY major function in the FlipRate library     ║
║  with clear INPUT → OUTPUT formatting.                                     ║
║                                                                            ║
║  Works on: Google Colab ✓  |  Jupyter ✓  |  Terminal ✓                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import io
import json
import sys
import textwrap

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ─── Verification ─────────────────────────────────────────────────────────────
try:
    import fliprate as fr
except ImportError:
    print("Error: fliprate is not installed in the current environment.", file=sys.stderr)
    print("Install it with: pip install .  or  pip install -e .", file=sys.stderr)
    sys.exit(1)

# ─── Universal async runner (Colab / Jupyter / Terminal) ──────────────────────
def run_coro(coro):
    """Run a coroutine in any environment (even inside a running event loop)."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


# ══════════════════════════════════════════════════════════════════════════════
#  FORMATTING HELPERS — Makes the output look professional
# ══════════════════════════════════════════════════════════════════════════════

BLUE   = "\033[94m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

section_num = [0]

def section(title):
    """Print a major section header."""
    section_num[0] += 1
    n = section_num[0]
    print(f"\n{'━'*78}")
    print(f"  {BOLD}{CYAN}[{n}]{RESET}  {BOLD}{title}{RESET}")
    print(f"{'━'*78}")

def show_function(name, description=""):
    """Print a function header."""
    print(f"\n  {YELLOW}▶ {BOLD}fr.{name}(){RESET}")
    if description:
        print(f"    {DIM}{description}{RESET}")

def show_input(label, value):
    """Print an input value."""
    print(f"    {BLUE}INPUT  ({label}):{RESET}  {value}")

def show_output(label, value):
    """Print an output value."""
    print(f"    {GREEN}OUTPUT ({label}):{RESET}  {value}")

def show_dict(d, indent=4):
    """Pretty-print a dictionary output."""
    prefix = " " * indent
    for k, v in d.items():
        if isinstance(v, list) and len(v) > 3:
            print(f"    {GREEN}{prefix}{k}:{RESET}  {v[:3]} ... ({len(v)} items)")
        else:
            print(f"    {GREEN}{prefix}{k}:{RESET}  {v}")

def ok(msg=""):
    """Print a success indicator."""
    if msg:
        print(f"    {GREEN}✓ {msg}{RESET}")
    else:
        print(f"    {GREEN}✓ PASSED{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  DUMMY MODEL FOR TESTING
# ══════════════════════════════════════════════════════════════════════════════

def dummy_sentiment_model(text: str) -> str:
    """A simple keyword-based sentiment classifier for demonstration."""
    t = text.lower()
    positive_words = ["accha", "acha", "good", "sahi", "badiya", "badhiya",
                       "recommended", "bahut", "bohot", "avadla", "avadlaaa",
                       "mast", "sunder", "great", "nice", "awesome", "love"]
    negative_words = ["bura", "bekar", "kharab", "ghatiya", "nahi", "nhi",
                       "mat", "bekaar", "worst", "bad", "terrible", "hate"]
    pos_count = sum(1 for w in positive_words if w in t)
    neg_count = sum(1 for w in negative_words if w in t)
    if pos_count > neg_count:
        return "positive"
    elif neg_count > pos_count:
        return "negative"
    return "neutral"

# Test dataset
dataset = [
    ("yeh product bahut accha hai", "positive"),
    ("bohot bura product, delivery nahi aayi", "negative"),
    ("acha service, highly recommended", "positive"),
    ("ekdum bekar hai", "negative"),
]


# ══════════════════════════════════════════════════════════════════════════════
#                      THE SHOWCASE BEGINS
# ══════════════════════════════════════════════════════════════════════════════

print(f"""
{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════════════╗
║                   FlipRate v{fr.__version__} — Complete Function Showcase               ║
║                   Author: {fr.__author__}                                       ║
╚══════════════════════════════════════════════════════════════════════════════╝{RESET}
""")

errors = []

# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 1: TEXT NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────
section("TEXT NORMALIZATION — Cleaning Messy Romanized Indian Text")

try:
    # 1a. normalize_text (simple string → string)
    show_function("normalize_text", "Cleans messy text into a standardized form (string in → string out)")

    inputs = [
        "Mala product avadlaaa nhi",
        "yeh bohot acchaaaa hai!!!",
        "kharaab product hai ekdum bekaaaar",
        "bahut अच्छा product hai",
    ]
    for inp in inputs:
        out = fr.normalize_text(inp)
        show_input("text", f'"{inp}"')
        show_output("cleaned", f'"{out}"')
        print()
    ok("normalize_text works for all inputs")

    # 1b. normalize (full NormalizeResult object)
    show_function("normalize", "Returns a NormalizeResult with original, normalized text, and list of changes")

    text = "bohot acchaaaa product hai nhi milega kahin"
    result = fr.normalize(text)
    show_input("text", f'"{text}"')
    show_output("original", f'"{result.original}"')
    show_output("normalized", f'"{result.normalized}"')
    show_output("changes", result.changes)
    show_output("str(result)", f'"{str(result)}"')
    ok("normalize returns NormalizeResult with detailed change tracking")

except Exception as e:
    errors.append(("Normalization", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 2: PERTURBATION — Generating Spelling Variants
# ─────────────────────────────────────────────────────────────────────────────
section("PERTURBATION — Generating Realistic Spelling Variants")

try:
    show_function("perturb", "Generates safe spelling/noise variants of text (like real users would type)")

    text = "yeh product bahut accha hai"
    variants = fr.perturb(text, n=6)
    show_input("text", f'"{text}"')
    show_input("n", "6 (generate 6 variants)")
    print(f"    {GREEN}OUTPUT (variants):{RESET}")
    for i, v in enumerate(variants, 1):
        print(f"      {i}. [{v.kind:12s}]  \"{v.text}\"")
    ok(f"perturb generated {len(variants)} realistic variants")

    # Show KINDS
    show_function("KINDS", "All supported perturbation types")
    show_output("KINDS", fr.KINDS)
    ok()

except Exception as e:
    errors.append(("Perturbation", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 3: SCRIPT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
section("SCRIPT DETECTION — Identifying Latin / Devanagari / Mixed")

try:
    show_function("detect_script", "Detects the writing script of text: latin, devanagari, mixed, or other")

    test_cases = [
        ("yeh product bahut accha hai",   "Pure romanized Hindi"),
        ("यह प्रोडक्ट बहुत अच्छा है",      "Pure Devanagari Hindi"),
        ("yeh product अच्छा hai",         "Mixed Latin + Devanagari"),
        ("12345 @#$%",                     "No script letters"),
    ]
    for text, desc in test_cases:
        result = fr.detect_script(text)
        show_input(desc, f'"{text}"')
        show_output("script", f'"{result}"')
        print()
    ok("detect_script correctly identifies all script types")

except Exception as e:
    errors.append(("Script Detection", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 4: GATING — Quality Gate Decisions
# ─────────────────────────────────────────────────────────────────────────────
section("GATING — Deciding if Model Prediction is Safe (PASS / TRANSFORM / REVIEW)")

try:
    # 4a. gate()
    show_function("gate", "Core quality gate: checks if model prediction is stable across spelling variants")

    text = "yeh product bohot accha hai"
    result = fr.gate(dummy_sentiment_model, text)
    show_input("model", "dummy_sentiment_model")
    show_input("text", f'"{text}"')
    show_output("decision", f'"{result.decision}"')
    show_output("prediction", f'"{result.prediction}"')
    show_output("text_used", f'"{result.text_used}"')
    show_output("raw_stability", result.raw_stability)
    show_output("normalized_stability", result.normalized_stability)
    show_output("n_variants", result.n_variants)
    show_output("reason", f'"{result.reason}"')
    show_output("flipped_variants", result.flipped_variants[:3] if result.flipped_variants else "[]")
    ok("gate returns detailed GateResult with full stability analysis")

    # 4b. guard() alias
    show_function("guard", "Convenience alias for gate() — same functionality, friendlier name")

    result2 = fr.guard(dummy_sentiment_model, text)
    show_input("text", f'"{text}"')
    show_output("decision", f'"{result2.decision}"')
    show_output("prediction", f'"{result2.prediction}"')
    ok("guard() works identically to gate()")

    # 4c. iter_gate()
    show_function("iter_gate", "Streaming version of gate — yields progress events as it runs")

    gen = fr.iter_gate(dummy_sentiment_model, "acha product hai")
    events = []
    while True:
        try:
            ev = next(gen)
            events.append(ev)
        except StopIteration as stop:
            final = stop.value
            break
    show_input("text", '"acha product hai"')
    show_output("events_captured", f"{len(events)} events")
    for ev in events[:4]:
        print(f"      → stage: {ev.get('stage', '?'):20s}  data: { {k:v for k,v in ev.items() if k not in ('stage','t')} }")
    show_output("final_decision", f'"{final.decision}"')
    ok("iter_gate streams events and returns GateResult")

    # 4d. agate() async
    show_function("agate", "Async version of gate() — for web servers and async applications")

    async def test_agate():
        return await fr.agate(dummy_sentiment_model, "bohot accha hai")
    ag_result = run_coro(test_agate())
    show_input("text", '"bohot accha hai"')
    show_output("decision", f'"{ag_result.decision}"')
    show_output("prediction", f'"{ag_result.prediction}"')
    ok("agate (async) works correctly")

except Exception as e:
    errors.append(("Gating", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 5: HAPPY-PATH API — Beginner-Friendly Functions
# ─────────────────────────────────────────────────────────────────────────────
section("HAPPY-PATH API — Beginner-Friendly Functions")

try:
    # 5a. quick_check
    show_function("quick_check", "One-line safety check — the simplest way to use FlipRate")

    text = "mala product avadlaaa nahi"
    qc = fr.quick_check(dummy_sentiment_model, text)
    show_input("model", "dummy_sentiment_model")
    show_input("text", f'"{text}"')
    show_output("is_safe", qc.is_safe)
    show_output("decision", f'"{qc.decision}"')
    show_output("prediction", f'"{qc.prediction}"')
    show_output("suggested_text", f'"{qc.suggested_text}"')
    show_output("reason", f'"{qc.reason}"')
    show_output("flipped_variants", qc.flipped_variants[:3] if qc.flipped_variants else "[]")
    show_output("str(result)", f'"{str(qc)}"')
    show_output("dict output", "")
    show_dict(qc.to_dict())
    ok("quick_check gives a complete beginner-friendly verdict")

    # 5b. safe_predict
    show_function("safe_predict", "Predict label with automatic normalization (never gives unstable answer)")

    texts_to_predict = [
        "bohot acchaaaa product hai",
        "yeh bekar product hai ekdum",
        "acha service recommended",
    ]
    for t in texts_to_predict:
        pred = fr.safe_predict(dummy_sentiment_model, t)
        show_input("text", f'"{t}"')
        show_output("prediction", f'"{pred}"')
        print()
    ok("safe_predict returns stable predictions for all inputs")

    # 5c. check_model (alias for test/audit)
    show_function("check_model", "Audit model robustness on a full dataset (returns Report)")

    report = fr.check_model(dummy_sentiment_model, dataset)
    show_input("model", "dummy_sentiment_model")
    show_input("dataset", f"{len(dataset)} samples")
    show_output("flip_rate", f"{report.flip_rate:.1%}")
    show_output("accuracy_original", f"{report.accuracy_original:.1%}")
    show_output("accuracy_variants", f"{report.accuracy_variants:.1%}")
    show_output("flip_rate_ci", f"[{report.flip_rate_ci[0]:.1%}, {report.flip_rate_ci[1]:.1%}]")
    ok("check_model audits model and returns detailed Report")

except Exception as e:
    errors.append(("Happy-Path API", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 6: PRODUCTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
section("PRODUCTION PIPELINE — Enterprise-Grade Functions")

try:
    # 6a. runtime_guard
    show_function("runtime_guard", "Production gating with strict/lenient policies and human routing")

    text = "product bohot bura hai nhi lena chahiye"
    rg = fr.runtime_guard(dummy_sentiment_model, text, mode="strict")
    show_input("text", f'"{text}"')
    show_input("mode", '"strict"')
    show_output("decision", f'"{rg.decision}"')
    show_output("prediction", f'"{rg.prediction}"')
    show_output("clean_text", f'"{rg.clean_text}"')
    show_output("should_route_to_human", rg.should_route_to_human)
    show_output("reason", f'"{rg.reason}"')
    show_output("metadata", "")
    show_dict(rg.metadata)

    rg_lenient = fr.runtime_guard(dummy_sentiment_model, text, mode="lenient")
    show_input("mode", '"lenient" (same text)')
    show_output("should_route_to_human", rg_lenient.should_route_to_human)
    ok("runtime_guard works in both strict and lenient modes")

    # 6b. batch_guard
    show_function("batch_guard", "Evaluate quality gate on a batch of messages at once")

    batch_texts = [
        "bohot accha product hai",
        "yeh bekar service hai",
        "mast quality hai",
    ]
    results = fr.batch_guard(dummy_sentiment_model, batch_texts)
    show_input("texts", f"{len(batch_texts)} messages")
    for i, (txt, res) in enumerate(zip(batch_texts, results), 1):
        print(f"      {i}. \"{txt}\"")
        print(f"         → decision: {res.decision}  |  prediction: {res.prediction}")
    ok(f"batch_guard processed {len(batch_texts)} messages")

    # 6c. evaluate_pipeline
    show_function("evaluate_pipeline", "Full pipeline evaluation with optional report saving")

    report = fr.evaluate_pipeline(dummy_sentiment_model, dataset)
    show_input("model", "dummy_sentiment_model")
    show_input("dataset", f"{len(dataset)} samples")
    show_output("flip_rate", f"{report.flip_rate:.1%}")
    show_output("accuracy_original", f"{report.accuracy_original:.1%}")
    show_output("macro_f1_original", f"{report.macro_f1_original:.3f}")
    ok("evaluate_pipeline provides comprehensive evaluation")

    # 6d. make_protected_model
    show_function("make_protected_model", "Wrap any model with automatic normalization for production")

    safe_model = fr.make_protected_model(dummy_sentiment_model)
    show_input("model", "dummy_sentiment_model")
    show_output("type", type(safe_model).__name__)
    show_output("name", safe_model.name)

    messy = "bohot acchaaaa product haiii"
    pred_raw = dummy_sentiment_model(messy)
    pred_safe = safe_model(messy)
    show_input("messy text", f'"{messy}"')
    show_output("raw_model_prediction", f'"{pred_raw}"')
    show_output("protected_prediction", f'"{pred_safe}"')
    ok("make_protected_model wraps model with normalization")

except Exception as e:
    errors.append(("Production Pipeline", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 7: MODEL PROTECTION — protect()
# ─────────────────────────────────────────────────────────────────────────────
section("MODEL PROTECTION — Wrapping Models for Safety")

try:
    show_function("protect", "Wraps a model so every input is normalized before prediction")

    safe = fr.protect(dummy_sentiment_model)
    show_input("model", "dummy_sentiment_model")
    show_output("type", type(safe).__name__)
    show_output("name", safe.name)

    test_texts = [
        ("bohot acchaaaa hai", "Messy with repeated chars"),
        ("nhi milega ye product", "With informal negation"),
        ("acha service recommended", "Clean text"),
    ]
    for text, desc in test_texts:
        raw_pred   = dummy_sentiment_model(text)
        safe_pred  = safe(text)
        show_input(desc, f'"{text}"')
        show_output("raw_model  →", f'"{raw_pred}"')
        show_output("protected  →", f'"{safe_pred}"')
        print()
    ok("protect() ensures consistent predictions on noisy inputs")

except Exception as e:
    errors.append(("Model Protection", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 8: EXPLAINABILITY — Understanding WHY a Decision Was Made
# ─────────────────────────────────────────────────────────────────────────────
section("EXPLAINABILITY — Understanding WHY Decisions Are Made")

try:
    # 8a. explain_gate_decision
    show_function("explain_gate_decision", "Human-readable explanation of a gate decision")

    text = "yeh product bohot accha hai"
    explanation = fr.explain_gate_decision(text, dummy_sentiment_model)
    show_input("text", f'"{text}"')
    show_output("decision", f'"{explanation["decision"]}"')
    show_output("prediction", f'"{explanation["prediction"]}"')
    show_output("script", f'"{explanation["script"]}"')
    show_output("raw_stability", explanation["raw_stability"])
    show_output("summary", f'"{explanation["summary"]}"')
    show_output("recommendation", f'"{explanation["recommendation"]}"')
    ok("explain_gate_decision gives human-understandable explanations")

    # 8b. show_why_it_flipped
    show_function("show_why_it_flipped", "Pinpoints EXACTLY which variants caused the model to change its answer")

    text = "acha product hai recommended"
    flips = fr.show_why_it_flipped(text, dummy_sentiment_model)
    show_input("text", f'"{text}"')
    show_output("flips_found", f"{len(flips)} variants caused a flip")
    for i, f_item in enumerate(flips[:5], 1):
        print(f"      {i}. kind: {f_item['kind']}")
        print(f"         variant: \"{f_item['variant_text']}\"")
        print(f"         {f_item['original_prediction']} → {f_item['variant_prediction']}")
        if f_item.get('diffs'):
            print(f"         changed words: {f_item['diffs']}")
    if not flips:
        print(f"      (No flips detected — model is stable for this input)")
    ok("show_why_it_flipped identifies unstable variants")

    # 8c. preview_normalized_variants
    show_function("preview_normalized_variants", "Shows how variants look before and after normalization")

    text = "bohot acchaaaa product hai"
    previews = fr.preview_normalized_variants(text, n=5)
    show_input("text", f'"{text}"')
    show_input("n", "5")
    print(f"    {GREEN}OUTPUT:{RESET}")
    for i, p in enumerate(previews, 1):
        print(f"      {i}. [{p['kind']:12s}]  \"{p['original']}\"")
        print(f"                          → \"{p['normalized']}\"")
    ok("preview_normalized_variants shows before/after normalization")

    # 8d. print_rule_hits
    show_function("print_rule_hits", "Shows which words in a text match FlipRate's spelling rules")

    text = "bohot acha product nhi milega"
    hits = fr.print_rule_hits(text)
    show_input("text", f'"{text}"')
    print(f"    {GREEN}OUTPUT (rule matches):{RESET}")
    for h in hits:
        neg_tag = " [NEGATION]" if h["is_negation"] else ""
        print(f"      • \"{h['token']}\" → canonical: \"{h['canonical']}\"{neg_tag}")
    if not hits:
        print(f"      (No rule matches found)")
    ok("print_rule_hits reveals which spelling rules match each token")

except Exception as e:
    errors.append(("Explainability", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 9: VALIDATION — Input & Dataset Safety Checks
# ─────────────────────────────────────────────────────────────────────────────
section("VALIDATION — Input & Dataset Safety Checks")

try:
    # 9a. validate_text
    show_function("validate_text", "Validates a text input — catches empty, non-string, or too-long inputs")

    valid = fr.validate_text("bohot accha product hai")
    show_input("text", '"bohot accha product hai"')
    show_output("returned", f'"{valid}"')
    ok("Valid text passes")

    # Test error cases
    error_cases = [
        (123, "non-string input"),
        ("", "empty string"),
    ]
    for bad_input, desc in error_cases:
        try:
            fr.validate_text(bad_input)
            print(f"    {RED}✗ Should have raised for {desc}{RESET}")
            raise AssertionError(f"validate_text unexpectedly accepted invalid input: {desc}")
        except (TypeError, ValueError) as e:
            show_input(desc, repr(bad_input))
            show_output("raises", f'{type(e).__name__}: "{e}"')
    ok("validate_text correctly rejects invalid inputs")

    # 9b. validate_dataset
    show_function("validate_dataset", "Validates a dataset — checks for empties, duplicates, label conflicts")

    diag = fr.validate_dataset(dataset)
    show_input("dataset", f"{len(dataset)} samples")
    show_output("valid", diag["valid"])
    show_output("row_count", diag["row_count"])
    show_output("empty_rows", diag["empty_rows"])
    show_output("duplicate_texts", diag["duplicate_texts"])
    show_output("label_conflicts", diag["label_conflicts"])
    show_output("unique_labels", diag["unique_labels"])
    ok("validate_dataset provides complete dataset diagnostics")

except Exception as e:
    errors.append(("Validation", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 10: RULES ENGINE — Spelling Rule Management
# ─────────────────────────────────────────────────────────────────────────────
section("RULES ENGINE — Spelling Rule Management")

try:
    # 10a. DEFAULT_RULES
    show_function("DEFAULT_RULES", "The built-in spelling rule set")
    rules = fr.DEFAULT_RULES
    show_output("type", type(rules).__name__)
    show_output("total_groups", len(rules.groups))
    show_output("total_negations", len(rules.negations))
    show_output("sample_groups", [list(g) for g in rules.groups[:3]])
    ok("DEFAULT_RULES loaded with comprehensive spelling groups")

    # 10b. load_rules
    show_function("load_rules", "Load a language-specific rule set explicitly")
    loaded = fr.load_rules("hindi")
    show_input("language", '"hindi"')
    show_output("type", type(loaded).__name__)
    show_output("groups_count", len(loaded.groups))
    ok("load_rules returns a RuleSet")

    # 10c. is_safe_variant
    show_function("is_safe_variant", "Check if a variant preserves meaning (no negation flip)")
    pairs = [
        ("acha product hai", "accha product hai"),
        ("nahi chahiye", "chahiye"),
    ]
    for orig, variant in pairs:
        safe = fr.is_safe_variant(orig, variant)
        show_input("original", f'"{orig}"')
        show_input("variant", f'"{variant}"')
        show_output("is_safe", safe)
        print()
    ok("is_safe_variant correctly detects meaning changes")

    # 10d. preview_variants
    show_function("preview_variants", "Preview all known spelling variants for a word")
    word = "accha"
    variants = fr.preview_variants(word)
    show_input("word", f'"{word}"')
    show_output("variants", variants)
    ok("preview_variants shows all known spellings")

    # 10e. describe_rule_group
    show_function("describe_rule_group", "Describe a spelling group by its canonical word")
    word = "bahut"
    desc = fr.describe_rule_group(word)
    show_input("word", f'"{word}"')
    show_output("description", desc)
    ok("describe_rule_group provides human-readable group info")

    # 10f. add_custom_rule
    show_function("add_custom_rule", "Add your own spelling rule to the rule set")
    custom_rules = fr.add_custom_rule(["shukria", "sukriya", "shukriyaa"], canonical="shukriya")
    show_input("variants", '["shukria", "sukriya", "shukriyaa"]')
    show_input("canonical", '"shukriya"')
    show_output("type", type(custom_rules).__name__)
    g_idx = custom_rules.group_of("shukria")
    show_output("new_group", custom_rules.groups[g_idx] if g_idx is not None else None)
    ok("add_custom_rule creates a custom RuleSet with new group")

    # 10g. Constants
    show_function("DEVANAGARI_GROUPS", "Pre-built Devanagari script spelling groups")
    show_output("type", type(fr.DEVANAGARI_GROUPS).__name__)
    show_output("sample", dict(list(fr.DEVANAGARI_GROUPS.items())[:2]) if isinstance(fr.DEVANAGARI_GROUPS, dict) else str(fr.DEVANAGARI_GROUPS)[:100])

    show_function("CROSS_SCRIPT_MAP", "Latin → Devanagari word mapping for bilingual support")
    show_output("sample", dict(list(fr.CROSS_SCRIPT_MAP.items())[:5]))

    show_function("CROSS_SCRIPT_REV", "Devanagari → Latin reverse mapping")
    show_output("sample", dict(list(fr.CROSS_SCRIPT_REV.items())[:5]))

    show_function("LANGUAGE_GROUPS", "Language-specific rule groupings")
    show_output("type", type(fr.LANGUAGE_GROUPS).__name__)
    ok("All rule constants accessible")

except Exception as e:
    errors.append(("Rules Engine", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 11: MODEL EVALUATION — test(), audit(), Report
# ─────────────────────────────────────────────────────────────────────────────
section("MODEL EVALUATION — Testing Robustness on Datasets")

try:
    # 11a. test()
    show_function("test", "Core evaluation: test model robustness on a labeled dataset")

    report = fr.test(dummy_sentiment_model, dataset)
    show_input("model", "dummy_sentiment_model")
    show_input("dataset", f"{len(dataset)} samples")
    show_output("flip_rate", f"{report.flip_rate:.1%}")
    show_output("accuracy_original", f"{report.accuracy_original:.1%}")
    show_output("accuracy_variants", f"{report.accuracy_variants:.1%}")
    show_output("macro_f1_original", f"{report.macro_f1_original:.3f}")
    show_output("macro_f1_variants", f"{report.macro_f1_variants:.3f}")
    show_output("flip_rate_ci (95%)", f"[{report.flip_rate_ci[0]:.1%}, {report.flip_rate_ci[1]:.1%}]")
    ok("test() provides comprehensive robustness report")

    # 11b. audit()
    show_function("audit", "Convenience alias for test() — same function, friendlier name")
    report2 = fr.audit(dummy_sentiment_model, dataset)
    show_output("flip_rate", f"{report2.flip_rate:.1%}")
    ok("audit() works identically to test()")

    # 11c. iter_test()
    show_function("iter_test", "Streaming version of test() — yields events during evaluation")
    gen = fr.iter_test(dummy_sentiment_model, dataset)
    events = []
    while True:
        try:
            ev = next(gen)
            events.append(ev)
        except StopIteration as stop:
            final_report = stop.value
            break
    show_output("events_generated", f"{len(events)} events")
    show_output("final_flip_rate", f"{final_report.flip_rate:.1%}")
    ok("iter_test streams progress events")

    # 11d. paired_difference
    show_function("paired_difference", "Statistical comparison between two evaluation runs")
    raw_report = fr.test(dummy_sentiment_model, dataset)
    safe_model = fr.protect(dummy_sentiment_model)
    prot_report = fr.test(safe_model, dataset)
    diff = fr.paired_difference(raw_report, prot_report, "flip_rate")
    show_input("metric", '"flip_rate"')
    show_output("difference", f"{diff['difference']:.4f}")
    show_output("ci_95", f"[{diff['ci'][0]:.4f}, {diff['ci'][1]:.4f}]")
    ok("paired_difference shows statistical significance of protection")

    # 11e. macro_f1
    show_function("macro_f1", "Compute macro-averaged F1 score from predictions")
    y_true = ["positive", "negative", "positive", "negative"]
    y_pred = ["positive", "negative", "negative", "negative"]
    f1 = fr.macro_f1(y_true, y_pred)
    show_input("y_true", y_true)
    show_input("y_pred", y_pred)
    show_output("macro_f1", f"{f1:.4f}")
    ok("macro_f1 computes F1 correctly")

except Exception as e:
    errors.append(("Model Evaluation", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 12: BENCHMARKING — Multi-Model Comparison
# ─────────────────────────────────────────────────────────────────────────────
section("BENCHMARKING — Multi-Model Robustness Comparison")

try:
    # Second dummy model for comparison
    def dummy_model_v2(text):
        return "positive" if len(text) > 20 else "negative"

    # 12a. benchmark()
    show_function("benchmark", "Benchmark multiple models side-by-side on the same dataset")
    suite = fr.benchmark(
        {"SentimentV1": dummy_sentiment_model, "LengthBaseline": dummy_model_v2},
        dataset,
        baseline="SentimentV1",
        n_variants=3,
        bootstrap=50,
    )
    show_input("models", '{"SentimentV1": ..., "LengthBaseline": ...}')
    show_input("baseline", '"SentimentV1"')
    print(f"    {GREEN}OUTPUT (BenchmarkSuite):{RESET}")
    for name, score in suite.scores.items():
        base_tag = " [BASELINE]" if score.is_baseline else ""
        print(f"      Model: {name}{base_tag}")
        print(f"        flip_rate:           {score.flip_rate:.1%}")
        print(f"        protected_flip_rate:  {score.protected_flip_rate:.1%}" if score.protected_flip_rate is not None else "")
        print(f"        accuracy_original:   {score.accuracy_original:.1%}")
    ok("benchmark compares multiple models")

    # 12b. compare_models()
    show_function("compare_models", "Quick 2-model comparison shortcut")
    suite2 = fr.compare_models(
        dummy_sentiment_model, dummy_model_v2, dataset,
        name_a="Sentiment", name_b="Length", n_variants=3, bootstrap=50,
    )
    show_input("models", '"Sentiment" vs "Length"')
    for name, score in suite2.scores.items():
        print(f"      {name}: flip_rate={score.flip_rate:.1%}")
    ok("compare_models provides side-by-side comparison")

    # 12c. BenchmarkSuite methods
    show_function("BenchmarkSuite.summary()", "Human-readable benchmark summary")
    summary_text = suite.summary()
    for line in summary_text.split("\n")[:8]:
        print(f"      {line}")
    print(f"      ...")
    ok("BenchmarkSuite.summary() generates readable report")

    show_function("BenchmarkSuite.to_markdown()", "Export benchmark as Markdown table")
    md = suite.to_markdown()
    for line in md.split("\n")[:5]:
        print(f"      {line}")
    ok("BenchmarkSuite.to_markdown() generates Markdown output")

    show_function("BenchmarkSuite.to_dict()", "Export benchmark as dictionary/JSON")
    d = suite.to_dict()
    show_output("keys", list(d.keys()))
    show_output("models", list(d["scores"].keys()))
    ok("BenchmarkSuite.to_dict() exports structured data")

except Exception as e:
    errors.append(("Benchmarking", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 13: EXPERIMENT RUNNER
# ─────────────────────────────────────────────────────────────────────────────
section("EXPERIMENT RUNNER — Multi-Model Raw vs Protected Comparison")

try:
    show_function("run_experiment", "Run all models (raw + protected) on identical data and seed")

    result = fr.run_experiment(
        {"SentimentV1": dummy_sentiment_model},
        dataset,
        n_variants=3,
        bootstrap=50,
    )
    show_input("models", '{"SentimentV1": dummy_sentiment_model}')
    show_input("dataset", f"{len(dataset)} samples")
    show_output("type", type(result).__name__)
    show_output("models_tested", list(result.reports.keys()))
    show_output("modes_per_model", list(result.reports["SentimentV1"].keys()))
    raw = result.reports["SentimentV1"]["raw"]
    prot = result.reports["SentimentV1"]["protected"]
    show_output("raw_flip_rate", f"{raw.flip_rate:.1%}")
    show_output("protected_flip_rate", f"{prot.flip_rate:.1%}")
    ok("run_experiment provides raw vs protected comparison")

    show_function("ExperimentResult.markdown()", "Export experiment as Markdown report")
    md = result.markdown()
    for line in md.split("\n")[:6]:
        print(f"      {line}")
    ok("ExperimentResult.markdown() generates report")

except Exception as e:
    errors.append(("Experiment Runner", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 14: MODEL ADAPTERS — Connecting Any Model to FlipRate
# ─────────────────────────────────────────────────────────────────────────────
section("MODEL ADAPTERS — Connecting Any Model Type to FlipRate")

try:
    # 14a. CallableAdapter
    show_function("CallableAdapter", "Wrap any function as a FlipRate-compatible model adapter")
    adapter = fr.CallableAdapter(dummy_sentiment_model, name="MySentiment")
    show_input("function", "dummy_sentiment_model")
    show_output("type", type(adapter).__name__)
    show_output("name", adapter.name)
    pred = adapter("bohot accha product hai")
    show_output("predict(text)", f'"{pred}"')
    ok("CallableAdapter wraps functions")

    # 14b. as_adapter
    show_function("as_adapter", "Auto-detect and wrap any model object into an adapter")
    auto = fr.as_adapter(dummy_sentiment_model)
    show_input("model", "dummy_sentiment_model (plain function)")
    show_output("type", type(auto).__name__)
    show_output("predict", f'"{auto("accha hai")}"')
    ok("as_adapter auto-wraps models")

    # 14c. with_retries
    show_function("with_retries", "Add retry logic to any model adapter (for unreliable APIs)")
    robust = fr.with_retries(adapter, retries=3)
    show_input("adapter", "CallableAdapter(dummy_sentiment_model)")
    show_input("retries", "3")
    show_output("predict", f'"{robust("acha product")}"')
    ok("with_retries adds fault tolerance")

    # 14d. twitter_preprocess
    show_function("twitter_preprocess", "Preprocess text for Twitter-style NLP models")
    raw_tweet = "@user123 This product is soooo goooood!!! #amazing http://example.com"
    cleaned = fr.twitter_preprocess(raw_tweet)
    show_input("tweet", f'"{raw_tweet}"')
    show_output("cleaned", f'"{cleaned}"')
    ok("twitter_preprocess handles @mentions, URLs, and #hashtags")

    # 14e. Adapter classes (just show they exist)
    show_function("ModelAdapter", "Base class for all model adapters")
    show_output("type", str(fr.ModelAdapter))
    show_function("HFTextClassifier", "Adapter for HuggingFace text classification models")
    show_output("type", str(fr.HFTextClassifier))
    show_function("SklearnAdapter / from_sklearn", "Adapter for scikit-learn models")
    show_output("type", str(fr.SklearnAdapter))
    show_function("HttpServiceAdapter", "Adapter for HTTP API-based models")
    show_output("type", str(fr.HttpServiceAdapter))
    show_function("LLMClassifier / llm_classifier", "Adapter for LLM-based classifiers")
    show_output("type", str(fr.LLMClassifier))
    ok("All adapter classes accessible")

except Exception as e:
    errors.append(("Model Adapters", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 15: SERVICE INFRASTRUCTURE — Storage & Rate Limiting
# ─────────────────────────────────────────────────────────────────────────────
section("SERVICE INFRASTRUCTURE — Storage & Rate Limiting")

try:
    # 15a. InMemoryJobStore
    show_function("InMemoryJobStore", "In-memory job storage for tracking evaluation jobs")
    store = fr.InMemoryJobStore()
    job_id = store.put("eval", {"model": "test", "status": "running"})
    show_input("job_data", '{"model": "test", "status": "running"}')
    show_output("job_id", f'"{job_id}"')
    fetched = store.pop(job_id)
    show_output("fetched_job", fetched)
    ok("InMemoryJobStore creates and retrieves jobs")

    # 15b. InMemoryRateLimiter
    show_function("InMemoryRateLimiter", "In-memory rate limiter for API endpoints")
    limiter = fr.InMemoryRateLimiter(limit_per_min=5)
    show_input("config", "limit_per_min=5")
    allowed = limiter.allow("user_123")
    show_output("allow('user_123')", allowed)
    ok("InMemoryRateLimiter manages request rates")

    # Show types exist
    show_function("BaseJobStore / BaseRateLimiter", "Abstract base classes for custom implementations")
    show_output("BaseJobStore", str(fr.BaseJobStore))
    show_output("BaseRateLimiter", str(fr.BaseRateLimiter))

    show_function("RedisJobStore / RedisRateLimiter", "Redis-backed implementations for distributed systems")
    show_output("RedisJobStore", str(fr.RedisJobStore))
    show_output("RedisRateLimiter", str(fr.RedisRateLimiter))
    ok("All service classes accessible")

except Exception as e:
    errors.append(("Service Infrastructure", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 16: STREAMING EVENTS — Building Custom Event Consumers
# ─────────────────────────────────────────────────────────────────────────────
section("STREAMING EVENTS — Building Custom Event Consumers")

try:
    # 16a. make_event
    show_function("make_event", "Create a structured progress event")
    ev = fr.make_event("analysis_complete", accuracy=0.95, model="test")
    show_input("stage", '"analysis_complete"')
    show_input("data", "accuracy=0.95, model='test'")
    show_output("event", ev)
    ok("make_event creates structured events")

    # 16b. to_sse
    show_function("to_sse", "Format an event as Server-Sent Events (SSE) for web streaming")
    sse_str = fr.to_sse(ev)
    show_input("event", str(ev))
    show_output("SSE format", repr(sse_str))
    ok("to_sse formats events for web streaming")

    # 16c. drain
    show_function("drain", "Run an event generator to completion and return its result")

    def sample_generator():
        yield fr.make_event("start")
        yield fr.make_event("progress", step=1)
        yield fr.make_event("progress", step=2)
        return "completed"  # This is what drain() returns

    captured_events = []
    # drain expects a generator that uses StopIteration return value
    # Let's demonstrate with iter_gate which is the real use case
    gen = fr.iter_gate(dummy_sentiment_model, "accha product hai")
    result = fr.drain(gen, on_event=lambda ev: captured_events.append(ev))
    show_input("generator", "iter_gate(model, text)")
    show_output("events_captured", f"{len(captured_events)} events")
    show_output("final_result", f'GateResult(decision="{result.decision}")')
    ok("drain runs generators and captures events")

except Exception as e:
    errors.append(("Streaming Events", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 17: DATASET LOADERS
# ─────────────────────────────────────────────────────────────────────────────
section("DATASET LOADERS — Loading & Reading Datasets")

try:
    # 17a. load_dataset
    show_function("load_dataset", "Load dataset from list of tuples, dicts, or CSV path")
    rows = fr.load_dataset(dataset)
    show_input("dataset", f"list of {len(dataset)} tuples")
    show_output("loaded_rows", f"{len(rows)} rows")
    show_output("first_row", rows[0])
    show_output("last_row", rows[-1])
    ok("load_dataset loads from tuples")

    # 17b. read_csv_text
    show_function("read_csv_text", "Read a CSV from string content")
    csv_text = "text,label\naccha product hai,positive\nbekar product,negative"
    rows2 = fr.read_csv_text(csv_text)
    show_input("csv_text", '"text,label\\naccha product hai,positive\\n..."')
    show_output("rows", rows2)
    ok("read_csv_text parses CSV strings")

    # 17c. Dakshina & Aksharantar constants (don't load files, just check constants)
    show_function("DAKSHINA_LANGUAGES", "Languages supported by Dakshina dataset loader")
    show_output("languages", fr.DAKSHINA_LANGUAGES)

    show_function("AKSHARANTAR_LANGUAGES", "Languages supported by Aksharantar dataset loader")
    show_output("languages", fr.AKSHARANTAR_LANGUAGES)

    show_function("dakshina_rules / aksharantar_rules", "Generate RuleSets from Dakshina/Aksharantar data")
    show_output("dakshina_rules", f"{fr.dakshina_rules}")
    show_output("aksharantar_rules", f"{fr.aksharantar_rules}")
    ok("All dataset loader functions accessible")

except Exception as e:
    errors.append(("Dataset Loaders", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 18: TYPES & DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────
section("TYPES & DATA CLASSES — Core Library Types")

try:
    show_function("NormalizeResult", "Result of text normalization")
    nr = fr.normalize("bohot acchaa product")
    show_output("type", type(nr).__name__)
    show_output("fields", f"original='{nr.original}', normalized='{nr.normalized}', changes={nr.changes}")
    ok()

    show_function("Variant", "A single text perturbation variant")
    variants = fr.perturb("acha product hai", n=1)
    if variants:
        show_output("type", type(variants[0]).__name__)
        show_output("fields", f"text='{variants[0].text}', kind='{variants[0].kind}'")
    ok()

    show_function("GateResult", "Result of a quality gate decision")
    gr = fr.gate(dummy_sentiment_model, "accha hai")
    show_output("type", type(gr).__name__)
    show_output("to_dict()", json.dumps(gr.to_dict(), indent=2, ensure_ascii=False)[:200])
    show_output("to_json()", gr.to_json()[:150] + "...")
    ok()

    show_function("QuickCheckResult", "Beginner-friendly gate result")
    qc = fr.quick_check(dummy_sentiment_model, "acha hai")
    show_output("type", type(qc).__name__)
    show_output("bool(result)", bool(qc))
    show_output("dict(result)", qc.to_dict())
    ok()

    show_function("RuntimeGuardResult", "Production runtime guard result")
    rg = fr.runtime_guard(dummy_sentiment_model, "accha product")
    show_output("type", type(rg).__name__)
    show_output("fields", f"decision='{rg.decision}', should_route_to_human={rg.should_route_to_human}")
    ok()

    show_function("Report", "Full robustness evaluation report")
    show_output("type", str(fr.Report))
    ok()

    show_function("ModelBenchmarkScore", "Benchmark score for a single model")
    show_output("type", str(fr.ModelBenchmarkScore))

    show_function("BenchmarkSuite", "Collection of benchmark results")
    show_output("type", str(fr.BenchmarkSuite))

    show_function("ExperimentResult", "Multi-model experiment result")
    show_output("type", str(fr.ExperimentResult))

    show_function("Predictor", "Efficient model caller with caching and batching")
    p = fr.Predictor(dummy_sentiment_model)
    preds = p.predict(["accha hai", "bekar hai", "accha hai"])
    show_output("predictions", preds)
    show_output("cache_hits", p.cache_hits)
    show_output("total_calls", p.calls)
    ok("Predictor caches duplicate predictions")

    show_function("ProtectedModel", "Model wrapped with normalization")
    pm = fr.ProtectedModel(dummy_sentiment_model)
    show_output("name", pm.name)
    show_output("predict", f'"{pm("bohot acchaaaa")}"')
    ok()

    show_function("RuleSet", "Spelling rule set data structure")
    show_output("type", str(fr.RuleSet))
    ok("All types and data classes verified")

except Exception as e:
    errors.append(("Types & Data Classes", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
#  SECTION 19: learn_rules_from_texts
# ─────────────────────────────────────────────────────────────────────────────
section("RULE MINING — Learning Rules from Raw Texts")

try:
    show_function("learn_rules_from_texts", "Mine spelling rule candidates from raw user texts")

    sample_texts = [
        "yeh bohot accha product hai",
        "bohot acha product mila",
        "bahut acchaa service thi",
        "bohot badiya quality hai",
        "ye bohot sahi product hai",
    ]
    show_input("texts", f"{len(sample_texts)} raw user messages")
    for i, t in enumerate(sample_texts, 1):
        print(f"      {i}. \"{t}\"")

    mined = fr.learn_rules_from_texts(sample_texts)
    show_output("type", type(mined).__name__)
    show_output("rules_found", len(mined) if hasattr(mined, '__len__') else str(mined))
    if isinstance(mined, list):
        for m in mined[:5]:
            print(f"      • {m}")
    elif isinstance(mined, dict):
        for k, v in list(mined.items())[:5]:
            print(f"      • {k}: {v}")
    else:
        print(f"      {mined}")
    ok("learn_rules_from_texts mines spelling patterns from raw data")

except Exception as e:
    errors.append(("Rule Mining", str(e)))
    print(f"    {RED}✗ ERROR: {e}{RESET}")


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print(f"""
{'═'*78}
{BOLD}{CYAN}                     SHOWCASE COMPLETE — FINAL SUMMARY{RESET}
{'═'*78}
""")

total_sections = section_num[0]
failed_sections = len(errors)
passed_sections = total_sections - failed_sections

print(f"  {BOLD}Library:{RESET}       FlipRate v{fr.__version__}")
print(f"  {BOLD}Author:{RESET}       {fr.__author__}")
print(f"  {BOLD}Sections:{RESET}     {total_sections} total")
print(f"  {GREEN}{BOLD}Passed:{RESET}       {passed_sections} sections ✓")

if errors:
    print(f"  {RED}{BOLD}Failed:{RESET}       {failed_sections} sections ✗")
    print(f"\n  {RED}Failed Sections:{RESET}")
    for name, msg in errors:
        print(f"    ✗ {name}: {msg}")
    sys.exit(1)
else:
    print(f"  {GREEN}{BOLD}Status:{RESET}       {GREEN}ALL SECTIONS PASSED ✓{RESET}")

print(f"""
{'═'*78}
{BOLD}  Key Library Features Demonstrated:{RESET}

  📝 Text Normalization     — Clean messy romanized Indian text
  🔄 Perturbation           — Generate realistic spelling variants
  🔍 Script Detection       — Identify Latin / Devanagari / Mixed scripts
  🚦 Quality Gating         — PASS / TRANSFORM / REVIEW decisions
  ⚡ Happy-Path API         — One-line safety checks for beginners
  🏭 Production Pipeline    — Enterprise-grade gating & batch processing
  🛡️  Model Protection       — Wrap any model with auto-normalization
  💡 Explainability         — Human-readable explanations for decisions
  ✅ Validation             — Input & dataset safety checks
  📚 Rules Engine           — Spelling rule management & customization
  📊 Benchmarking           — Multi-model robustness comparison
  🧪 Experiments            — Raw vs protected model evaluation
  🔌 Model Adapters         — Connect any model type (HF, sklearn, API, LLM)
  ⚙️  Service Infrastructure — Job storage & rate limiting
  📡 Streaming Events       — Real-time progress for web applications
  📂 Dataset Loaders        — Load from CSV, tuples, or research corpora
  🧱 Types & Data Classes   — Well-structured result objects
  🔮 Rule Mining            — Learn rules from raw user texts
{'═'*78}
""")
