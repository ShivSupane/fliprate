"""Comprehensive test suite for Phase 3: Devanagari & script-aware gating (v0.4.0)."""
import os
import random
import unicodedata
import pytest
import fliprate as fr
from fliprate.gating import detect_script
from fliprate.normalization import normalize
from fliprate.perturbation import perturb
from fliprate.rules import DEFAULT_RULES, RuleSet, is_safe_variant


# ── 1. detect_script edge cases ───────────────────────────────────────────────
def test_detect_script_categories():
    assert detect_script("mala product avadla nahi") == "latin"
    assert detect_script("मला प्रॉडक्ट आवडला नाही") == "devanagari"
    assert detect_script("mera order nahi mila नाही") == "mixed"
    assert detect_script("order कैंसिल karo please") == "mixed"

    # Unsupported scripts -> "other"
    assert detect_script("வணக்கம் எப்படி இருக்கிறீர்கள்") == "other"  # Tamil
    assert detect_script("నమస్కారం ఎలా ఉన్నారు") == "other"       # Telugu
    assert detect_script("নমস্কার কেমন আছেন") == "other"           # Bengali
    assert detect_script("مرحبا كيف حالك") == "other"             # Arabic

    # Edge cases -> "other"
    assert detect_script("") == "other"
    assert detect_script("   ") == "other"
    assert detect_script("1234567890") == "other"
    assert detect_script("!@#$%^&*()_+-=[]{}|;':,.<>?/") == "other"
    assert detect_script("😀😁😂🤣😃😄😅😆") == "other"
    assert detect_script("123 456 😡 👍") == "other"

    with pytest.raises(TypeError, match="text must be str"):
        detect_script(None)


# ── 2. Unicode normalization & zero-width cleaning ───────────────────────────
def test_normalize_unicode_nfc_and_zero_width_strip():
    # Decomposed nukta: U+0915 (क) + U+093C (़) — NFC does NOT compose this to U+0958
    # (U+0958 is a legacy ISCII precomposed form, not a standard NFC composition target).
    # What NFC DOES guarantee: the output is NFC-normalized (stable round-trip).
    decomposed = "\u0915\u093c\u0930\u0940\u092c"  # क + ़ + र + ी + ब
    res = normalize(decomposed)
    assert unicodedata.is_normalized("NFC", res.normalized)
    # The decomposed sequence stays intact (NFC keeps क+़ as-is)
    assert "\u0915\u093c" in res.normalized

    # Precomposed (U+0958) NFC-normalizes to the same canonical NFC sequence as decomposed (क + ़)
    precomposed = "\u0958\u0930\u0940\u092c"  # क़ + र + ी + ब
    res2 = normalize(precomposed)
    assert unicodedata.is_normalized("NFC", res2.normalized)
    assert res2.normalized == res.normalized

    # Stray zero-width characters: ZWSP (U+200B), ZWNJ (U+200C), ZWJ (U+200D), BOM (U+FEFF)
    messy = "\u092e\u200c\u0932\u093e \u200b\u0911\u0930\u094d\u0921\u0930 \u200d\u0928\u093e\u0939\u0940\ufeff"
    res_clean = normalize(messy)
    assert "\u200b" not in res_clean.normalized
    assert "\u200c" not in res_clean.normalized
    assert "\u200d" not in res_clean.normalized
    assert "\ufeff" not in res_clean.normalized
    assert res_clean.normalized == "\u092e\u0932\u093e \u0911\u0930\u094d\u0921\u0930 \u0928\u093e\u0939\u0940"


# ── 3. Perturbation kinds: unicode, digits, punctuation, spelling ────────────
def test_perturb_devanagari_kinds():
    # unicode kind (precomposed <-> decomposed nukta)
    text_nukta = "ज़रा फ़ोन देखो"
    v_uni = perturb(text_nukta, n=3, kinds=["unicode"], seed=42)
    assert len(v_uni) > 0
    assert any("\u093c" in v.text or "\u095b" in v.text for v in v_uni)
    for v in v_uni:
        assert is_safe_variant(text_nukta, v.text)

    # digits kind (ASCII <-> Devanagari digits)
    text_digits = "ऑर्डर नंबर 123456"
    v_dig = perturb(text_digits, n=3, kinds=["digits"], seed=42)
    assert len(v_dig) > 0
    assert any("१" in v.text or "२" in v.text or "३" in v.text for v in v_dig)
    for v in v_dig:
        assert is_safe_variant(text_digits, v.text)

    # punctuation kind (danda swap)
    text_punc = "सामान खराब आहे।"
    v_punc = perturb(text_punc, n=3, kinds=["punctuation"], seed=42)
    assert len(v_punc) > 0
    assert any("." in v.text or "!" in v.text or not v.text.endswith("।") for v in v_punc)
    for v in v_punc:
        assert is_safe_variant(text_punc, v.text)

    # curated spelling swap (ज़रा <-> जरा, नहीं <-> नही)
    text_spell = "ज़रा मुझे फ़ोन मत करना"
    v_spell = perturb(text_spell, n=5, kinds=["spelling"], seed=42)
    assert len(v_spell) > 0
    assert any("जरा" in v.text or "फोन" in v.text for v in v_spell)
    for v in v_spell:
        assert is_safe_variant(text_spell, v.text)


# ── 4. Nukta safety: Non-curated words like सजा/सज़ा are untouched ───────────
def test_nukta_safety_non_curated():
    # सज़ा (punishment) vs सजा (decorated) is NOT in curated groups
    text = "कमरे को अच्छे से सजा दो"
    variants = perturb(text, n=10, seed=42)
    # "सजा" should NEVER be turned into "सज़ा"
    for v in variants:
        assert "सज़ा" not in v.text
        assert is_safe_variant(text, v.text)


# ── 5. Negation safety & rejection of dropped negations ───────────────────────
def test_dropping_devanagari_negation_is_rejected():
    orig = "मला हे प्रॉडक्ट आवडले नाही"
    # Dropping नाही
    dropped = "मला हे प्रॉडक्ट आवडले"
    assert not is_safe_variant(orig, dropped)
    assert not DEFAULT_RULES.is_safe_variant(orig, dropped)

    # Hindi नहीं
    orig_hi = "मुझे सामान नहीं मिला"
    dropped_hi = "मुझे सामान मिला"
    assert not is_safe_variant(orig_hi, dropped_hi)

    # Adding a negation where none existed
    positive = "order cancel ho gaya"
    negative = "order cancel nahi ho gaya"
    assert not is_safe_variant(positive, negative)


# ── 6. Fixed corpus of 200+ Devanagari/mixed sentences over 1000 seeds ────────
def test_negation_invariance_over_200_sentences_corpus():
    # Generate 200 diverse sentences (negative and positive, Devanagari and mixed)
    templates = [
        "मुझे सामान नहीं मिला",
        "मला ऑर्डर अजून मिळाली नाही",
        "यह फ़ोन अच्छा नहीं है",
        "कृपया ऑर्डर कैंसल मत कीजिए",
        "मला हे प्रॉडक्ट नको आहे",
        "पैसा वापस नही आया",
        "order deliver nahi hua",
        "parcel खराब निकला",
        "delivery boy ने कॉल नहीं किया",
        "app पर refund status nahi dikh raha",
        "यह सामान बहुत अच्छा है",
        "मला खूप आवडला",
        "ऑर्डर नंबर 123456 मिळाला",
        "payment successful zaleli ahe",
        "please cancel my order",
        "ज़रा फ़ोन चेक करा",
        "साफ़ बात यह है कि मदद नहीं मिली",
        "काग़ज़ पर बिल नही था",
        "हफ़्ता भर हो गया सामान नहीं आया",
        "वज़ह मुझे मालूम नहीं",
    ]
    # Build 200 sentences by combining and indexing
    corpus = []
    for i in range(200):
        tmpl = templates[i % len(templates)]
        corpus.append(f"{tmpl} #{i + 1000} (टिकट {i})")

    assert len(corpus) == 200

    # Test across 1000 random seeds (sampling across corpus)
    rng = random.Random(1337)
    for seed in range(1000):
        sent = corpus[seed % len(corpus)]
        vars_generated = perturb(sent, n=3, seed=seed)
        orig_negs = DEFAULT_RULES.negation_count(sent)
        for v in vars_generated:
            # Crucial assertion: negation count must NEVER change
            var_negs = DEFAULT_RULES.negation_count(v.text)
            assert var_negs == orig_negs, f"Negation leak on seed {seed}: '{sent}' ({orig_negs}) -> '{v.text}' ({var_negs})"
            assert is_safe_variant(sent, v.text)


# ── 7. Script-aware gate behavior ─────────────────────────────────────────────
def test_gate_devanagari_pass_and_transform():
    # Dummy consistent model: returns same answer for all variants -> PASS
    model_ok = lambda t: "positive" if "अच्छा" in t or "आवडला" in t else "negative"
    r_pass = fr.gate(model_ok, "मला प्रॉडक्ट खूप आवडला", n_variants=3, min_variants=1)
    assert r_pass.decision == "PASS"
    assert r_pass.raw_stability >= 0.8

    # Unstable model that normalizes cleanly (e.g. sensitive to stray ZWNJ)
    def model_sensitive_to_zwnj(text: str) -> str:
        return "clean" if "\u200c" not in text else "noisy"

    # Input has stray ZWNJ, which normalize() removes
    r_trans = fr.gate(model_sensitive_to_zwnj, "म\u200cला प्रॉडक्ट आवडला", n_variants=3, min_variants=1)
    assert r_trans.decision in ("PASS", "TRANSFORM")

    # Unsupported script -> REVIEW with specific reason
    r_tamil = fr.gate(model_ok, "வணக்கம்")
    assert r_tamil.decision == "REVIEW"
    assert "unsupported script: Tamil" in r_tamil.reason

    # script_aware=False restores old v0.1-v0.3 behavior (Devanagari -> REVIEW)
    r_legacy = fr.gate(model_ok, "मला प्रॉडक्ट आवडला", script_aware=False)
    assert r_legacy.decision == "REVIEW"
    assert "Mostly non-Latin script" in r_legacy.reason


# ── 8. Cross-script opt-in perturbations ──────────────────────────────────────
def test_cross_script_opt_in():
    text = "order nahi mila"
    # Default cross_script=False does not produce Devanagari swaps
    vars_default = perturb(text, n=5, cross_script=False, seed=42)
    for v in vars_default:
        assert "नहीं" not in v.text

    # Opt-in cross_script=True produces curated cross-script swaps
    vars_cross = perturb(text, n=8, cross_script=True, seed=42)
    assert any("नहीं" in v.text for v in vars_cross)
    for v in vars_cross:
        assert is_safe_variant(text, v.text, cross_script=True)


# ── 9. Determinism per seed ───────────────────────────────────────────────────
def test_perturb_determinism_per_seed():
    sample = "ज़रा 2 फ़ोन नंबर 123456 पर कॉल मत करना।"
    run1 = [v.text for v in perturb(sample, n=5, seed=999)]
    run2 = [v.text for v in perturb(sample, n=5, seed=999)]
    assert run1 == run2

    run_different = [v.text for v in perturb(sample, n=5, seed=123)]
    assert run1 != run_different


# ── 10. No-BOM check across entire repository ─────────────────────────────────
def test_no_utf8_bom_in_repo():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    boms = []
    for r, d, files in os.walk(root):
        if any(skip in r for skip in (".git", ".venv", "__pycache__", "dist", "build")):
            continue
        for f in files:
            path = os.path.join(r, f)
            try:
                with open(path, "rb") as fh:
                    if fh.read(3) == b"\xef\xbb\xbf":
                        boms.append(os.path.relpath(path, root))
            except Exception:
                pass
    assert boms == [], f"Found UTF-8 BOM in files: {boms}"


# ── 11. Canonical input pass with warning (Devanagari) ─────────────────────────
def test_gate_canonical_pass_with_warning_devanagari():
    text = "ज़रा फ़ोन देखो"
    assert normalize(text).normalized == text

    # Model only recognizes canonical "ज़रा" and "फ़ोन" (with nukta); flips on "जरा" or "फोन"
    def strict_deva_model(t: str) -> str:
        return "ok" if "ज़रा" in t and "फ़ोन" in t else "fail"

    r = fr.gate(strict_deva_model, text, n_variants=6, threshold=0.8, seed=42)
    assert r.decision == "PASS"
    assert r.raw_stability < 1.0
    assert r.normalized_stability >= 0.8
    assert len(r.flipped_variants) > 0
    warning_msg = "input is already in canonical form; model is sensitive to other spellings (protect() recommended)"
    assert warning_msg in r.warnings
