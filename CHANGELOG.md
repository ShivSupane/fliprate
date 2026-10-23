# Changelog

## 0.4.0
- **Devanagari & Script-Aware Gating**: Native support for Devanagari (Hindi, Marathi, Nepali) and mixed-script text. Added `detect_script(text)` returning `"latin"`, `"devanagari"`, `"mixed"`, or `"other"`.
- **Script-Aware Gating Behavior**: `gate()` now evaluates safe orthographic variations for Devanagari and mixed text (`PASS` / `TRANSFORM` / `REVIEW`). Unsupported scripts (e.g. Tamil, Bengali, Telugu script) route to `REVIEW` with reason `"unsupported script: <name>"`. Added `script_aware=True` parameter to `gate()`, `iter_gate()`, and `agate()` (set to `False` to restore legacy Latin-only behavior).
- **Unicode Normalization & Zero-Width Cleaning**: `normalize()` enforces stable NFC encoding and strips invisible zero-width noise (`\u200B` ZWSP, `\u200C` ZWNJ, `\u200D` ZWJ, `\uFEFF` BOM).
- **New Perturbation Kinds for Devanagari**:
  - `"unicode"`: Evaluates precomposed vs decomposed nuktas (e.g. U+0958..U+095F vs base + U+093C).
  - `"digits"`: Bidirectional conversion between ASCII digits (`0-9`) and Devanagari digits (`०-९`).
  - `"punctuation"`: Danda (`।`) and double danda (`॥`) interchange with Western punctuation (`.`, `!`, `...`).
  - `"spelling"`: Curated nukta-safe equivalence groups (`ज़रा`/`जरा`, `फ़ोन`/`फोन`, etc.) avoiding blind substitutions.
- **Negation Safety in Devanagari**: Extended negation set with Hindi and Marathi negation words (`नहीं`, `नही`, `नहिं`, `मत`, `नाही`, `नाहीं`, `नको`, `नकोस`). Verified over corpus tests that negation counts remain strictly invariant across all perturbations.
- **Canonical Form Gate Pass**: In `gate()`, when input is already in canonical normalized form (`normalize(text).normalized == text`) and raw variants flip, it now evaluates normalized stability. If normalized variants are stable, returns `PASS` with warning `"input is already in canonical form; model is sensitive to other spellings (protect() recommended)"` instead of `REVIEW`.
- **Developer Toolkit & Beginner Happy Path**: Added `fr.quick_check()`, `fr.normalize_text()`, `fr.safe_predict()`, and `fr.check_model()` providing an intuitive 1-line surface for learners and developers.
- **Standardized `ModelAdapter` Interface**: Added `ModelAdapter` abstract base class with `.predict()`, `.predict_batch()`, `.metadata()`, and `.supported_languages()`, alongside `CallableAdapter`, `HttpServiceAdapter` (REST endpoints with timeouts & retries), and `as_adapter()`.
- **Production Pipeline Wrappers**: Added `fr.runtime_guard(mode="lenient"|"strict")`, `fr.batch_guard()`, `fr.evaluate_pipeline()`, and `fr.make_protected_model()`.
- **Rule Management & Diagnostics**: Added `fr.load_rules()`, `fr.add_custom_rule()`, `fr.preview_variants()`, and `fr.describe_rule_group()`.
- **Explainability & Debugging**: Added `fr.explain_gate_decision()`, `fr.show_why_it_flipped()`, `fr.preview_normalized_variants()`, and `fr.print_rule_hits()`.
- **Benchmarking Suite & Quality Dashboard**: Added `fr.benchmark()`, `fr.compare_models()`, `BenchmarkSuite`, and exportable Markdown and standalone HTML dashboards.
- **Distributed Infrastructure & Security**: Added `BaseJobStore` and `BaseRateLimiter` interfaces with pluggable `RedisJobStore` and `RedisRateLimiter`, `X-Request-ID` correlation middleware, structured request logging, and production CORS checks.
- **Cookbook & Guides**: Added practical real-world recipes in `docs/COOKBOOK.md` and enterprise architecture guidelines in `docs/DEPLOYMENT_GUIDE.md`.
- **Opt-in Cross-Script Transliteration**: Optional `cross_script=True` parameter in `perturb()` and `is_safe_variant()` for deterministic 1-to-1 transliterated word pairs.

## 0.3.0
- **Enhanced `load_dataset` / `read_csv_text`**: Auto-detect delimiter (comma, semicolon, tab) via `csv.Sniffer` with fallback, support custom column mappings (`text_col`, `label_col`), whitespace-stripped headers, and `utf-8-sig` encoding.
- **Serialization on `GateResult`**: Added `.to_dict()` and `.to_json()` methods serializing `decision`, `prediction`, `reason`, `raw_stability`, `normalized_stability`, `n_variants`, `warnings`, and `flipped_variants`.
- **Async Gate API (`fr.agate`)**: Native `async def agate(...)` running gate checks in worker threads via `asyncio.to_thread` for non-blocking integration in asynchronous apps.
- **CLI Report Command**: Added `fliprate report --data <file.csv> --model <spec> --out <report.md> [--rules <rules.csv>]` for Markdown report generation directly from the command line.

## 0.2.2
- **URL & Entity Preservation in `normalize()`**: URLs (`http://`, `https://`, `www.`), emails, `@handles`, `#hashtags`, and numbers are protected with internal masks and restored byte-for-byte.
- **Emoji Support in `gate()`**: Emoji and unicode modifiers are preserved in text and ignored when evaluating script Latin ratio, preventing false REVIEW decisions.
- **Argument & Type Validation**: Strict validation for `threshold` in `(0, 1]`, `n_variants >= 1`, `min_variants >= 1`, `bootstrap >= 1`, and `on_error` options across `gate`, `iter_gate`, `perturb`, and `test`. Clear `TypeError` on non-string inputs and `ValueError` on malformed dataset rows.
- **Graceful Model Error Handling in `gate()`**: Added `on_error="raise" | "review"` parameter to `gate()` and `iter_gate()`, allowing model exceptions to produce a `REVIEW` decision instead of crashing.
- **Documentation Corrections**: Updated docs to reflect actual `Report` methods (`summary`, `to_dict`, `to_markdown`, `save`, `load`), `GateResult.flipped_variants`, `read_csv_text` taking CSV text, Aksharantar 20 languages, and `LANGUAGE_GROUPS` 12 languages + shared group.

## 0.2.1
- **Multi-language rules**: Added `LANGUAGE_GROUPS` dict covering 12 Indian languages plus a shared group with HIGH-confidence romanized spelling groups for Hindi, Urdu, Marathi, Bengali, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Odia, Assamese, and shared.
- **Dakshina integration** (`fliprate.dakshina`): Load 225K+ human-verified spelling variant groups from the Google Dakshina dataset v1.0 (CC BY-SA 4.0). 12 languages, from Wikipedia text annotated by native speakers.
  - `fr.dakshina_rules("ta")` — Tamil rules from Dakshina
  - `fr.load_dakshina_groups("hi", "mr")` — raw groups for inspection
- **Aksharantar integration** (`fliprate.aksharantar`): Load 428K+ spelling variant groups from AI4Bharat Aksharantar dataset (CC BY 4.0). 20 Indian languages, 24M+ word pairs from IndicCorp, Samanantar, Dakshina, and other corpora.
  - `fr.aksharantar_rules("tam", "tel", "kan")` — multi-language rules
  - `fr.load_aksharantar_groups("hin")` — raw groups
- **UI**: Replaced Streamlit console with built-in zero-dependency web console at `/`.
- **Normalization**: Improved run-collapse (consonants 3+→1, vowels 3+→2). Punctuation runs tracked in `changes`.
- **pyproject.toml**: Removed Streamlit dependency.
- **Scripts**: `scripts/build_rules_from_dakshina.py` and `scripts/build_rules_from_aksharantar.py` for regenerating groups from source data.

## 0.2.0 (alpha)
- Real-model support: batched Hugging Face adapter, LLM adapter, scikit-learn adapter, retries, caching.
- `fr.protect()`, paired bootstrap comparison (`fr.paired_difference`), reproducible reports with metadata.
- Reviewable rules (`RuleSet`), variant miner, dataset checks and leak-free split, human audit tooling.
- Gate now prefers safe normalization when it stabilizes the answer.
- Hardened API: API key, rate limit, concurrency cap, `/ready`, `/metrics`, privacy-safe logging.

## 0.1.0
- First version: perturbation, normalizer, robustness test, gate, live console.
