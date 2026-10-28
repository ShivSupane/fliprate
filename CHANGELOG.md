# Changelog

## 1.0.1 — Ergonomics & Extended Showcase

- **Ergonomics & Convenience Properties**:
  - `Report`: Added `.confidence_interval` (alias for `flip_rate_ci`) and `.n_flips` (`len(flips)`).
  - `Decision`: Added `.flip_rate` property (`1.0 - raw_stability`).
  - `RuntimeGuardResult`: Added `.flip_rate` property.
  - `BenchmarkSuite`: Added `.baseline` property (alias for `baseline_name`).
  - `NormalizeResult`: Added `.changed` property (`bool(changes)`).
- **Pass-through Parameters**:
  - `normalize_text()` now accepts `protected_terms` parameter and forwards to `normalize()`.
  - `make_protected_model()` now accepts `protected_terms` parameter.
  - `ProtectedModel` and `protect()` now accept `protected_terms`.
- **Showcase & Living Reference**:
  - Complete, executable `scratch/fliprate_showcase.py` covering all 23 API modules end-to-end with deep explanations (WHAT, WHY, HOW, WHEN) and ASCII-safe Windows terminal support.
- All 414 test cases pass across Python 3.9 → 3.13.

## 1.0.0 — Stable release: API frozen, contract supported

This is the first production-stable release of FlipRate. The public API is now frozen;
future releases will maintain backward compatibility.

- **API freeze**: all functions, classes, and CLI commands documented in `README.md` and
  `docs/API_REFERENCE.md` are guaranteed stable. No removals or breaking signature changes
  without a major version bump.
- **All 414 tests pass** across Python 3.9 → 3.13 on Linux.
- **README completely rewritten** for developers and AI agents: quick-start, concept explanations,
  plug-in-any-model guide, CLI reference, CI/CD integration, project layout, and full API table.
- **`pyproject.toml` classifier** updated from Alpha → `Production/Stable`.
- **Version bump**: `0.9.0` → `1.0.0` in `pyproject.toml` and `src/fliprate/__init__.py`.
- No new runtime dependencies introduced.

## 0.9.0 — Evidence, Documentation, Privacy and Release Readiness
- **Real Benchmark Protocol (`audit_dataset` & `run_benchmark_protocol`)**:
  - Added `audit_dataset(dataset, train_dataset=None, rules=None)` to programmatically audit evaluation datasets: class balance, script distribution (Latin, Devanagari, mixed), exact/near-duplicate pairs, label conflicts, and train/test leakage detection — all in one call.
  - Added `run_benchmark_protocol(models, dataset, ...)` convenience helper: runs a full reproducible comparison (raw vs protected) with `audit=True` by default, 95% Wilson confidence intervals, per-kind flip-rate breakdown, runtime/call accounting, and leakage reporting.
  - Both functions exported from top-level `fliprate` and in `__all__`.
- **CLI Hardening & Exit Codes**:
  - `fliprate benchmark` now accepts `--train-data` flag to supply a training CSV for automatic leakage auditing during benchmarks.
  - `fliprate report` and `fliprate benchmark` now return `2` on model/data loading errors (config or data errors) vs `1` for threshold violations — consistent with documented exit codes.
- **Security Policy (`SECURITY.md`)**:
  - Added `SECURITY.md` with supported versions table (0.6.x–0.9.x), private vulnerability disclosure process (email `shiv.supane777@gmail.com`), response timeline (48h acknowledgement, 5-day triage), and `Privacy & Security Best Practices` for FlipRate deployments.
- **Privacy Preservation Verified**:
  - Confirmed `explain_report_flips(report, opt_in_examples=False)` (the default) never surfaces raw original or variant texts in report summaries.
  - New regression test `test_privacy_preservation_by_default` verifies text containing sensitive tokens does not appear in report summaries unless `opt_in_examples=True` is explicitly set.
- **Packaging & Type Stubs**:
  - `py.typed` marker continues to be shipped in the package. Zero new required runtime dependencies.
  - Version bumped to `0.9.0` in both `pyproject.toml` and `src/fliprate/__init__.py`.
- **Version-Agnostic Test Hygiene**:
  - `test_report_version_is_v080` in `test_v08_governance_and_diagnosis.py` updated to compare `rep.metadata["fliprate_version"]` against `fliprate.__version__` rather than a hardcoded string — stays correct across future releases.

## 0.8.0 — Better Rule Governance and Diagnosis
- **Protected Terms (`protected_terms`)**:
  - Allow developers and production pipelines to specify words, phrases, brand names, product titles, account codes, or domain terms that normalization and perturbation must leave unchanged byte-for-byte.
  - Deterministic longest-phrase-first boundary matching prevents multiword phrases (e.g. `"Tata Motors"`) from colliding with shorter sub-terms (e.g. `"Tata"`).
  - Explicit word boundary lookahead/lookbehind matching (`(?<!\w)...(?!\w)`) handles substrings, punctuation boundaries, Latin, and Devanagari mixed scripts safely without corrupting internal characters.
  - Integrated into `normalize()`, `perturb()`, `gate()`, `test()`, `FlipRateNormalizer`, and `FlipRate` engine.
- **Reviewed Rule-Mining Workflow (`CandidateRuleSet` & Enhanced `Candidate`)**:
  - `learn_rules_from_texts()` and `mine_variants()` produce a `CandidateRuleSet` containing candidate rules awaiting human review rather than silently activating unverified rules.
  - Enhanced `Candidate` dataclass with governance fields: `canonical`, `variants`, `counts`, `risk`, `language`, `confidence` (heuristic ranking score), `reviewer_status` (`"pending"`, `"approved"`, `"rejected"`), `evidence_source`, and `notes`.
  - `CandidateRuleSet` provides `.approve()`, `.reject()`, `.save_reviewed_csv()` (fully compatible with `RuleSet.from_csv()`), and `.to_ruleset(approved_only=True)`.
  - Subclasses `list` for 100% backward compatibility with existing code.
- **Explainability & Non-Causal Diagnosis**:
  - Enhanced `show_why_it_flipped()`: Uses sequence matching (`difflib`) to accurately compute `changed_tokens` / `diffs` across deletions, insertions, and word substitutions even when token counts vary.
  - Generates objective, non-causal descriptions describing observed input/output shifts without claiming internal model reasoning causation.
  - Added `explain_report_flips(report, opt_in_examples=False)`: Provides privacy-preserving analysis of dataset test reports, keeping raw input/variant text opt-in.
- **Script Coverage & Documented Abstention**:
  - Validated that unsupported Indic scripts (Tamil, Telugu, Bengali, etc.) continue to yield documented `REVIEW` decisions with explicit `reason="unsupported script: <name>"`, avoiding unsafe or uncontrolled normalization.
  - Updated report schema version to `"0.8.0"`.

## 0.7.0 — ML Workflow Integration and Experimental Augmentation
- **Scikit-Learn Transformer (`FlipRateNormalizer`)**:
  - `BaseEstimator` and `TransformerMixin` compatible text transformer for scikit-learn `Pipeline`.
  - Works with `Pipeline([('norm', FlipRateNormalizer()), ('tfidf', TfidfVectorizer()), ('clf', LogisticRegression())])`.
  - Fully compatible with `sklearn.base.clone()`.
  - Preserves row order and row count; zero dependencies required in core if scikit-learn is not installed.
- **Training-Time Data Augmentation (`fr.augment`)**:
  - Generates controlled training examples using reviewed meaning-preserving transformations.
  - Strict safety contract: preserves labels only for transformations passing semantic safety checks.
  - Strictly invariant negation checks: never flips or deletes negative particles (`nahi`, `nhi`, `not`, etc.).
  - Preserves original dataset examples by default; supports deterministic seeded generation.
  - Provides detailed diagnostics: transformation kind breakdown and count of skipped unsafe candidates.
  - Added empirical validation helper `fr.compare_augmentation()` comparing baseline vs augmented accuracy, macro-F1, and flip rate.
- **Dataset Ergonomics & Explainable Diagnostics**:
  - Duck-typed support for pandas DataFrames in `load_dataset()` without importing pandas in the core path.
  - `find_near_duplicates()`: Identifies exact duplicates, near-duplicate spelling variants, and conflicting labels with 1-indexed row references without silently dropping data.
  - `check_leakage()`: Cross-split diagnostic detecting exact and spelling-variant train/test leakage with explainable row references.
- **Engine Integration**: Added `fr.FlipRate.augment()` to the convenient engine interface.

## 0.6.0 — Cost-Aware Evaluation and Measured Performance
- **Three Explicit Modes (`fast`, `balanced`, `thorough`)**:
  - `fast`: Executes standard model inference on original inputs without generating or predicting safe variants (`n_variants_evaluated = 0`). Explicitly states that robustness was not evaluated, avoiding hidden compute costs.
  - `balanced`: Evaluates a deterministic, seeded sample of reviewed transformations (`sampling_policy="deterministic_sample"`), providing cost-effective regression testing.
  - `thorough`: Full robustness evaluation across all configured transformations and variants.
- **Sound Early Stopping & Mathematical Bound Checks**:
  - In `gate(..., early_stop=True)`: Halts immediately upon observing the first prediction flip or when threshold achievement is mathematically impossible. Marks `evaluation_complete=False` and `stop_reason="first_flip_detected"` / `"threshold_unachievable"`, preventing early-stopped runs from being misrepresented as exact flip rates.
  - In `test(..., early_stop=True)`: Halts dataset evaluation early if threshold breach is mathematically guaranteed.
- **Granular Cost & Inference Accounting in `Predictor`**:
  - Distinguishes individual input evaluations (`calls`) from batch/HTTP requests (`request_count`).
  - Added cache tracking: `total_lookups`, `cache_hits`, and `cache_hit_rate` property.
  - Added cache invalidation and counter reset: `invalidate_cache()`, `reset_stats()`.
- **Reproducible Report Schema (`schema_version = "0.6.0"`)**:
  - Expanded `Report` and `Report.to_dict()` with `schema_version`, `mode`, `evaluation_complete`, `n_variants_requested`, `n_variants_evaluated`, `stop_reason`, and detailed performance metadata (`model_calls`, `request_count`, `cache_hits`, `cache_hit_rate`, `seconds`).
- **CLI Options**: Added `--mode {fast,balanced,thorough}` and `--early-stop` flags to `fliprate test` and `fliprate gate`.
- **Engine & CI Assertion Helpers**: `fr.FlipRate` and `fr.assert_stable` now seamlessly support `mode` and `early_stop`.

## 0.5.0
- **Canonical Decision Type**: Added unified `Decision` dataclass for single-message gate decisions with `.is_safe`, `.should_route_to_human`, mapping access, and JSON serialization. `GateResult` remains fully supported as a backwards-compatible alias.
- **Model-Bound FlipRate Engine**: Added `fr.FlipRate(model, ...)` configuration object for binding a model, stability thresholds, variant count, and rules into a single reusable object.
- **CI Regression Assertion Helper**: Added `fr.assert_stable(model, dataset, ...)` supporting explicit requirements (`max_flip_rate`, `min_accuracy`, `max_accuracy_drop`) with actionable error messages.
- **CLI Regression Gate**: Extended `fliprate test` with `--max-flip-rate`, `--min-accuracy`, and `--max-accuracy-drop` flags. Exits `0` on success, `1` on threshold breach, and `2` on data/model error. Stores CI results in report metadata when `--save` is used.
- **Exhaustive Test Coverage**: Added contract test suite (`tests/test_v05_contracts.py`) bringing automated test count to 365 tests with 100% pass rate.
- **Benchmarking CLI**: Added `fliprate benchmark --models <spec> --data <csv> --out <path>` supporting standalone HTML scorecards and Markdown reports.
- **Explainability CLI**: Added `fliprate explain --model <spec> --text <text>` for interactive diagnostic debugging.
- **Showcase & Validation Hardening**: Eliminated silent `pip install` in showcase scripts, added UTF-8 console support, and removed false-pass risks in validation error checks.

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
