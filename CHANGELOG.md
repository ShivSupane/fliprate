# Changelog

## 1.0.3 — Release pipeline cleanup and final publish version

- Bumped the package version to `1.0.3` so the next GitHub Release can publish cleanly without colliding with the already-uploaded `1.0.2` PyPI release.
- Kept the repository's current CI/CD setup aligned with the package metadata and trusted-publishing release flow.
- Finalized the repo state for a clean GitHub release tag matching the package version exactly.
- The Colab notebook, docs refresh, and latest showcase additions remain included in this final release version.

## 1.0.2 — Colab Showcase Notebook & Docs Refresh

- **Google Colab notebook** (`examples/fliprate_colab_showcase.ipynb`): a browser-runnable
  twin of the local showcase. It downloads the **real Dakshina + Aksharantar**
  transliteration datasets from their official sources (Google Storage + Hugging Face),
  runs a **live Hugging Face** multilingual sentiment model, and classifies text through
  **live LLM APIs** (Google Gemini / OpenAI) via Colab Secrets — then fixes flips with
  `fr.protect()`. All live sections degrade gracefully to a local fallback when a key or
  download is missing, so the notebook always runs end-to-end. The downloaded dataset
  rules are wired straight into `gate()`, `test()`, and the reproducible benchmark protocol.
- **Docs refresh**: README gains a "Showcase & Colab notebook" section with an
  Open-in-Colab badge and an updated project layout; `docs/QUICKSTART.md` and
  `docs/PROJECT_OVERVIEW.md` link the Colab notebook (and the stale PROJECT_OVERVIEW
  version/status header is corrected).
- No core library code changed. The public API remains frozen at 1.0.0.
- All 414 tests pass across Python 3.9 → 3.13.

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
  - Added `audit_dataset(dataset, train_dataset=None, rules=None)` to programmatically audit evaluation datasets: class balance, script distribution (Latin, Devanagari, mixed), exact/near-duplicate checks, and hidden leakage diagnostics.
  - Added `run_benchmark_protocol(models, dataset, ...)` convenience helper: runs a full reproducible comparison (raw vs protected) with `audit=True` by default, 95% Wilson confidence intervals, and exportable reports.
  - Both functions exported from top-level `fliprate` and in `__all__`.
- **CLI Hardening & Exit Codes**:
  - `fliprate benchmark` now accepts `--train-data` flag to supply a training CSV for automatic leakage auditing during benchmarks.
  - `fliprate report` and `fliprate benchmark` now return `2` on model/data loading errors (config or data errors) vs `1` for threshold violations — consistent with documented exit codes.
- **Security Policy (`SECURITY.md`)**:
  - Added `SECURITY.md` with supported versions table, private vulnerability disclosure process, and response timeline.
- **Privacy Preservation Verified**:
  - Confirmed `explain_report_flips(report, opt_in_examples=False)` (the default) never surfaces raw original or variant texts in report summaries.
  - New regression test verifies text containing sensitive tokens does not appear in report summaries unless `opt_in_examples=True` is explicitly set.
- **Packaging & Type Stubs**:
  - `py.typed` marker continues to be shipped in the package. Zero new required runtime dependencies.
  - Version bumped to `0.9.0` in both `pyproject.toml` and `src/fliprate/__init__.py`.

## 0.8.0 — Better Rule Governance and Diagnosis
- **Protected Terms (`protected_terms`)**:
  - Allow developers and production pipelines to specify words, phrases, brand names, product titles, account codes, or domain terms that normalization and perturbation must leave unchanged byte-for-byte.
- **Reviewed Rule-Mining Workflow (`CandidateRuleSet` & Enhanced `Candidate`)**:
  - `learn_rules_from_texts()` and `mine_variants()` produce a `CandidateRuleSet` containing candidate rules awaiting human review rather than silently activating unverified rules.
- **Explainability & Non-Causal Diagnosis**:
  - Enhanced `show_why_it_flipped()` and `explain_report_flips(report, opt_in_examples=False)`.
- **Script Coverage & Documented Abstention**:
  - Unsupported Indic scripts yield documented `REVIEW` decisions with explicit reasons.

## 0.7.0 — ML Workflow Integration and Experimental Augmentation
- **Scikit-Learn Transformer (`FlipRateNormalizer`)**:
  - Added `BaseEstimator` and `TransformerMixin` compatibility for scikit-learn pipelines.
- **Training-Time Data Augmentation (`fr.augment`)**:
  - Generates controlled training examples with semantic safety checks.
- **Dataset Ergonomics & Explainable Diagnostics**:
  - Added duplicate and leakage detection helpers.
- **Engine Integration**: Added `fr.FlipRate.augment()`.

## 0.6.0 — Cost-Aware Evaluation and Measured Performance
- **Three explicit modes (`fast`, `balanced`, `thorough`)**.
- **Early stopping and mathematical guardrails** for gate and test evaluation.
- **Granular inference accounting** in `Predictor`.
- **Reproducible report schema** with `schema_version` and detailed metrics.

## 0.5.0
- **Canonical Decision Type** and model-bound `FlipRate` engine.
- **CI regression assertion helper**: `fr.assert_stable`.
- **Benchmarking & explanation CLI**.

## 0.4.0
- **Devanagari & script-aware gating**.
- **Unicode normalization & zero-width cleanup**.
- **Production pipeline wrappers** and rule management tools.

## 0.3.0
- **Enhanced dataset loading** and CLI report command.
- **Async gate API** and serialization improvements.

## 0.2.2
- **URL & entity preservation**, emoji handling, argument validation, and graceful model error handling.

## 0.2.1
- **Multi-language rules** and Dakshina/Aksharantar integrations.
- **UI** and normalization improvements.

## 0.2.0 (alpha)
- Real-model support: Hugging Face, LLM, scikit-learn, retries, caching.
- `fr.protect()`, paired bootstrap comparison, reproducible reports.

## 0.1.0
- First version: perturbation, normalizer, robustness test, gate, live console.
