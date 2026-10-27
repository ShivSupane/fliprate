# FlipRate — Master Product and Engineering Roadmap

**Roadmap date:** 24 October 2026  
**Current published baseline:** FlipRate 0.4.0  
**North-star:** Make FlipRate the dependable, low-friction robustness regression gate for messy, code-switched and romanized Indian-language NLP.

This document is the single source of direction for planning the next FlipRate versions. It is a proposed engineering roadmap, not a claim that planned features already exist. Complete a phase only when its acceptance criteria pass; do not release merely because a version number has been reached.

---

## 1. Product definition

### The problem FlipRate should own

NLP predictions can change when the same intended message is typed with spelling variation, punctuation changes, transliteration variation, script mixing, or other meaning-preserving surface changes. This is particularly relevant to romanized and code-switched Indian-language text, such as Hindi/Marathi mixed with English.

FlipRate should help a developer:

1. **Measure** prediction instability and task quality under reviewed transformations.
2. **Explain** which input changes caused observed prediction changes.
3. **Protect** runtime inference with explicit, inspectable normalization and routing decisions.
4. **Prevent regressions** by comparing runs and failing a test or CI job when defined requirements are breached.
5. **Control evaluation cost** by making the number and type of extra model evaluations explicit.

### Positioning

> **FlipRate is a robustness regression gate for messy Indian-language NLP.** It tests whether predictions survive meaning-preserving text variation, explains observed failures, and helps prevent regressions before deployment.

Do not claim that FlipRate makes any model accurate, guarantees semantic equivalence, or fixes all Indian languages. Claims must follow tested coverage and published evidence.

### What makes it distinctive

- A clear focus on romanized Hindi/Marathi, code-switched text, and carefully reviewed language-specific transformations.
- A simple Python API plus CI-friendly machine-readable reports and exit codes.
- Transparent distinction between **prediction stability**, **task correctness**, and **normalization effects**.
- Cost-aware evaluation with explicit completeness and inference accounting.

General NLP evaluation, adapters and dashboards are supporting features, not the central product identity.

---

## 2. Verified baseline and important caveats

The PyPI listing currently presents **0.4.0** as the published release and documents a zero-dependency core, optional integrations, a CLI, normalization, perturbation, gating, evaluation, and model protection. The package metadata says Python `>=3.9`. Treat the actual 0.4.0 source, `pyproject.toml`, tests and signatures as authoritative when inventorying behavior. A function in the separate showcase file is **not** proof that the function exists in 0.4.0 or that it is correct.

The current public PyPI page also documents important caveats about its example evidence: the processed SentiMix copy is lowercased, stopword-stripped and stemmed; some spelling-swap evaluations use transformations derived from the same rules as the normalizer, making their protected result internally consistent by construction; language rules have not all been reviewed by fluent speakers; and the transformer sample is limited. Treat these results as a proof of concept, not as proof that FlipRate improves real-world robustness.

Sources to re-check before a release:

- PyPI / current package description: https://pypi.org/project/fliprate/
- GitHub repository: https://github.com/ShivSupane/fliprate
- Uploaded function showcase: `fliprate_showcase.py` (demonstration, not a substitute for tests)

---

## 3. Hard engineering rules

These rules apply throughout every phase unless the project owner explicitly changes them.

1. **Python compatibility:** Keep Python 3.9 support. Do not use syntax or standard-library behavior unavailable in Python 3.9.
2. **Core dependencies:** Do not add a new required runtime dependency. Optional integrations must remain optional and fail with actionable messages when missing.
3. **Negation safety:** A transformation must never be classified as meaning-preserving merely because its normalized text looks similar. Adding, deleting, or changing negation (including supported Hindi/Marathi negations) must be caught by dedicated tests. Ambiguous candidates must be rejected or sent for review.
4. **No silent installation:** Library code and showcase scripts must not invoke `pip install` automatically. Explain how to install the missing extra and exit clearly.
5. **No BOM:** Generated Python, TOML, JSON and documentation text files should be UTF-8 without a byte-order mark. Add a check if practical.
6. **No silent data mutation:** Never change ground-truth labels, silently drop rows, or modify users' original text without exposing the change.
7. **No raw text by default in logs/reports:** Store aggregate metrics and safe metadata by default. Per-example text must be an explicit opt-in.
8. **No unsupported claims:** Clearly separate toy examples, synthetic or rule-generated tests, and results on natural held-out data.
9. **Compatibility:** Audit which names and behaviors are actually public in 0.4.0. Preserve published names where practical; deprecate before removing them. Do not deprecate names that were only proposed in the showcase and never shipped.
10. **No publishing or tagging as part of implementation work:** Build, test and inspect artifacts locally. Do not tag a release or publish to PyPI unless the project owner explicitly authorizes that action.
11. **Readable code beats minimum line count:** Remove duplicated behavior and keep the code modular; do not squash useful abstractions just to reduce the number of lines.
12. **Reproducibility:** Randomness, transformation configuration, rule fingerprint and evaluation settings must be identifiable in reports.

---

## 4. Phase 0 — Audit and freeze the 0.4.0 baseline

**Goal:** Know exactly what exists and what breaks before refactoring.

### Work

- Inspect `pyproject.toml`, `fliprate/__init__.py`, module files, current CLI entry points, tests, documentation, examples, and evidence files.
- Create a public API inventory from the actual installed/source 0.4.0 package: exported names, signatures, result types, exceptions, dependencies and CLI commands.
- Compare the inventory with `fliprate_showcase.py`. Mark every showcased call as `exists`, `missing in 0.4.0`, `not tested`, or `verified by a test`.
- Run the current test suite and save the exact command, environment, Python version and result. Build the wheel and source distribution locally and inspect their contents.
- Record existing CLI options before changing their behavior.
- Record current runtime call counts and evaluation time on a small deterministic fixture; do not optimize yet.

### Deliverables

- `docs/API_INVENTORY.md`
- A baseline test report or CI artifact
- A short list of bugs and untested contracts, ordered by severity
- Baseline latency and model-invocation measurements

### Acceptance criteria

- A clean checkout can be installed and tested using documented commands.
- Every public 0.4.0 name has a recorded signature or an explicit reason it could not be inspected.
- The showcase no longer implies that a missing 0.4.0 API is already available.
- Baseline build artifacts can be inspected locally without publishing.

Do not begin API pruning until this phase is complete.

---

## 5. Version 0.5.0 — Correctness, a stable API and the first CI gate

**Goal:** Make current promises real and let developers fail a build when a measured requirement is violated.

### 5.1 Replace showcase success messages with actual tests

Keep the showcase as a tutorial/demo if useful, but add a true `pytest` suite. Each test must assert behavior, not simply that the function did not crash.

Test at least:

- Normalizer expected outputs, idempotence where intended, change tracking and preservation of untouched text.
- Perturbation count, output shape, deterministic behavior for a fixed seed, duplicate handling and variant safety.
- Script detection for Latin, Devanagari, mixed script, digits-only and punctuation-only strings.
- Gate decisions and fields, including stable, transformable and review cases.
- Correct metric calculations against hand-computed fixtures: accuracy, macro-F1, flip rate and confidence-interval bounds.
- Dataset loading/validation: empty inputs, malformed rows, duplicates, conflicting labels, unknown labels and invalid file paths.
- Model adapters: normal output, exceptions, timeout/retry behavior where supported, invalid output types, and optional dependency missing.
- Predictor caching: duplicate inputs, result order, cache-key correctness and call counts.
- Async/streaming APIs: event ordering, completion result, exceptions and cancellation behavior where supported.
- CLI success, validation error, model/evaluation error and threshold failure exit codes.

Edge cases must include `None`, non-string values, empty and whitespace-only text, zero-width characters, Unicode combining marks, long inputs, empty datasets, duplicate examples and conflicting labels. Do not assert that every unusual Unicode string must be normalized into one particular output unless that is the documented contract.

### 5.2 Fix the showcase's false-pass behavior

In the supplied showcase, the invalid-input test can print a red failure message when invalid input is accepted yet fail to add that problem to the final `errors` list. The script can then claim that the validation section passed. Replace these patterns with assertions or a test runner that tracks failures consistently.

Remove the automatic `pip install` attempt. If an import fails, display the supported installation command or optional extra and exit with a non-zero code.

### 5.3 Settle the public API

Use these as the core top-level entry points:

- `normalize(...)` — inspectable normalization result.
- `perturb(...)` — controlled generation of candidate variants.
- `gate(model, text, ...)` — runtime single-message decision.
- `test(model, dataset, ...)` — offline robustness and task-quality report.
- `protect(model, ...)` — explicit wrapper for inference-time normalization/protection.

Keep `normalize_text(...)` only if its compatibility status and distinction from `normalize(...)` justify it; otherwise treat it as a compatibility wrapper. Do not create overlapping names just because the showcase contains them.

### 5.4 Add a configuration object without forcing it on beginners

Provide a model-bound `FlipRate` convenience object, subject to signature review:

```python
engine = fr.FlipRate(
    model,
    rules=None,
    stability_threshold=0.8,
    n_variants=8,
    seed=0,
)

result = engine.gate("mala product avadlaaa nahi")
report = engine.test(dataset)
safe_model = engine.protect()
```

The module-level functions remain available and share the same internal implementation. Avoid a mysterious mutable global default instance. Ensure `stability_threshold` is not confused with a CI policy such as `max_flip_rate`.

### 5.5 Unify decision results carefully

Provide one canonical `Decision` type for the single-message gate family. Include documented fields for the decision, prediction, text actually sent to the model, stability evidence, variant count, explanation and review/routing status as appropriate.

Preserve distinct types that describe different things, such as `NormalizeResult`, `Variant`, dataset `Report`, benchmark scores, benchmark suites, experiment results, `RuleSet` and model-adapter interfaces. Do not merge reports and single-message decisions into one oversized class.

### 5.6 Compatibility and deprecation

- Map actual 0.4.0 names to the new canonical API.
- Keep thin wrappers for compatible old names where practical.
- Emit `DeprecationWarning` only for truly deprecated published names, with the replacement and expected removal window in the message and documentation.
- Do not remove a 0.4.0 public name in the same release that first deprecates it.
- Keep adapters as separate implementations of a shared interface when they serve distinct backends.

### 5.7 Ship an initial CI regression gate

Extend the existing CLI rather than creating a parallel CLI. Verify its actual current syntax before implementation. An intended workflow is:

```bash
fliprate test --data data/validation.csv --max-flip-rate 0.05 --save report.json
```

The exact arguments must fit the existing command and explicit model-selection mechanism. Required behavior:

- Exit `0` when the requested quality thresholds pass.
- Exit non-zero when any requested threshold fails.
- Exit non-zero with an actionable diagnostic for invalid data, configuration or evaluation errors.
- Record the applied thresholds and whether the run completed fully.
- Report accuracy alongside flip rate, and include metrics by transformation type when supported.
- Never print a reassuring success result after a failed assertion or failed evaluation.

If straightforward after the existing report API is understood, add a small `fr.assert_stable(...)` helper as a thin wrapper over the same evaluation logic, not a second implementation.

### Version 0.5 acceptance gate

- All unit and contract tests pass on the supported Python matrix, including Python 3.9.
- Regression tests reproduce every fixed bug.
- No new required runtime dependencies.
- The CLI's pass/fail behavior and exit codes are tested.
- Wheel and sdist build successfully and contain the correct files.
- Documentation explains which features are genuinely in 0.5.0.
- Local verification only; no tagging or publishing without authorization.

---

## 6. Version 0.6.0 — Cost-aware evaluation and measured performance

**Goal:** Give developers control over the cost-versus-confidence trade-off without mislabelling incomplete tests as full reports.

### 6.1 Three explicit modes

- **`fast`:** one normal inference path, optionally with normalization/protection. No claim of verified robustness. This mode should not run hidden extra variant predictions.
- **`balanced`:** evaluate a deterministic, configured sample of reviewed transformations. Report the sample size and sampling policy.
- **`thorough`:** evaluate all configured variants for each selected input and calculate full requested metrics.

Use modes on the existing gate/evaluation APIs rather than adding redundant function names. Document that `protect()` is a wrapper for inference-time behavior; it is not a robustness evaluation.

### 6.2 Early stopping rules

- For a question such as “did any tested transformation flip this prediction?”, stop once a flip is observed only if the returned status explicitly means “instability detected; evaluation incomplete.”
- Do not use a first-flip early exit to report an exact flip rate.
- Early stop for a threshold decision only if the decision is mathematically established by the observed bounds; otherwise continue or return an indeterminate/incomplete result.
- Include `evaluation_complete`, `n_variants_requested`, `n_variants_evaluated` and the reason for stopping in the report.

### 6.3 Measure cost, don't guess

- Count underlying prediction examples separately from HTTP/API requests; one batched request may still evaluate many examples.
- Test cache hits, batch ordering, duplicate messages, partial failures, timeouts, retries and concurrency limits.
- Add bounded concurrency only if needed and supported safely by the adapter contract.
- Test cache invalidation/identity when model version, preprocessing, labels or relevant settings change.
- Record elapsed time and model-call counts; token/cost estimates should be optional and only shown when provider information is available.

### 6.4 Save reproducible reports

A versioned JSON report should include, where available:

- report schema version and FlipRate version
- model identifier and user-supplied model version/revision
- dataset fingerprint or ID without exposing raw text by default
- random seed, rules fingerprint, transformation configuration, mode and thresholds
- requested/completed variant counts
- original and transformed accuracy, macro-F1, flip rate and confidence intervals
- per-transformation summary, call/request counts and elapsed time
- warnings, evaluation completeness and reasons for abstention

Avoid claiming exact reproducibility for a remote nondeterministic model unless the underlying provider and configuration support it.

### Version 0.6 acceptance gate

- Mode semantics are documented and tested.
- `fast` makes no hidden robustness calls; the report clearly states that robustness was not checked.
- Incomplete evaluations cannot be confused with full statistics.
- Benchmarks include call counts and timing before/after the optimization.
- Repeating a seeded local-model evaluation produces the same variant set and metrics.

---

## 7. Version 0.7.0 — ML workflow integration and experimental augmentation

**Goal:** Let users integrate FlipRate into familiar training/testing workflows.

### 7.1 Optional scikit-learn transformer

Implement an optional `FlipRateNormalizer` in the scikit-learn extra, compatible with `BaseEstimator` and `TransformerMixin`.

- `fit(X, y=None)` returns `self` and does not learn from held-out labels.
- `transform(X)` preserves order and row count.
- It works in `Pipeline` with text vectorizers and is compatible with `sklearn.clone`.
- Core installation still works without scikit-learn installed.
- It documents that normalization can change model behavior and does not guarantee correctness.

The current Hugging Face integration should be tested and documented before building another wrapper for the same use case.

### 7.2 Training-time augmentation — experimental first

Provide an `augment(...)` helper only after the transformation and safety contract is tested.

- Augment training data only; never generate augmented examples from validation/test rows used for final estimates.
- Preserve labels only for transformations that pass the approved meaning-preservation checks.
- Preserve originals unless the caller explicitly asks otherwise.
- Support a seed and deterministic output.
- Report counts by transformation type and skipped/unsafe candidates.
- Test negation, entity/brand names, numbers, intensifiers, domain terms and ambiguous words.
- Mark the API experimental until a held-out experiment compares baseline training with augmented training.

Acceptance requires a reproducible comparison of original accuracy, transformed accuracy, macro-F1 and flip rate. Do not claim augmentation helps unless the evidence supports that conclusion.

### 7.3 Dataset ergonomics

- Consider duck-typed support for pandas DataFrames without importing pandas in the core path.
- Preserve columns, row order and labels when converting or exporting.
- Add near-duplicate and train/test leakage checks with explainable row references.
- Keep duplicate detection separate from silently dropping rows; return diagnostics and let the user decide.

### 7.4 CI testing helper

Expose an `assert_stable(...)` helper that calls the canonical evaluation path. Support clearly named constraints such as `max_flip_rate`, `min_accuracy` or `max_accuracy_drop`, rather than one ambiguous threshold. Test its exception messages and ensure failure diagnostics identify the violated criterion.

---

## 8. Version 0.8.0 — Better rule governance and diagnosis

**Goal:** Make rules safer to extend and failures faster to debug.

### 8.1 Protected terms

Allow callers to mark words or phrases that normalization/perturbation must leave unchanged, such as product names, organizations, account codes and domain-specific terms.

- Use explicit boundaries and deterministic matching.
- Define casing and Unicode matching behavior.
- Handle multiword phrases and overlaps consistently.
- Add tests for substrings, punctuation boundaries and mixed scripts.
- Keep the option language/domain-scoped where needed.

### 8.2 Reviewed rule-mining workflow

`learn_rules_from_texts` should produce **candidates**, not automatically activated production rules.

Suggested candidate fields: language, canonical form, observed variants, counts, evidence source, confidence/score with documented meaning, reviewer status, rule-set version and optional notes. Avoid exporting raw text by default.

Workflow: mine candidates → inspect examples → approve/reject → save a versioned rules file → test its fingerprint and regression effects → explicitly activate it.

A confidence score is not a guarantee of semantic equivalence. Human review by fluent speakers remains important for language-specific safety.

### 8.3 Explainability

- Identify the exact original and variant input, transformation type, prediction before/after and changed tokens where the information is available.
- Avoid claiming that a word caused the model's internal reasoning; describe observed input/output differences instead.
- Make per-transformation metrics available in reports.
- Keep examples opt-in for privacy-sensitive outputs.

### 8.4 Expand language/script coverage only with tests

Prioritize romanized Hindi/Marathi and supported Latin/Devanagari/mixed-script behavior first. Additional Indic scripts should not be advertised as supported merely because language rules exist. Add a script/language only after tests, reviewed examples and dataset evidence are available. Unsupported or ambiguous text should produce a documented review/abstention outcome rather than unsafe rewriting.

---

## 9. Version 0.9.0 — Evidence, documentation, privacy and release readiness

**Goal:** Make the public claims defensible and the package easy to adopt.

### 9.1 Real benchmark protocol

Build a reproducible benchmark using natural, held-out text and at least:

- one relevant transformer model
- one scikit-learn baseline
- raw inference and protected/normalized inference
- reviewed, meaning-preserving transformations that are not simply the normalizer reversing its own generated variants
- original and transformed accuracy, macro-F1, flip rate and appropriate confidence intervals
- per-transformation breakdown, runtime/call cost, and failed-example analysis

Audit duplicates, train/test leakage, class distribution, label quality and text preprocessing. Record all preprocessing steps. Do not rely only on stemming/stopword-stripped text to justify a claim about natural user typing noise.

Where possible, ask fluent Hindi/Marathi reviewers to inspect a stratified sample of variants and record whether the meaning truly remains stable. State the sample size, review method and limitations. Keep synthetic/toy examples separate from natural-data results.

### 9.2 Documentation

Required documentation:

- a 5-minute quickstart
- installation and optional extras
- core API reference and supported-model contract
- CLI reference with exit codes and examples
- how to create and validate a dataset
- cost modes and what each does/does not prove
- CI integration and baseline comparison
- privacy/logging behavior
- troubleshooting and known limitations
- migration guide for deprecated 0.4.0 names
- benchmark protocol, source data provenance, scripts and caveats
- contribution and security reporting instructions

A docs site (for example MkDocs) is optional. A correct, versioned README and API reference are more important than a fancy site.

### 9.3 Security and packaging

- Add/review `SECURITY.md` and ensure supported security-reporting instructions are real.
- Ensure type hints are useful and `py.typed` is shipped if inline typing is part of the supported contract.
- Review optional dependency boundaries, network calls, file handling, CSV limits, API-key handling and logging.
- Build wheel and source distribution locally; inspect file lists; run packaging checks such as `twine check` if already available as a development tool.
- Test a clean installation in an isolated environment, including core-only install and relevant extras.
- Maintain Python 3.9 compatibility and avoid adding required dependencies.

### 9.4 Monitoring — defer full infrastructure

Do not build a dashboard or require Prometheus/OpenTelemetry/MLflow/W&B for the core library. First provide a stable structured event/report interface. A minimal optional callback/hook can be considered only if users need it and privacy behavior is tested. Full monitoring integrations should follow real usage feedback, not precede it.

---

## ✅ 10. Version 1.0.0 — Freeze and support the contract *(COMPLETED)*

All criteria have been met:

- ✅ All advertised core behavior is covered by automated tests (414 passing, Python 3.9–3.13).
- ✅ Supported Python versions, clean installs, wheel/sdist and optional extras have been tested.
- ✅ Negation and meaning-preservation invariants have dedicated regression tests.
- ✅ CLI thresholds reliably return documented exit codes (0 / 1 / 2).
- ✅ Cost modes and incomplete results cannot be misrepresented as complete robustness evaluations.
- ✅ Reports are versioned and reproducible to the extent supported by the model provider.
- ✅ API compatibility and deprecation policies are documented in `README.md` and `API_REFERENCE.md`.
- ✅ Real-model benchmark data, evaluation scripts, and caveats are available via `audit_dataset` and `run_benchmark_protocol`.
- ✅ README claims match evidence and supported language/script coverage.
- ✅ Users can protect private text by default; `opt_in_examples=True` must be set deliberately.
- ✅ Release contents have been inspected and all checks pass.
- ✅ `pyproject.toml` classifier updated to `Production/Stable`.
- ✅ Tagged and published by explicit project owner authorization only.

**Released: v1.0.0**

---

## 11. What NOT to build yet

Keep these out of the critical path until core correctness, cost and evidence are established:

- A large dashboard or mandatory hosted service
- Required Redis, database, telemetry or vendor SDK dependencies
- Many aliases for the same operation
- Claims of broad support for every Indic script without tests
- Automatic activation of mined spelling rules
- Large domain-specific rule packs without review provenance
- Complex automatic model routing or retraining
- A benchmark with only toy data, only one model, or transformations generated by the same rules the normalizer uses
- Optimization without a measured bottleneck
- A broad rewrite that breaks 0.4.0 names without a migration strategy

---

## 12. Definition of done for every PR

Every change should meet the following checklist before moving on:

- [ ] The problem and intended behavior are written down.
- [ ] The existing implementation and public API were inspected before editing.
- [ ] New behavior has a regression test and relevant edge-case tests.
- [ ] Negation/meaning-safety tests were run when text transformations are involved.
- [ ] No new required runtime dependency was introduced.
- [ ] Python 3.9 compatibility was considered and tested where possible.
- [ ] No UTF-8 BOM was introduced.
- [ ] Documentation and examples match the actual implementation.
- [ ] Tests and packaging/build checks pass.
- [ ] No raw text is logged or exported by default.
- [ ] The change has not tagged or published a release.

---

## 13. The exact order to work in

Follow this sequence, and do not skip ahead because a later feature feels more exciting:

1. **Audit the real 0.4.0 source and API.** Establish the baseline test and build results.
2. **Fix false passes and build contract tests.** The test suite becomes the safety net for every later refactor.
3. **Finalize the small core API and compatibility layer.** Implement `FlipRate` as an optional convenience wrapper, not a replacement for useful functions.
4. **Implement and test the CLI regression gate.** Add clear threshold semantics, reliable exit codes and a machine-readable report.
5. **Measure call count and latency.** Keep benchmark data before optimizing.
6. **Add cost modes and truthful early-exit behavior.** Never call a partial evaluation a full flip-rate report.
7. **Add sklearn integration and experimental augmentation.** Validate on held-out data and preserve labels only for meaning-safe changes.
8. **Add protected terms, reviewed rule workflow and detailed explanations.** Preserve a human approval step before rule activation.
9. **Run an independent real-data evaluation.** Publish scripts, preprocessing, limitations and results together.
10. **Harden docs, security and packaging.** Build/test release artifacts locally.
11. **Review every 1.0 criterion.** Only publish/tag after explicit authorization.

## 14. Recommended implementation prompt for each phase

Use a phase-specific prompt like this with your coding assistant:

> Work only on Phase [N] of `FLIPRATE_MASTER_ROADMAP.md`. First inspect the repository and report the actual files, APIs and tests relevant to the phase. Do not assume the showcase represents the published package. Make the smallest safe set of changes, add assertion-based tests for every new behavior and regression, preserve Python 3.9 support, add no required runtime dependencies, preserve negation safety, do not introduce UTF-8 BOMs, and do not tag or publish anything. Run the relevant tests and packaging checks, report exact commands/results, list changed files, and clearly disclose any check you could not run. Do not start the next phase until the acceptance criteria for this phase pass.

---

## Final product decision

**Build FlipRate as a focused, dependable robustness regression gate for messy Indian-language NLP.** Make correctness and reproducibility the foundation; make CI adoption the first practical payoff; then improve inference cost, training integration, rules and evidence. A small set of trustworthy capabilities is a stronger product than a large surface of unverified functions.
