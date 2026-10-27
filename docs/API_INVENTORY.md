# FlipRate 0.4.0 / 0.5.0-dev Baseline Public API Inventory

**Audit Date:** 27 October 2026  
**Target Environment:** Python >= 3.9 (Audited on Python 3.13.2, Windows AMD64)  
**Branch:** `phase-0-baseline-audit`  
**Status:** Phase 0 Acceptance Criteria Passed  

---

## 1. Executive Summary

This inventory establishes the frozen baseline of the FlipRate package before any API refactoring or pruning.
It catalogues all 86 public symbols declared in `fliprate.__all__`, evaluates existing test coverage, audits current CLI commands, records baseline latency and model invocation costs on deterministic fixtures, and documents discrepancies between the package source and demonstration scripts.

### Baseline Key Facts
- **Total symbols exported in `__all__`:** 86
- **Symbols with dedicated tests:** 78 (90.7%)
- **Symbols untested in `tests/`:** 8 (9.3% — primarily abstract base classes and internal dictionary constants)
- **Current pytest test suite:** 354 passed in 25.64s (`100% pass rate`)
- **All-functions verification script (`scratch/test_all_functions.py`):** 15/15 sections passed (`100%`)
- **Package build:** Local source distribution (`.tar.gz`) and wheel (`.whl`) build cleanly (`python -m build`) and pass `twine check` with zero warnings.
- **Zero runtime dependencies for core:** Verified `dependencies = []` in `pyproject.toml`. Optional extras (`dev`, `sklearn`, `api`, `hf`, `all`) remain cleanly decoupled.

---

## 2. Baseline Test Report & Build Artifacts

### Test Suite Run
```text
Command:        .\.venv\Scripts\python.exe -m pytest
Environment:    win32 -- Python 3.13.2, pytest-9.1.1, pluggy-1.7.0
Configuration:  pyproject.toml
Root Directory: C:\Users\ROYAL\shiv_DOS\peroject\nlp fliprare own lib\fliprate v4\fliprate
Results:        354 passed in 25.64s
```

### Distribution Build & Inspection
```text
Command:        .\.venv\Scripts\python.exe -m build
Output Files:   dist/fliprate-0.5.0.tar.gz
                dist/fliprate-0.5.0-py3-none-any.whl
Twine Check:    twine check dist/* -> All PASSED
```

Package file contents inspect clean: includes all `src/fliprate/*.py` modules, `py.typed`, `data/sample_support.csv`, and `web/index.html`.

---

## 3. Baseline Latency and Model-Invocation Accounting

Measured on a deterministic fixture (10 romanized Hinglish examples, toy keyword classifier):

| Operation | Parameters | Elapsed Time | Model Calls | Call Accounting Formula | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`fr.gate()`** | 1 input, `n_variants=8`, `seed=42` | **1.44 ms** | **9 calls** | 1 original + 8 variants | Deterministic |
| **`fr.test()`** | 10 inputs, `n_variants=5`, `seed=42` | **11.54 ms** | **60 calls** | $N \times (1 + \text{variants}) = 10 \times (1 + 5)$ | Deterministic |
| **`fr.protect()`** | 10 inputs inference | **0.33 ms** | **10 calls** | 1 call per input (normalized) | Deterministic |

> **Cost Note for Phase 0.6.0:** The baseline evaluation runs full brute-force model invocations without early-stopping or cost controls. In Phase 0.6.0, `fast` mode will eliminate the 8 variant evaluations when robustness testing is not requested, and `balanced`/`thorough` will implement mathematically grounded early-stopping.

---

## 4. Public API Inventory (86 Exported Symbols)

| Symbol | Type | Origin Module | Signature | In Showcase? | Test Suite Coverage | Baseline Status |
| :--- | :--- | :--- | :--- | :---: | :--- | :--- |
| `AKSHARANTAR_LANGUAGES` | dict | `fliprate` | `Constant` | Yes | test_complete_coverage.py | Verified by test |
| `BaseJobStore` | ABCMeta | `fliprate.service` | `()` | Yes | Untested in tests/ | Untested in tests/ |
| `BaseRateLimiter` | ABCMeta | `fliprate.service` | `()` | Yes | Untested in tests/ | Untested in tests/ |
| `BenchmarkSuite` | type | `fliprate.benchmark` | `(scores: 'Dict[str, ModelBenchmarkScore]' = <factory>, baseline_name: 'Optional[str]' = None, metadata: 'Dict[str, Any]' = <factory>) -> None` | Yes | test_complete_coverage.py | Verified by test |
| `CROSS_SCRIPT_MAP` | dict | `fliprate` | `Constant` | Yes | Untested in tests/ | Untested in tests/ |
| `CROSS_SCRIPT_REV` | dict | `fliprate` | `Constant` | Yes | Untested in tests/ | Untested in tests/ |
| `CallableAdapter` | ABCMeta | `fliprate.adapters` | `(fn: 'Callable[[str], str]', name: 'Optional[str]' = None)` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `DAKSHINA_LANGUAGES` | dict | `fliprate` | `Constant` | Yes | test_complete_coverage.py | Verified by test |
| `DEFAULT_RULES` | RuleSet | `fliprate.rules` | `Constant` | Yes | test_complete_coverage.py, test_dataset_rules.py, test_devanagari.py, test_service_gate_experiment.py | Verified by test |
| `DEVANAGARI_GROUPS` | list | `fliprate` | `Constant` | Yes | Untested in tests/ | Untested in tests/ |
| `ExperimentResult` | type | `fliprate.experiment` | `(reports: 'Dict[str, Dict[str, Report]]', config: 'Dict[str, Any]' = <factory>) -> None` | Yes | test_complete_coverage.py | Verified by test |
| `GateResult` | type | `fliprate.gating` | `(decision: str, reason: str, original_text: str, text_used: str, prediction: Any, raw_stability: Optional[float], normalized_stability: Optional[float], n_variants: int, warnings: List[str] = <factory>, flipped_variants: List[str] = <factory>) -> None` | No | test_complete_coverage.py, test_v03_features.py | Verified by test |
| `HFTextClassifier` | ABCMeta | `fliprate.adapters` | `(model_name: 'str', device: 'Any' = None, batch_size: 'int' = 32, max_length: 'int' = 256, label_map: 'Optional[Dict[str, str]]' = None, lowercase_labels: 'bool' = False, preprocess: 'Optional[Callable[[str], str]]' = None, pipeline_factory: 'Optional[Callable[[], Callable]]' = None)` | Yes | test_complete_coverage.py, test_models_adapters.py | Verified by test |
| `HttpServiceAdapter` | ABCMeta | `fliprate.adapters` | `(endpoint_url: 'str', headers: 'Optional[Dict[str, str]]' = None, timeout: 'float' = 5.0, retries: 'int' = 2, name: 'str' = 'http_service', parse_response: 'Optional[Callable[[Dict[str, Any]], str]]' = None)` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `InMemoryJobStore` | ABCMeta | `fliprate.service` | `(max_jobs: 'int' = 500, ttl: 'float' = 300, clock: 'Callable[[], float]' = <built-in function monotonic>)` | Yes | test_benchmark_and_hardening.py, test_complete_coverage.py | Verified by test |
| `InMemoryRateLimiter` | ABCMeta | `fliprate.service` | `(limit_per_min: 'int', clock: 'Callable[[], float]' = <built-in function monotonic>)` | Yes | test_benchmark_and_hardening.py, test_complete_coverage.py | Verified by test |
| `JobStore` | ABCMeta | `fliprate.service` | `(max_jobs: 'int' = 500, ttl: 'float' = 300, clock: 'Callable[[], float]' = <built-in function monotonic>)` | No | test_service_gate_experiment.py | Verified by test |
| `KINDS` | tuple | `fliprate` | `Constant` | Yes | Untested in tests/ | Untested in tests/ |
| `LANGUAGE_GROUPS` | dict | `fliprate` | `Constant` | Yes | Untested in tests/ | Untested in tests/ |
| `LLMClassifier` | ABCMeta | `fliprate.adapters` | `(complete: 'Callable[[str], str]', labels: 'Sequence[str]', instruction: 'Optional[str]' = None, name: 'str' = 'llm', invalid_label: 'str' = '__invalid__')` | Yes | test_complete_coverage.py, test_models_adapters.py | Verified by test |
| `ModelAdapter` | ABCMeta | `fliprate.adapters` | `()` | Yes | Untested in tests/ | Untested in tests/ |
| `ModelBenchmarkScore` | type | `fliprate.benchmark` | `(name: 'str', accuracy_original: 'float', accuracy_variants: 'float', macro_f1_original: 'float', macro_f1_variants: 'float', flip_rate: 'float', flip_rate_ci: 'Tuple[float, float]', protected_flip_rate: 'Optional[float]' = None, protected_accuracy: 'Optional[float]' = None, flip_rate_reduction: 'Optional[float]' = None, is_baseline: 'bool' = False, score_vs_baseline: 'Optional[float]' = None) -> None` | Yes | test_complete_coverage.py | Verified by test |
| `NormalizeResult` | type | `fliprate.normalization` | `(original: 'str', normalized: 'str', changes: 'List[Tuple[str, str]]' = <factory>) -> None` | No | test_complete_coverage.py | Verified by test |
| `Predictor` | type | `fliprate.models` | `(model: 'Callable', cache: 'bool' = True, batch_size: 'int' = 64, max_cache: 'int' = 50000)` | Yes | test_complete_coverage.py, test_models_adapters.py | Verified by test |
| `ProtectedModel` | type | `fliprate.models` | `(model: 'Callable', rules: 'Optional[RuleSet]' = None, lowercase: 'bool' = True)` | Yes | test_complete_coverage.py | Verified by test |
| `QuickCheckResult` | type | `fliprate.pipeline` | `(is_safe: 'bool', decision: 'str', prediction: 'str', suggested_text: 'str', reason: 'str', flipped_variants: 'List[str]') -> None` | No | test_complete_coverage.py | Verified by test |
| `RateLimiter` | ABCMeta | `fliprate.service` | `(limit_per_min: 'int', clock: 'Callable[[], float]' = <built-in function monotonic>)` | No | test_service_gate_experiment.py | Verified by test |
| `RedisJobStore` | ABCMeta | `fliprate.service` | `(redis_url: 'str', ttl: 'int' = 300, key_prefix: 'str' = 'fliprate:job:')` | Yes | test_benchmark_and_hardening.py, test_complete_coverage.py | Verified by test |
| `RedisRateLimiter` | ABCMeta | `fliprate.service` | `(redis_url: 'str', limit_per_min: 'int', key_prefix: 'str' = 'fliprate:rl:')` | Yes | test_benchmark_and_hardening.py, test_complete_coverage.py | Verified by test |
| `Report` | type | `fliprate.evaluation` | `(n_messages: 'int', n_variants: 'int', accuracy_original: 'float', macro_f1_original: 'float', accuracy_variants: 'float', macro_f1_variants: 'float', flip_rate: 'float', flip_rate_ci: 'Tuple[float, float]', accuracy_drop: 'float', accuracy_drop_ci: 'Tuple[float, float]', by_kind: 'Dict[str, Dict[str, float]]', flips: 'List[Dict[str, Any]]' = <factory>, seed: 'int' = 0, transitions: 'Dict[str, int]' = <factory>, metadata: 'Dict[str, Any]' = <factory>, message_stats: 'List[List[int]]' = <factory>) -> None` | Yes | test_complete_coverage.py, test_developer_toolkit.py, test_service_gate_experiment.py, test_v03_features.py | Verified by test |
| `RuleSet` | type | `fliprate.rules` | `(groups=None, lexicon=None, negations=None, ambiguous=None)` | Yes | test_complete_coverage.py, test_dataset_rules.py, test_devanagari.py, test_v03_features.py | Verified by test |
| `RuntimeGuardResult` | type | `fliprate.pipeline` | `(decision: 'str', prediction: 'Optional[str]', clean_text: 'str', should_route_to_human: 'bool', reason: 'str', metadata: 'Dict[str, Any]') -> None` | No | test_complete_coverage.py | Verified by test |
| `SklearnAdapter` | ABCMeta | `fliprate.adapters` | `(pipeline: 'Any', name: 'str' = 'sklearn')` | Yes | test_complete_coverage.py | Verified by test |
| `Variant` | type | `fliprate.perturbation` | `(text: 'str', kind: 'str') -> None` | No | test_complete_coverage.py | Verified by test |
| `__author__` | str | `fliprate` | `Constant` | Yes | test_complete_coverage.py | Verified by test |
| `__version__` | str | `fliprate` | `Constant` | Yes | test_complete_coverage.py | Verified by test |
| `add_custom_rule` | function | `fliprate.rules` | `(variants: 'Sequence[str]', canonical: 'Optional[str]' = None, rules: 'Optional[RuleSet]' = None) -> 'RuleSet'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `agate` | function | `fliprate.gating` | `(model: Callable[[str], Any], text: str, n_variants: int = 8, threshold: float = 0.8, min_variants: int = 3, seed: int = 0, kinds: Optional[Sequence[str]] = None, on_event: Optional[Callable[[Dict[str, Any]], NoneType]] = None, rules: Optional[fliprate.rules.RuleSet] = None, cache: bool = True, on_error: str = 'raise', script_aware: bool = True) -> fliprate.gating.GateResult` | Yes | test_complete_coverage.py, test_v03_features.py | Verified by test |
| `aksharantar_rules` | function | `fliprate.aksharantar` | `(*languages: 'str', base: 'Optional[RuleSet]' = None, min_variants: 'int' = 2, aksharantar_path: 'Optional[str]' = None) -> 'RuleSet'` | Yes | test_complete_coverage.py | Verified by test |
| `as_adapter` | function | `fliprate.adapters` | `(model: 'Any') -> 'ModelAdapter'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `audit` | function | `fliprate` | `(model, dataset, **kwargs)` | Yes | test_complete_coverage.py, test_models_adapters.py | Verified by test |
| `batch_guard` | function | `fliprate.pipeline` | `(model: 'Any', texts: 'Sequence[str]', **kwargs) -> 'List[GateResult]'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `benchmark` | function | `fliprate.benchmark` | `(models: 'Dict[str, Any]', dataset: 'Any', baseline: 'Optional[str]' = None, with_protection: 'bool' = True, n_variants: 'int' = 5, bootstrap: 'int' = 200, **kwargs) -> 'BenchmarkSuite'` | Yes | test_benchmark_and_hardening.py, test_complete_coverage.py | Verified by test |
| `check_model` | function | `fliprate.pipeline` | `(model: 'Any', dataset: 'Any', **kwargs) -> 'Report'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `compare_models` | function | `fliprate.benchmark` | `(model_a: 'Any', model_b: 'Any', dataset: 'Any', name_a: 'str' = 'model_a', name_b: 'str' = 'model_b', **kwargs) -> 'BenchmarkSuite'` | Yes | test_benchmark_and_hardening.py, test_complete_coverage.py | Verified by test |
| `dakshina_rules` | function | `fliprate.dakshina` | `(*languages: 'str', base: 'Optional[RuleSet]' = None, min_total: 'int' = 3, min_variants: 'int' = 2, dakshina_path: 'Optional[str]' = None) -> 'RuleSet'` | Yes | test_complete_coverage.py | Verified by test |
| `describe_rule_group` | function | `fliprate.rules` | `(word_or_group: 'str', rules: 'Optional[RuleSet]' = None) -> 'Optional[Dict[str, Any]]'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `detect_script` | function | `fliprate.gating` | `(text: str) -> str` | Yes | test_complete_coverage.py, test_devanagari.py | Verified by test |
| `drain` | function | `fliprate.events` | `(gen: 'Generator', on_event: 'Optional[Callable[[Event], None]]' = None)` | Yes | test_complete_coverage.py | Verified by test |
| `evaluate_pipeline` | function | `fliprate.pipeline` | `(model: 'Any', dataset: 'Any', save_path: 'Optional[str]' = None, **kwargs) -> 'Report'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `explain_gate_decision` | function | `fliprate.explain` | `(text_or_result: 'Any', model: 'Optional[Any]' = None, **kwargs) -> 'Dict[str, Any]'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `from_sklearn` | function | `fliprate.adapters` | `(pipeline: 'Any', name: 'str' = 'sklearn') -> 'SklearnAdapter'` | No | test_complete_coverage.py | Verified by test |
| `gate` | function | `fliprate.gating` | `(model: Callable[[str], Any], text: str, n_variants: int = 8, threshold: float = 0.8, min_variants: int = 3, seed: int = 0, kinds: Optional[Sequence[str]] = None, on_event: Optional[Callable[[Dict[str, Any]], NoneType]] = None, rules: Optional[fliprate.rules.RuleSet] = None, cache: bool = True, on_error: str = 'raise', script_aware: bool = True) -> fliprate.gating.GateResult` | Yes | test_complete_coverage.py, test_devanagari.py, test_gate.py, test_service_gate_experiment.py, test_v03_features.py | Verified by test |
| `guard` | function | `fliprate` | `(model, text, **kwargs)` | Yes | test_complete_coverage.py, test_models_adapters.py | Verified by test |
| `is_safe_variant` | function | `fliprate.rules` | `(original: 'str', variant: 'str', cross_script: 'bool' = False) -> 'bool'` | Yes | test_complete_coverage.py, test_devanagari.py, test_perturb_rules.py, test_v03_features.py | Verified by test |
| `iter_gate` | function | `fliprate.gating` | `(model: Callable[[str], Any], text: str, n_variants: int = 8, threshold: float = 0.8, min_variants: int = 3, seed: int = 0, kinds: Optional[Sequence[str]] = None, rules: Optional[fliprate.rules.RuleSet] = None, cache: bool = True, on_error: str = 'raise', script_aware: bool = True) -> Generator[Dict[str, Any], NoneType, fliprate.gating.GateResult]` | Yes | test_complete_coverage.py | Verified by test |
| `iter_test` | function | `fliprate.evaluation` | `(model: 'Model', dataset: 'Any', n_variants: 'int' = 5, kinds: 'Optional[Sequence[str]]' = None, seed: 'int' = 0, bootstrap: 'int' = 1000, rules: 'Optional[RuleSet]' = None, cache: 'bool' = True, batch_messages: 'int' = 8) -> 'Generator[Event, None, Report]'` | Yes | test_complete_coverage.py | Verified by test |
| `learn_rules_from_texts` | function | `fliprate` | `(texts, **kwargs)` | Yes | test_complete_coverage.py | Verified by test |
| `llm_classifier` | function | `fliprate.adapters` | `(complete: 'Callable[[str], str]', labels: 'Sequence[str]', **kw) -> 'LLMClassifier'` | No | test_complete_coverage.py | Verified by test |
| `load_aksharantar_groups` | function | `fliprate.aksharantar` | `(*languages: 'str', min_variants: 'int' = 2, aksharantar_path: 'Optional[str]' = None) -> 'List[Tuple[str, ...]]'` | No | test_complete_coverage.py | Verified by test |
| `load_dakshina_groups` | function | `fliprate.dakshina` | `(*languages: 'str', min_total: 'int' = 3, min_variants: 'int' = 2, dakshina_path: 'Optional[str]' = None) -> 'List[Tuple[str, ...]]'` | No | test_complete_coverage.py | Verified by test |
| `load_dataset` | function | `fliprate.evaluation` | `(source: 'Any', text_col: 'str' = 'text', label_col: 'str' = 'label', delimiter: 'Optional[str]' = None) -> 'List[Row]'` | Yes | test_complete_coverage.py, test_evaluation.py, test_v03_features.py | Verified by test |
| `load_rules` | function | `fliprate.rules` | `(language: 'str') -> 'RuleSet'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `macro_f1` | function | `fliprate.evaluation` | `(y_true: 'Sequence', y_pred: 'Sequence') -> 'float'` | Yes | test_complete_coverage.py, test_evaluation.py | Verified by test |
| `make_event` | function | `fliprate.events` | `(stage: 'str', **data: 'Any') -> 'Event'` | Yes | test_complete_coverage.py, test_gate.py | Verified by test |
| `make_protected_model` | function | `fliprate.pipeline` | `(model: 'Any', rules: 'Optional[Any]' = None, lowercase: 'bool' = True) -> 'ProtectedModel'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `normalize` | function | `fliprate.normalization` | `(text: 'str', lowercase: 'bool' = True, rules: 'Optional[RuleSet]' = None) -> 'NormalizeResult'` | Yes | test_complete_coverage.py, test_devanagari.py, test_gate.py, test_models_adapters.py, test_perturb_rules.py, test_v03_features.py | Verified by test |
| `normalize_text` | function | `fliprate.pipeline` | `(text: 'str', lowercase: 'bool' = True, rules: 'Optional[Any]' = None) -> 'str'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `paired_difference` | function | `fliprate.evaluation` | `(a: "'Report'", b: "'Report'", stat: 'str' = 'flip_rate', boots: 'int' = 2000, seed: 'int' = 0) -> 'Dict[str, Any]'` | Yes | test_complete_coverage.py, test_service_gate_experiment.py | Verified by test |
| `perturb` | function | `fliprate.perturbation` | `(text: 'str', n: 'int' = 10, kinds: 'Optional[Sequence[str]]' = None, seed: 'int' = 0, rules: 'Optional[RuleSet]' = None, cross_script: 'bool' = False) -> 'List[Variant]'` | Yes | test_complete_coverage.py, test_dataset_rules.py, test_devanagari.py, test_perturb_rules.py, test_v03_features.py | Verified by test |
| `preview_normalized_variants` | function | `fliprate.explain` | `(text: 'str', n: 'int' = 5, **kwargs) -> 'List[Dict[str, str]]'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `preview_variants` | function | `fliprate.rules` | `(text: 'str', n: 'int' = 10, **kwargs) -> 'List[str]'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `print_rule_hits` | function | `fliprate.explain` | `(text: 'str', rules: 'Optional[RuleSet]' = None) -> 'List[Dict[str, Any]]'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `protect` | function | `fliprate.models` | `(model: 'Callable', rules: 'Optional[RuleSet]' = None, lowercase: 'bool' = True) -> 'ProtectedModel'` | Yes | test_complete_coverage.py, test_devanagari.py, test_gate.py, test_models_adapters.py, test_service_gate_experiment.py | Verified by test |
| `quick_check` | function | `fliprate.pipeline` | `(model: 'Any', text: 'str', **kwargs) -> 'QuickCheckResult'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `read_csv_text` | function | `fliprate.evaluation` | `(content: 'str', text_col: 'str' = 'text', label_col: 'str' = 'label', delimiter: 'Optional[str]' = None) -> 'List[Row]'` | Yes | test_complete_coverage.py, test_evaluation.py, test_v03_features.py | Verified by test |
| `run_experiment` | function | `fliprate.experiment` | `(models: 'Dict[str, Callable]', dataset: 'Any', n_variants: 'int' = 5, seed: 'int' = 0, rules: 'Optional[RuleSet]' = None, with_protection: 'bool' = True, bootstrap: 'int' = 1000, on_progress: 'Optional[Callable[[str, str, Dict[str, Any]], None]]' = None) -> 'ExperimentResult'` | Yes | test_complete_coverage.py, test_service_gate_experiment.py | Verified by test |
| `runtime_guard` | function | `fliprate.pipeline` | `(model: 'Any', text: 'str', mode: 'str' = 'lenient', **kwargs) -> 'RuntimeGuardResult'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `safe_predict` | function | `fliprate.pipeline` | `(model: 'Any', text: 'str', on_review: 'str' = 'normalized', **kwargs) -> 'str'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `show_why_it_flipped` | function | `fliprate.explain` | `(text_or_model: 'Any', model_or_text: 'Any', **kwargs) -> 'List[Dict[str, Any]]'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `test` | function | `fliprate.evaluation` | `(model: 'Model', dataset: 'Any', n_variants: 'int' = 5, kinds: 'Optional[Sequence[str]]' = None, seed: 'int' = 0, bootstrap: 'int' = 1000, on_event: 'Optional[Callable[[Event], None]]' = None, rules: 'Optional[RuleSet]' = None, cache: 'bool' = True, batch_messages: 'int' = 8) -> 'Report'` | Yes | test_api.py, test_benchmark_and_hardening.py, test_complete_coverage.py, test_dataset_rules.py, test_devanagari.py, test_evaluation.py, test_models_adapters.py, test_service_gate_experiment.py, test_v03_features.py | Verified by test |
| `to_sse` | function | `fliprate.events` | `(event: 'Event') -> 'str'` | Yes | test_complete_coverage.py, test_gate.py | Verified by test |
| `twitter_preprocess` | function | `fliprate.adapters` | `(text: 'str') -> 'str'` | Yes | test_complete_coverage.py, test_models_adapters.py, test_v03_features.py | Verified by test |
| `validate_dataset` | function | `fliprate.validation` | `(dataset: 'Any', text_col: 'str' = 'text', label_col: 'str' = 'label') -> 'Dict[str, Any]'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `validate_text` | function | `fliprate.validation` | `(text: 'Any', max_length: 'int' = 10000, allow_empty: 'bool' = False) -> 'str'` | Yes | test_complete_coverage.py, test_developer_toolkit.py | Verified by test |
| `with_retries` | function | `fliprate.adapters` | `(fn: 'Callable', retries: 'int' = 3, backoff: 'float' = 0.5, exceptions: 'tuple' = (<class 'Exception'>,)) -> 'Callable'` | Yes | test_complete_coverage.py, test_models_adapters.py | Verified by test |

---

## 5. CLI Commands and Options Baseline

Current entrypoint: `fliprate = "fliprate.cli:main"`

| Command | Arguments / Options | Current Exit Code Behavior | Baseline Status |
| :--- | :--- | :--- | :--- |
| `perturb` | `text`, `-n 8`, `--rules` | Exits `0` on success | Functional |
| `normalize` | `text`, `--rules` | Exits `0` on success | Functional |
| `gate` | `text`, `--model`, `--rules` | Exits `0` on success | Functional |
| `demo` | `-n 5`, `--model`, `--rules` | Exits `0` on success | Functional |
| `test` | `--data`, `-n 5`, `--seed 0`, `--protect`, `--save`, `--model`, `--rules` | **Always exits `0`** (No threshold checking) | **Missing CI regression gate** (Needs `--max-flip-rate` and non-zero exit in v0.5.0) |
| `report` | `--data`, `--out`, `-n 5`, `--seed 0`, `--protect`, `--model`, `--rules` | Exits `0` on success | Functional |
| `check-data` | `data`, `--against`, `--rules` | Exits `0` if OK, `1` if invalid rows | Functional dataset gate |
| `split` | `data`, `--out-dir .`, `--test-frac 0.6`, `--seed 0`, `--rules` | Exits `0` on success | Functional |
| `mine` | `data`, `--out candidates.csv`, `--min-count 2`, `--rules` | Exits `0` on success | Functional |
| `audit` | `data`, `--out audit.csv`, `-n 100`, `--seed 0`, `--rules` | Exits `0` on success | Functional |
| `audit-summary`| `audit_csv` | Exits `0` on success | Functional |
| `explain` | `text`, `--model`, `--rules` | Exits `0` on success | Functional |
| `benchmark` | `--data`, `--models`, `--baseline`, `--out`, `-n 5`, `--rules` | Exits `0` on success | Functional |

---

## 6. Showcase Discrepancies and Bugs Identified

During Phase 0 audit, the following bugs, false assumptions, and untested contracts were identified and catalogued:

### High Severity
1. **False-Pass Behavior in Showcase Validation Section:**
   - *Description:* In `scratch/fliprate_showcase.py`, when testing `validate_text` on invalid inputs (`123`, `""`), if the function failed to raise, the script merely printed a red error string to stdout without raising an error or registering the problem in the `errors` list. It then executed `ok("validate_text correctly rejects invalid inputs")`, falsely reporting success.
   - *Status:* Corrected in Phase 0 showcase check by adding an explicit assertion.
2. **Automatic `pip install` on Import:**
   - *Description:* `scratch/fliprate_showcase.py` caught `ImportError` and silently attempted `subprocess.check_call([sys.executable, "-m", "pip", "install", "fliprate"])`, violating Rule 4.
   - *Status:* Replaced with an explicit actionable error message explaining manual installation and exiting non-zero.
3. **Missing Exit Codes on Quality Threshold Breach:**
   - *Description:* `fliprate test` CLI command currently always exits `0`, even if flip rate is 100%. A developer cannot use it as a CI regression gate.
   - *Status:* Slated for Phase 0.5.0 implementation.

### Medium Severity
4. **`RuleSet.groups` Data Structure Misconception:**
   - *Description:* The showcase assumed `rules.groups` was a dictionary and called `.items()`, whereas `RuleSet.groups` is `List[Tuple[str, ...]]`.
   - *Status:* Documented and aligned in showcase.
5. **`with_retries` Parameter Mismatch:**
   - *Description:* Showcase passed `max_retries=3`, but `fliprate.adapters.with_retries` takes `retries=3`.
   - *Status:* Aligned in showcase; compatibility alias or standardized name recommended for v0.5.0.
6. **`JobStore` Method Assumptions:**
   - *Description:* Showcase attempted to call `store.create(...)` and `store.get(...)`, but `JobStore` implements `put(kind, params) -> job_id` and `pop(job_id) -> data`.
   - *Status:* Aligned in showcase; interface clarified.
7. **Windows `cp1252` Unicode Console Encoding:**
   - *Description:* Unconfigured Windows PowerShell/cmd terminals threw `UnicodeEncodeError` when printing Unicode box-drawing characters.
   - *Status:* Added UTF-8 stdout reconfiguration guard.

### Low Severity
8. **Untested Abstract Base Classes in `tests/`:**
   - `BaseJobStore`, `BaseRateLimiter`, and `ModelAdapter` are defined as ABCs but have no direct test instantiating custom subclasses (only their concrete implementations are tested).
9. **Untested Linguistic Constant Mappings:**
   - `CROSS_SCRIPT_MAP`, `CROSS_SCRIPT_REV`, `DEVANAGARI_GROUPS`, and `LANGUAGE_GROUPS` are populated in `rules.py` but lacked explicit assertion tests verifying round-trip mapping integrity.

---

## 7. Phase 0 Acceptance Criteria Checklist

- [x] **A clean checkout can be installed and tested using documented commands:** Tested via `python -m pytest` (354 passed) and `scratch/test_all_functions.py` (100% passed).
- [x] **Every public 0.4.0 name has a recorded signature or an explicit reason it could not be inspected:** All 86 symbols inventoried above with exact signatures and module origins.
- [x] **The showcase no longer implies that a missing 0.4.0 API is already available:** All showcase calls aligned with actual signatures; false-passes and auto-pip-install removed; 19/19 sections pass cleanly.
- [x] **Baseline build artifacts can be inspected locally without publishing:** Local sdist (`.tar.gz`) and wheel (`.whl`) built and validated with `twine check`.

