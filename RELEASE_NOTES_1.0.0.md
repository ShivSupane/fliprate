# FlipRate v1.0.0 — Release Notes

**Release date:** October 2026  
**Author:** Shiv Supane  
**PyPI:** `pip install fliprate==1.0.0`  
**Tests:** 414 passed, 0 failed — Python 3.9 / 3.10 / 3.11 / 3.12 / 3.13

---

## This is the stable release

FlipRate 1.0.0 is the first production-stable release.  
The public API is now **frozen** — all public functions, classes, and CLI commands are guaranteed backward compatible.  
Future releases will follow [Semantic Versioning](https://semver.org/): breaking changes only in a new major version.

---

## What's included in 1.0.0

### Core robustness engine
- `fr.gate()` — real-time quality gate: PASS / TRANSFORM / REVIEW
- `fr.quick_check()` — fastest one-line check
- `fr.safe_predict()` — normalize → gate → predict pipeline
- `fr.normalize_text()` — clean messy romanized text
- `fr.protect()` — protected model wrapper

### Batch evaluation
- `fr.test()` / `fr.check_model()` — flip rate + accuracy + 95% Wilson CI on any labeled CSV
- `fr.iter_test()` — streaming batch evaluation
- `fr.paired_difference()` — statistical comparison between two models

### Multi-model benchmarking
- `fr.benchmark()` — compare N models side-by-side with HTML / Markdown dashboards
- `fr.run_benchmark_protocol()` — rigorous reproducible benchmark with automatic dataset audit

### Dataset quality
- `fr.audit_dataset()` — class balance, script distribution, exact/near-duplicates, label conflicts, leakage

### Explainability
- `fr.show_why_it_flipped()` — per-variant word diffs
- `fr.explain_gate_decision()` — human-readable gate explanation
- `fr.explain_report_flips()` — privacy-safe report summary (raw text never included by default)
- `fr.preview_normalized_variants()` — preview normalization output

### Devanagari & multilingual
- `fr.detect_script()` — latin / devanagari / mixed detection
- Unicode nukta normalization, Devanagari digit swap, danda punctuation, meaning-safe spelling variants
- Dakshina and Aksharantar transliteration dataset loaders

### Model adapters
- `fr.HFTextClassifier` — Hugging Face pipelines (lazy-loaded, batched)
- `fr.from_sklearn()` / `fr.SklearnAdapter` — scikit-learn pipelines
- `fr.LLMClassifier` — OpenAI / Anthropic / Gemini via strict label extraction
- `fr.HttpServiceAdapter` — REST/HTTP services with automatic retries
- `fr.CallableAdapter` — any Python callable

### Training integration
- `fr.augment()` — meaning-safe data augmentation
- `fr.compare_augmentation()` — compare model performance before/after augmentation
- `fr.find_near_duplicates()` — detect spelling variant pairs in any dataset
- `fr.check_leakage()` — detect train/test overlap

### Rule management
- `fr.DEFAULT_RULES` / `fr.RuleSet` — curated spelling rules for Hindi / Marathi / Hinglish
- `fr.add_custom_rule()` — add project-specific spelling groups
- `fr.mine_variants()` — mine rule candidates from raw text
- Protected terms: preserve brand names, codes, and domain terms through normalization

### CLI
- `fliprate gate` · `fliprate normalize` · `fliprate perturb` · `fliprate explain`
- `fliprate test` · `fliprate benchmark` · `fliprate report`
- `fliprate check-data` · `fliprate split` · `fliprate mine`
- Exit codes: `0` pass, `1` threshold violation, `2` config/data error

### Web console
- `uvicorn fliprate.api:app --reload` → Playground + Robustness Lab tabs

### Security & privacy
- `SECURITY.md` with supported versions, private disclosure process, response timeline
- Privacy-by-default: raw input text never appears in report summaries without `opt_in_examples=True`
- Zero runtime dependencies in core package

---

## Migration from 0.9.0

No changes required. All 0.9.0 imports, function signatures, and CLI commands work unchanged.

---

## Known limitations

- Transliteration support is focused on Devanagari-script languages (Hindi, Marathi) and romanized code-mixed text.  
  Tamil, Telugu, Bengali, and other scripts receive Latin-only perturbations.
- LLM-based classification (`LLMClassifier`) requires the user to supply a client object; no LLM API key is bundled.
- Redis-backed infrastructure (`RedisJobStore`, `RedisRateLimiter`) requires an optional Redis server; not needed for core usage.

---

## Checksums

```
fliprate-1.0.0-py3-none-any.whl   (pure Python, any platform)
fliprate-1.0.0.tar.gz             (source distribution)
```

Both built with `python -m build` and verified with `twine check dist/*`.

---

*FlipRate — measure it. Fix it. Ship it.*
