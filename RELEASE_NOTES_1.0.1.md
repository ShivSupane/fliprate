# FlipRate v1.0.1 — Release Notes

**Release date:** October 2026  
**Author:** Shiv Supane  
**PyPI:** `pip install fliprate==1.0.1`  
**Tests:** 414 passed, 0 failed — Python 3.9 / 3.10 / 3.11 / 3.12 / 3.13

---

## This is an ergonomics & documentation release

FlipRate 1.0.1 keeps the **frozen 1.0.0 API** and adds convenience properties,
pass-through parameters, and a complete Colab-ready showcase. There are no
breaking changes — everything from 1.0.0 works unchanged.

---

## What's new in 1.0.1

### Ergonomics & convenience properties
- `Report.confidence_interval` (alias for `flip_rate_ci`) and `Report.n_flips` (`len(flips)`)
- `Decision.flip_rate` (derived from `raw_stability`)
- `RuntimeGuardResult.flip_rate`
- `BenchmarkSuite.baseline` (alias for `baseline_name`)
- `NormalizeResult.changed` (`bool(changes)`)

### Pass-through parameters
- `normalize_text()` now accepts and forwards `protected_terms` to `normalize()`
- `make_protected_model()` and `protect()` / `ProtectedModel` now accept `protected_terms`

### Showcase & living reference
- Complete, executable `scratch/fliprate_showcase.py` covering all 23 API modules end-to-end
  with deep WHAT / WHY / HOW / WHEN explanations and ASCII-safe Windows terminal support
- **New: Google Colab notebook** — `examples/fliprate_colab_showcase.ipynb`
  - Browser-runnable twin of the local showcase
  - Downloads the **real Dakshina + Aksharantar** transliteration datasets from their
    official sources (Google Storage + Hugging Face)
  - Runs a **live Hugging Face** multilingual sentiment model through the robustness gate
  - Classifies text through **live LLM APIs** (Google Gemini / OpenAI) using Colab Secrets,
    then fixes flips with `fr.protect()` — graceful local fallback when no key is set
  - Wires the downloaded dataset spelling rules straight into `gate()`, `test()`, and the
    reproducible benchmark protocol to show FlipRate's real power against real data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShivSupane/fliprate/blob/main/examples/fliprate_colab_showcase.ipynb)

### Documentation
- README updated with a Showcase & Colab notebook section and the 1.0.1 version entry
- `docs/QUICKSTART.md` and `docs/PROJECT_OVERVIEW.md` now link the Colab notebook

---

## Migration from 1.0.0

No changes required. All 1.0.0 imports, function signatures, and CLI commands work unchanged.

---

## Known limitations

Unchanged from 1.0.0 — transliteration support is focused on Devanagari-script
languages and romanized code-mixed text; `LLMClassifier` requires you to supply a
client / completion function (no API key is bundled); Redis-backed infrastructure
remains optional.

---

*FlipRate — measure it. Fix it. Ship it.*
