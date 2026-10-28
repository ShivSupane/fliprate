# FlipRate v1.0.1 — Release Notes

**Release date:** October 2026  
**Author:** Shiv Supane  
**PyPI:** `pip install fliprate==1.0.1`  
**Tests:** 414 passed, 0 failed — Python 3.9 / 3.10 / 3.11 / 3.12 / 3.13

---

## This is an ergonomics & showcase release

FlipRate 1.0.1 keeps the **frozen 1.0.0 API** and adds convenience properties and
pass-through parameters, plus a complete local executable showcase. There are no
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
  with deep WHAT / WHY / HOW / WHEN explanations and ASCII-safe Windows terminal support.

---

## Migration from 1.0.0

No changes required. All 1.0.0 imports, function signatures, and CLI commands work unchanged.

---

*FlipRate — measure it. Fix it. Ship it.*
