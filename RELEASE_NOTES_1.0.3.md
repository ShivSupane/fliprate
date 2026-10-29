# FlipRate v1.0.3 — Release Notes

**Release date:** October 2026  
**Author:** Shiv Supane  
**PyPI:** `pip install fliprate==1.0.3`  
**Tests:** 414 passed, 0 failed — Python 3.9 / 3.10 / 3.11 / 3.12 / 3.13

---

## Hardened PyPI release pipeline & version alignment

FlipRate 1.0.3 hardens the GitHub Actions release workflow for automated PyPI publishing and ensures version consistency across all packaging metadata, documentation, and showcase suites.

---

## What's new in 1.0.3

### 1. PyPI Publishing Pipeline Hardening
- Added `skip-existing: true` to `.github/workflows/release.yml` using `pypa/gh-action-pypi-publish@release/v1`.
- Protects the deployment pipeline against transient re-runs and packaging collisions: if a distribution wheel was already uploaded, re-running the job safely skips existing artifacts instead of failing.
- Maintains GitHub release tag verification against `pyproject.toml` version metadata.

### 2. Version Bump & Package Synchronization
- Bumped package version from `1.0.2` → `1.0.3` in `pyproject.toml` and `src/fliprate/__init__.py`.
- Synchronized all documentation, roadmap, and showcase suite headers to 1.0.3.

### 3. Google Colab Showcase & Datasets Integration
- Full verification of `examples/fliprate_colab_showcase.ipynb` and `scratch/fliprate_showcase.py`.
- Features real transliteration datasets (**Dakshina** from Google Storage and **Aksharantar** from AI4Bharat / Hugging Face).
- Demonstrates live LLM testing (**Google Gemini** and **OpenAI**) with strict single-label extraction and FlipRate's `protect()` wrapper to prevent prediction flips and cache queries.
- Multi-model benchmark suite comparing Scikit-Learn, Hugging Face Transformers, and LLMs side-by-side with interactive HTML scorecards.

---

## Migration from 1.0.2 / 1.0.1 / 1.0.0

No changes required. The public API remains completely frozen at 1.0.0 and backward-compatible.

```bash
pip install --upgrade fliprate
```

---

*FlipRate — measure it. Fix it. Ship it.*
