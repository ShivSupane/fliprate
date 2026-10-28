# FlipRate v1.0.2 — Release Notes

**Release date:** October 2026  
**Author:** Shiv Supane  
**PyPI:** `pip install fliprate==1.0.2`  
**Tests:** 414 passed, 0 failed — Python 3.9 / 3.10 / 3.11 / 3.12 / 3.13

---

## This is a documentation & showcase release

FlipRate 1.0.2 keeps the **frozen 1.0.0 API** and adds no library code. It ships a
Google Colab showcase notebook and refreshes the project documentation. Everything
from 1.0.1 works unchanged.

---

## What's new in 1.0.2

### Google Colab showcase notebook
- New `examples/fliprate_colab_showcase.ipynb` — a browser-runnable twin of the local
  `scratch/fliprate_showcase.py`. Open it in one click from the README's Colab badge.
- **Real datasets**: downloads the Dakshina (Google, CC BY-SA 4.0) and Aksharantar
  (AI4Bharat, CC BY 4.0) transliteration corpora from their official sources and turns
  thousands of human-verified transliteration pairs into FlipRate spelling rules wired
  straight into `gate()`, `test()`, and the reproducible benchmark protocol.
- **Live models**: runs a Hugging Face multilingual sentiment model through the robustness
  gate, and classifies text through live LLM APIs (Google Gemini / OpenAI) using Colab
  Secrets — then fixes the flips with `fr.protect()`.
- **Always runs**: every live section degrades gracefully to a local fallback when an
  API key or download is missing, so the notebook never crashes.

### Documentation refresh
- README: new "Showcase & Colab notebook" section with an Open-in-Colab badge, an updated
  project layout, and versioning-table entries through 1.0.2.
- `docs/QUICKSTART.md`: new "Try it in Colab" section.
- `docs/PROJECT_OVERVIEW.md`: corrected the stale version/status header and linked the
  Colab notebook.
- `scratch/fliprate_showcase.py`: header bumped and pointed to its Colab twin.

---

## Migration from 1.0.1

No changes required. All 1.0.1 imports, function signatures, and CLI commands work
unchanged. (The public API remains frozen at 1.0.0.)

---

## Known limitations

Unchanged from 1.0.1 — transliteration support is focused on Devanagari-script
languages and romanized code-mixed text; `LLMClassifier` requires you to supply a
client / completion function (no API key is bundled); Redis-backed infrastructure
remains optional.

---

*FlipRate — measure it. Fix it. Ship it.*
