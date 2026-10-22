# Changelog

## 0.3.0 (current)
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
