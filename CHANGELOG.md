# Changelog

## 0.2.1 (current)
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
