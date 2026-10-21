"""
Loader for the Dakshina dataset spelling variant groups.

The Dakshina dataset (Google Research, 2020, CC BY-SA 4.0) contains human-verified
romanizations for 12 South Asian languages from Wikipedia text.
For each native-script word, multiple annotators wrote their own romanization,
giving us real attested spelling variant groups.

Reference
---------
Roark et al. (2020) "Processing South Asian Languages Written in the Latin Script:
the Dakshina Dataset." LREC 2020. https://aclanthology.org/2020.lrec-1.294

Usage
-----
# Load groups for one language and extend DEFAULT_RULES
from fliprate.dakshina import load_dakshina_groups, SUPPORTED_LANGUAGES
from fliprate.rules import DEFAULT_RULES

# Single language
rules = DEFAULT_RULES.extended(groups=load_dakshina_groups("ta"))

# Multiple languages
rules = DEFAULT_RULES.extended(
    groups=load_dakshina_groups("ta") + load_dakshina_groups("kn")
)

# All languages
rules = DEFAULT_RULES.extended(groups=load_dakshina_groups())

# Pre-built RuleSet for a single language
from fliprate.dakshina import dakshina_rules
rules = dakshina_rules("hi")          # Hindi only
rules = dakshina_rules("hi", "ur")    # Hindi + Urdu
rules = dakshina_rules()              # all 12 languages

Dataset location
----------------
The loader looks for dakshina_dataset_v1.0/ relative to the current working
directory first, then next to the fliprate package source.
Override with the FLIPRATE_DAKSHINA_PATH environment variable.
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .rules import RuleSet, DEFAULT_RULES, word_key

# ---------------------------------------------------------------------------
SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "ur": "Urdu",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "sd": "Sindhi",
    "si": "Sinhala",
}

_RUN_RE = re.compile(r"(.)\1+")
_PURE_ALPHA = re.compile(r"^[a-z]{2,}$")

# Words that reverse meaning — groups containing them go to a review file
_NEGATION_KEYS = frozenset({
    "nahi", "nhi", "nahin", "nahee", "nako", "nko", "mat", "maat",
    "not", "no", "never",
    "ila", "ilai", "beda", "ledu", "vadu", "kadu", "venda", "ala",
    "noy", "nei", "nathi",
})

# Very short / ambiguous tokens — never include
_SKIP = frozenset({"na", "nai", "he", "to", "ho", "re", "ki", "ko", "po",
                   "la", "le", "se", "ka", "ke", "sa", "ne", "pe", "me",
                   "aa", "ee", "oo"})


def _find_dakshina() -> Optional[Path]:
    """Find the Dakshina dataset directory."""
    env = os.environ.get("FLIPRATE_DAKSHINA_PATH")
    if env:
        p = Path(env)
        if p.exists():
            return p

    candidates = [
        Path.cwd() / "dakshina_dataset_v1.0",
        Path(__file__).parent.parent.parent / "dakshina_dataset_v1.0",
        Path(__file__).parent.parent.parent.parent / "dakshina_dataset_v1.0",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _read_tsv(path: Path) -> Dict[str, Dict[str, int]]:
    """Read one lexicon TSV. Returns {native_word: {romanization: count}}."""
    data: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            native, roman, count_str = parts[0], parts[1].strip().lower(), parts[2]
            if not roman:
                continue
            try:
                data[native][roman] += int(count_str)
            except ValueError:
                pass
    return data


def _load_lang(dakshina_root: Path, lang: str) -> Dict[str, Dict[str, int]]:
    """Load + merge train/dev/test lexicons for one language."""
    merged: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lex_dir = dakshina_root / lang / "lexicons"
    for split in ("train", "dev", "test"):
        tsv = lex_dir / f"{lang}.translit.sampled.{split}.tsv"
        if not tsv.exists():
            continue
        for native, romans in _read_tsv(tsv).items():
            for roman, count in romans.items():
                merged[native][roman] += count
    return merged


def _extract_groups(
    lang_data: Dict[str, Dict[str, int]],
    min_total: int = 3,
    min_variants: int = 2,
) -> List[Tuple[str, ...]]:
    """
    Extract spelling-variant groups from Dakshina lexicon data.

    Only returns groups where:
    - All members are purely ASCII alphabetic (no digits, no punctuation)
    - At least `min_variants` distinct word_keys exist (real variation, not just stretch)
    - Total attestation >= `min_total`
    - No member is in _SKIP (very short / ambiguous)
    - No negation words (those need separate human review)
    - No cross-native-word key conflicts
    """
    groups: List[Tuple[str, ...]] = []
    registered_keys: Dict[str, int] = {}  # word_key -> group_id

    for native, roman_counts in lang_data.items():
        # Keep only clean ASCII romanizations
        clean = {
            r: c for r, c in roman_counts.items()
            if _PURE_ALPHA.match(r) and r not in _SKIP
        }
        if len(clean) < min_variants:
            continue

        total = sum(clean.values())
        if total < min_total:
            continue

        # Sort: most attested first (= canonical)
        sorted_r = sorted(clean.items(), key=lambda x: (-x[1], x[0]))
        members = [r for r, _ in sorted_r]

        # Deduplicate by word_key within this group
        seen: Dict[str, str] = {}
        deduped = []
        for r in members:
            k = word_key(r)
            if k not in seen:
                seen[k] = r
                deduped.append(r)

        if len(deduped) < min_variants:
            continue

        group = tuple(deduped)
        group_keys = {word_key(r) for r in group}

        # Skip negation-containing groups
        if group_keys & _NEGATION_KEYS:
            continue

        # Check cross-native-word conflicts — first-wins
        gid = len(groups)
        conflict = False
        for k in group_keys:
            existing = registered_keys.get(k)
            if existing is not None and existing != gid:
                conflict = True
                break
        if conflict:
            continue

        for k in group_keys:
            registered_keys[k] = gid
        groups.append(group)

    return groups


# ---------------------------------------------------------------------------
# Cache — so repeated calls don't re-read disk
# ---------------------------------------------------------------------------
_cache: Dict[str, List[Tuple[str, ...]]] = {}


def load_dakshina_groups(
    *languages: str,
    min_total: int = 3,
    min_variants: int = 2,
    dakshina_path: Optional[str] = None,
) -> List[Tuple[str, ...]]:
    """
    Load Dakshina spelling variant groups for the given languages.

    Parameters
    ----------
    *languages : str
        Language codes to load (e.g. "hi", "ta", "te").
        Pass no arguments to load all 12 supported languages.
    min_total : int
        Minimum total attestation count to include a group (default 3).
    min_variants : int
        Minimum number of distinct romanizations in a group (default 2).
    dakshina_path : str, optional
        Override the auto-detected Dakshina directory path.

    Returns
    -------
    List of tuples — ready to pass to RuleSet() or extended().

    Raises
    ------
    FileNotFoundError
        If the Dakshina dataset directory cannot be found.
    """
    langs = list(languages) if languages else list(SUPPORTED_LANGUAGES.keys())
    unknown = [l for l in langs if l not in SUPPORTED_LANGUAGES]
    if unknown:
        raise ValueError(f"Unknown language codes: {unknown}. "
                         f"Supported: {list(SUPPORTED_LANGUAGES.keys())}")

    root = Path(dakshina_path) if dakshina_path else _find_dakshina()
    if root is None:
        raise FileNotFoundError(
            "Dakshina dataset not found. Download from "
            "https://github.com/google-research-datasets/dakshina "
            "and set FLIPRATE_DAKSHINA_PATH=/path/to/dakshina_dataset_v1.0"
        )

    all_groups: List[Tuple[str, ...]] = []
    for lang in langs:
        cache_key = f"{lang}:{root}:{min_total}:{min_variants}"
        if cache_key not in _cache:
            lang_dir = root / lang
            if not lang_dir.exists():
                print(f"[fliprate.dakshina] Warning: {lang_dir} not found, skipping",
                      file=sys.stderr)
                _cache[cache_key] = []
                continue
            data = _load_lang(root, lang)
            _cache[cache_key] = _extract_groups(data, min_total, min_variants)
        all_groups.extend(_cache[cache_key])

    return all_groups


def dakshina_rules(
    *languages: str,
    base: Optional[RuleSet] = None,
    min_total: int = 3,
    min_variants: int = 2,
    dakshina_path: Optional[str] = None,
) -> RuleSet:
    """
    Build a RuleSet that combines `base` groups with Dakshina groups.

    Uses RuleSet() directly (not extended()) for performance with large group lists.

    Parameters
    ----------
    *languages : str
        Language codes. Pass none for all 12 languages.
    base : RuleSet, optional
        Base rules to combine with (default: DEFAULT_RULES).
    min_total, min_variants : int
        Forwarded to load_dakshina_groups.
    dakshina_path : str, optional
        Override auto-detected Dakshina path.

    Returns
    -------
    RuleSet

    Examples
    --------
    >>> from fliprate.dakshina import dakshina_rules
    >>> rules = dakshina_rules("ta", "kn")      # Tamil + Kannada
    >>> fr.test(my_model, dataset, rules=rules)
    """
    groups = load_dakshina_groups(
        *languages,
        min_total=min_total,
        min_variants=min_variants,
        dakshina_path=dakshina_path,
    )
    b = base if base is not None else DEFAULT_RULES
    # Build directly: base groups first, then Dakshina groups
    # RuleSet constructor handles dedup and conflict-skipping automatically
    combined = list(b.groups) + groups
    return RuleSet(
        groups=combined,
        lexicon=list(b.lexicon),
        negations=b.negations,
        ambiguous=dict(b.ambiguous),
    )


def list_languages() -> Dict[str, int]:
    """
    Return a dict of {lang_code: group_count} for all available languages.
    Useful for exploring what's available before loading.
    """
    root = _find_dakshina()
    if root is None:
        return {}
    result = {}
    for lang in SUPPORTED_LANGUAGES:
        groups = load_dakshina_groups(lang, dakshina_path=str(root))
        result[lang] = len(groups)
    return result
