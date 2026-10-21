"""
Loader for the Aksharantar dataset spelling variant groups.

The Aksharantar dataset (AI4Bharat, 2022, CC BY 4.0) contains 24M+ native-script
→ romanization word pairs for 21 Indian languages, sourced from IndicCorp,
Samanantar, Dakshina, and other curated Indian language corpora.

Reference
---------
Mittal et al. (2022) "Aksharantar: Towards Building Open Transliteration Tools
for the Next Billion Users." https://arxiv.org/abs/2205.03018

Dataset location
----------------
The loader looks for language directories (hin/, tam/, etc.) relative to the
current working directory first, then next to the fliprate package source.
Override with FLIPRATE_AKSHARANTAR_PATH environment variable.

Usage
-----
    from fliprate.aksharantar import aksharantar_rules, load_aksharantar_groups

    # Single language
    rules = aksharantar_rules("ta")

    # Multiple languages
    rules = aksharantar_rules("ta", "kn", "te")

    # All 21 languages
    rules = aksharantar_rules()

    # Just the groups (for manual inspection or further filtering)
    groups = load_aksharantar_groups("hi")
    print(groups[:5])
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .rules import RuleSet, DEFAULT_RULES, word_key

SUPPORTED_LANGUAGES = {
    "hin": "Hindi",
    "guj": "Gujarati",
    "tam": "Tamil",
    "tel": "Telugu",
    "kan": "Kannada",
    "mal": "Malayalam",
    "mar": "Marathi",
    "pan": "Punjabi",
    "ben": "Bengali",
    "urd": "Urdu",
    "ori": "Odia",
    "asm": "Assamese",
    "kas": "Kashmiri",
    "kok": "Konkani",
    "mai": "Maithili",
    "mni": "Manipuri",
    "nep": "Nepali",
    "san": "Sanskrit",
    "brx": "Bodo",
    "sid": "Sindhi",
}

_RUN_RE = re.compile(r"(.)\1+")
_PURE_ALPHA = re.compile(r"^[a-z]{2,30}$")
_NEGATION_KEYS = frozenset({
    "nahi", "nhi", "nahin", "nahee", "nako", "nko", "mat", "maat",
    "not", "no", "never",
    "ila", "ilai", "beda", "ledu", "vadu", "kadu", "venda", "ala",
    "noy", "nei", "nathi",
})
_SKIP = frozenset({
    "na", "nai", "he", "to", "ho", "re", "ki", "ko", "po",
    "la", "le", "se", "ka", "ke", "sa", "ne", "pe", "me",
    "aa", "ee", "oo", "ha", "hi", "hu",
})


def _find_root() -> Optional[Path]:
    env = os.environ.get("FLIPRATE_AKSHARANTAR_PATH")
    if env:
        p = Path(env)
        if p.exists():
            return p
    candidates = [
        # New consolidated folder first
        Path.cwd() / "Aksharantar_All_Languages",
        Path(__file__).parent.parent.parent / "Aksharantar_All_Languages",
        Path(__file__).parent.parent.parent.parent / "Aksharantar_All_Languages",
        # Legacy: language folders directly in cwd
        Path.cwd(),
        Path(__file__).parent.parent.parent,
        Path(__file__).parent.parent.parent.parent,
    ]
    for c in candidates:
        if (c / "hin").exists() and (c / "tam").exists():
            return c
    return None


def _load_lang(root: Path, lang: str) -> Dict[str, Dict[str, int]]:
    data: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split in ("train", "valid", "test"):
        f = root / lang / f"{lang}_{split}.json"
        if not f.exists():
            continue
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                native = d.get("native word", "").strip()
                roman = d.get("english word", "").strip().lower()
                if native and roman:
                    data[native][roman] += 1
    return data


def _extract(
    lang_data: Dict[str, Dict[str, int]],
    min_variants: int = 2,
) -> List[Tuple[str, ...]]:
    groups: List[Tuple[str, ...]] = []
    registered: Dict[str, int] = {}

    for native, roman_counts in lang_data.items():
        clean = {r: c for r, c in roman_counts.items()
                 if _PURE_ALPHA.match(r) and r not in _SKIP}
        if len(clean) < min_variants:
            continue

        sorted_r = sorted(clean.items(), key=lambda x: (-x[1], x[0]))
        members = [r for r, _ in sorted_r]

        seen: Dict[str, bool] = {}
        deduped = []
        for r in members:
            k = word_key(r)
            if k not in seen:
                seen[k] = True
                deduped.append(r)

        if len(deduped) < min_variants:
            continue

        group = tuple(deduped)
        group_keys = {word_key(r) for r in group}

        # Skip negation groups entirely (conservative)
        if group_keys & _NEGATION_KEYS:
            continue

        # Cross-native-word key conflict — first-wins
        gid = len(groups)
        existing_gid = None
        conflict = False
        for k in group_keys:
            existing = registered.get(k)
            if existing is not None:
                if existing_gid is None:
                    existing_gid = existing
                elif existing_gid != existing:
                    conflict = True
                    break
        if conflict:
            continue

        for k in group_keys:
            registered[k] = gid
        groups.append(group)

    return groups


# Cache
_cache: Dict[str, List[Tuple[str, ...]]] = {}


def load_aksharantar_groups(
    *languages: str,
    min_variants: int = 2,
    aksharantar_path: Optional[str] = None,
) -> List[Tuple[str, ...]]:
    """
    Load Aksharantar spelling variant groups for the given languages.

    Parameters
    ----------
    *languages : str
        Language codes (e.g. "hin", "tam", "tel").
        Pass no arguments to load all 21 supported languages.
    min_variants : int
        Minimum number of distinct romanizations per group (default 2).
    aksharantar_path : str, optional
        Override the auto-detected root directory.

    Returns
    -------
    List of tuples — ready to pass to RuleSet() or dakshina_rules().

    Raises
    ------
    FileNotFoundError
        If the Aksharantar language directories cannot be found.
    """
    langs = list(languages) if languages else list(SUPPORTED_LANGUAGES.keys())
    unknown = [l for l in langs if l not in SUPPORTED_LANGUAGES]
    if unknown:
        raise ValueError(f"Unknown language codes: {unknown}. "
                         f"Supported: {list(SUPPORTED_LANGUAGES.keys())}")

    root = Path(aksharantar_path) if aksharantar_path else _find_root()
    if root is None:
        raise FileNotFoundError(
            "Aksharantar language directories not found. "
            "Set FLIPRATE_AKSHARANTAR_PATH to the folder containing hin/, tam/, etc."
        )

    all_groups: List[Tuple[str, ...]] = []
    for lang in langs:
        cache_key = f"aksh:{lang}:{root}:{min_variants}"
        if cache_key not in _cache:
            lang_dir = root / lang
            if not lang_dir.exists():
                print(f"[fliprate.aksharantar] Warning: {lang_dir} not found, skipping",
                      file=sys.stderr)
                _cache[cache_key] = []
                continue
            data = _load_lang(root, lang)
            _cache[cache_key] = _extract(data, min_variants)
        all_groups.extend(_cache[cache_key])

    return all_groups


def aksharantar_rules(
    *languages: str,
    base: Optional[RuleSet] = None,
    min_variants: int = 2,
    aksharantar_path: Optional[str] = None,
) -> RuleSet:
    """
    Build a RuleSet combining `base` with Aksharantar groups.

    Parameters
    ----------
    *languages : str
        Language codes. Pass none for all 21 languages.
    base : RuleSet, optional
        Base rules to combine with (default: DEFAULT_RULES).
    min_variants : int
        Minimum distinct romanizations per group (default 2).
    aksharantar_path : str, optional
        Override auto-detected root path.

    Returns
    -------
    RuleSet

    Examples
    --------
    >>> from fliprate.aksharantar import aksharantar_rules
    >>> rules = aksharantar_rules("tam", "kn")
    >>> fr.test(my_model, dataset, rules=rules)

    Notes
    -----
    For the best coverage use both datasets:
    >>> from fliprate.dakshina import dakshina_rules
    >>> rules = dakshina_rules("ta").extended(
    ...     groups=load_aksharantar_groups("tam")
    ... )
    """
    groups = load_aksharantar_groups(
        *languages,
        min_variants=min_variants,
        aksharantar_path=aksharantar_path,
    )
    b = base if base is not None else DEFAULT_RULES
    combined = list(b.groups) + groups
    return RuleSet(
        groups=combined,
        lexicon=list(b.lexicon),
        negations=b.negations,
        ambiguous=dict(b.ambiguous),
    )
