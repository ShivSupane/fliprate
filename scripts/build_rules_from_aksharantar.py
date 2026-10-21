"""
Build FlipRate SAFE_GROUPS from the Aksharantar dataset.

Aksharantar (AI4Bharat, 2022) contains 24M+ native-script → romanization
word pairs for 21 Indian languages. Unlike Dakshina, each line has exactly
one romanization for one native word, but the same native word appears
multiple times across different sources with different romanizations.

Grouping strategy
-----------------
1. Group all romanizations by native_word.
2. Keep groups where ≥ 2 distinct romanizations exist (real spelling variation).
3. Apply the same quality filters as the Dakshina extractor.
4. Write output to data/rules_from_aksharantar/.

Output
------
  aksharantar_groups.py   — Python module with per-language group lists
  aksharantar_approved.csv — all approved groups (spellings col for RuleSet.from_csv)
  aksharantar_review_negations.csv — negation-containing groups for human review

Usage
-----
  python scripts/build_rules_from_aksharantar.py
  python scripts/build_rules_from_aksharantar.py --langs hin tam tel kan
  python scripts/build_rules_from_aksharantar.py --min-variants 2 --min-distinct-sources 1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

BASE = Path(__file__).parent.parent
OUT_DIR = BASE / "data" / "rules_from_aksharantar"

LANGUAGES = {
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

# Negation word_keys — groups containing these go to review
NEGATION_KEYS = frozenset({
    "nahi", "nhi", "nahin", "nahee", "nako", "nko", "mat", "maat",
    "not", "no", "never",
    "ila", "ilai", "beda", "ledu", "vadu", "kadu", "venda", "ala",
    "noy", "nei", "nathi",
})

# Short/ambiguous tokens — never include
SKIP = frozenset({
    "na", "nai", "he", "to", "ho", "re", "ki", "ko", "po",
    "la", "le", "se", "ka", "ke", "sa", "ne", "pe", "me",
    "aa", "ee", "oo", "ha", "hi", "hu",
})

_RUN_RE = re.compile(r"(.)\1+")
_PURE_ALPHA = re.compile(r"^[a-z]{2,30}$")


def word_key(word: str) -> str:
    return _RUN_RE.sub(r"\1", word.lower())


def is_clean(roman: str) -> bool:
    return bool(_PURE_ALPHA.match(roman.lower())) and roman not in SKIP


def load_lang(lang_dir: Path, lang: str) -> Dict[str, Dict[str, int]]:
    """
    Returns {native_word: {romanization: count}}.
    Count = number of times this (native, roman) pair appeared.
    """
    data: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for split in ("train", "valid", "test"):
        f = lang_dir / f"{lang}_{split}.json"
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


def extract_groups(
    lang_data: Dict[str, Dict[str, int]],
    min_variants: int = 2,
) -> Tuple[List[Tuple[str, ...]], List[Tuple[str, ...]]]:
    """
    Returns (approved_groups, review_groups).
    """
    approved: List[Tuple[str, ...]] = []
    review: List[Tuple[str, ...]] = []
    registered: Dict[str, int] = {}  # word_key -> group_id

    for native, roman_counts in lang_data.items():
        # Keep only clean ASCII romanizations
        clean = {r: c for r, c in roman_counts.items() if is_clean(r)}
        if len(clean) < min_variants:
            continue

        # Sort: most frequent first (= canonical)
        sorted_r = sorted(clean.items(), key=lambda x: (-x[1], x[0]))
        members = [r for r, _ in sorted_r]

        # Deduplicate by word_key
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

        # Cross-native-word key conflict check (first-wins)
        gid = len(approved) + len(review)
        conflict = False
        existing_gid = None
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

        # Assign
        for k in group_keys:
            registered[k] = gid

        # Negation check
        if group_keys & NEGATION_KEYS:
            review.append(group)
        else:
            approved.append(group)

    return approved, review


def format_python(groups: List[Tuple[str, ...]], lang_name: str, lang_code: str) -> str:
    lines = [
        f"\n# {'='*75}",
        f"# {lang_name.upper()} ({lang_code}) — auto-extracted from Aksharantar",
        f"# {len(groups)} groups from AI4Bharat Aksharantar dataset",
        f"# Source: https://github.com/AI4Bharat/Aksharantar (CC BY 4.0)",
        f"# MEDIUM confidence — spot-check with a fluent speaker before production.",
        f"# {'='*75}",
        f"{lang_code.upper()}_AKSHARANTAR_GROUPS: List[Tuple[str, ...]] = [",
    ]
    for group in sorted(groups, key=lambda g: g[0]):
        members = ", ".join(f'"{r}"' for r in group)
        lines.append(f"    ({members}),")
    lines.append("]")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", nargs="*", default=list(LANGUAGES.keys()))
    parser.add_argument("--min-variants", type=int, default=2)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_python = [
        '"""',
        "Auto-generated FlipRate spelling groups from Aksharantar.",
        "Source: AI4Bharat Aksharantar dataset (CC BY 4.0)",
        "https://github.com/AI4Bharat/Aksharantar",
        "",
        "Usage:",
        "    from fliprate.data.aksharantar_groups import HIN_AKSHARANTAR_GROUPS",
        "    from fliprate.aksharantar import aksharantar_rules",
        "    rules = aksharantar_rules('hin', 'tam')",
        '"""',
        "from __future__ import annotations",
        "from typing import List, Tuple",
    ]

    all_csv: List[dict] = []
    review_csv: List[dict] = []
    summary = []

    for lang in args.langs:
        if lang not in LANGUAGES:
            print(f"  Skipping unknown: {lang}", file=sys.stderr)
            continue
        lang_name = LANGUAGES[lang]
        lang_dir = BASE / lang
        if not lang_dir.exists():
            print(f"  {lang}: directory not found, skipping")
            continue

        print(f"  Loading {lang_name} ({lang})...", end=" ", flush=True)
        data = load_lang(lang_dir, lang)
        print(f"{len(data):,} native words", end=" -> ", flush=True)

        approved, review = extract_groups(data, args.min_variants)
        print(f"{len(approved):,} approved, {len(review):,} review")

        all_python.append(format_python(approved, lang_name, lang))

        for group in sorted(approved, key=lambda g: g[0]):
            all_csv.append({
                "language": lang, "language_name": lang_name,
                "canonical": group[0], "variants": "|".join(group[1:]),
                "spellings": "|".join(group), "status": "", "notes": "",
            })
        for group in sorted(review, key=lambda g: g[0]):
            review_csv.append({
                "language": lang, "language_name": lang_name,
                "canonical": group[0], "variants": "|".join(group[1:]),
                "spellings": "|".join(group), "status": "", "notes": "",
            })
        summary.append({"lang": lang, "name": lang_name,
                        "words": len(data), "approved": len(approved), "review": len(review)})

    # Write Python module
    all_python.append("\n\nAKSHARANTAR_LANGUAGE_GROUPS = {")
    for lang in args.langs:
        if lang in LANGUAGES:
            all_python.append(f'    "{lang}": {lang.upper()}_AKSHARANTAR_GROUPS,')
    all_python.append("}")
    all_python.append("\nALL_AKSHARANTAR_GROUPS = sum(AKSHARANTAR_LANGUAGE_GROUPS.values(), [])")

    py_out = OUT_DIR / "aksharantar_groups.py"
    with open(py_out, "w", encoding="utf-8") as f:
        f.write("\n".join(all_python))
    print(f"\n  Python module: {py_out}")

    # Write CSVs
    fields = ["language", "language_name", "canonical", "variants", "spellings", "status", "notes"]
    csv_out = OUT_DIR / "aksharantar_approved.csv"
    with open(csv_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(all_csv)
    print(f"  Approved CSV: {csv_out} ({len(all_csv):,} groups)")

    rev_out = OUT_DIR / "aksharantar_review_negations.csv"
    with open(rev_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(review_csv)
    print(f"  Review CSV:   {rev_out} ({len(review_csv):,} groups)")

    print("\n" + "="*65)
    print(f"{'Lang':<6} {'Name':<12} {'Words':>10} {'Groups':>8} {'Review':>8}")
    print("-"*65)
    total_g = total_r = 0
    for r in summary:
        print(f"{r['lang']:<6} {r['name']:<12} {r['words']:>10,} {r['approved']:>8,} {r['review']:>8,}")
        total_g += r["approved"]; total_r += r["review"]
    print("-"*65)
    print(f"{'TOTAL':<6} {'':<12} {sum(r['words'] for r in summary):>10,} {total_g:>8,} {total_r:>8,}")
    print("="*65)
    print(f"\nQuick start:")
    print(f"  from fliprate.aksharantar import aksharantar_rules")
    print(f"  rules = aksharantar_rules('tam', 'tel')  # Tamil + Telugu")


if __name__ == "__main__":
    print("Building FlipRate rules from Aksharantar dataset...\n")
    main()
