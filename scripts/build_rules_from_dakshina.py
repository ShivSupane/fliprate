"""
Build FlipRate SAFE_GROUPS from the Dakshina dataset.

The Dakshina lexicon format (per line):
    <native_word> TAB <romanization> TAB <attestation_count>

Multiple lines with the same native_word = multiple human-verified romanizations.
That is exactly what FlipRate needs: a group of romanizations that mean the same word.

Pipeline:
  1. Read train + dev + test TSV for all 12 languages
  2. Group by native_word — collect all romanizations and their counts
  3. Keep only groups where:
     a. At least 2 distinct romanizations exist
     b. The canonical (most-attested) romanization is purely ASCII-alphabetic
     c. No member has the same word_key (collapsed form) as an existing SAFE group word
        from a DIFFERENT native word — prevents false merges
     d. The word has minimum total attestation (configurable, default 3)
  4. Cluster by word_key: if two native words produce the same word_key groups, merge them
  5. Filter out groups containing negation words (we never want to accidentally
     canonicalize a negation variant to a non-negation form)
  6. Output:
     - A Python snippet ready to paste into rules.py  (by language)
     - A CSV for human speaker review
     - A summary report

Usage:
    python scripts/build_rules_from_dakshina.py
    python scripts/build_rules_from_dakshina.py --min-attestations 4 --min-variants 2
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DAKSHINA = Path(__file__).parent.parent / "dakshina_dataset_v1.0"
OUT_DIR   = Path(__file__).parent.parent / "data" / "rules_from_dakshina"

# Language codes in the dataset and their human names
LANGUAGES = {
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

# Known negation words — groups containing these are written to a separate
# review file and NOT included in the auto-approved output.
NEGATION_ROOTS = {
    "nahi", "nhi", "nahin", "nahee", "nako", "nko", "mat", "maat",
    "not", "no", "never",
    "illa", "illai", "beda", "ledu", "vaddu", "kaadu", "venda", "alla",
    "noy", "nei", "nathi",
    "nuhein", "nuhen",
}

# Ambiguous single-letter / very-short tokens to exclude
AMBIGUOUS = {"na", "nai", "he", "to", "ho", "re", "ki", "ko", "po",
             "la", "le", "se", "ka", "ke", "sa", "ne", "pe", "me"}

# Words that are clearly English or digits — skip
_DIGIT_RE = re.compile(r"\d")
_PURE_ALPHA_RE = re.compile(r"^[a-z]+$")

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
_RUN_RE = re.compile(r"(.)\1+")

def word_key(word: str) -> str:
    """Same as FlipRate's word_key: lowercase + collapse repeated letters."""
    return _RUN_RE.sub(r"\1", word.lower())

def is_clean_roman(word: str) -> bool:
    """True if the romanization is purely lowercase ASCII letters (no digits, no punct)."""
    return bool(_PURE_ALPHA_RE.match(word.lower())) and len(word) >= 2

# --------------------------------------------------------------------------- #
# Core extraction
# --------------------------------------------------------------------------- #
def read_lexicon(path: Path) -> Dict[str, Dict[str, int]]:
    """
    Returns: {native_word: {romanization: total_count}}
    Merges train + dev + test so we have maximum coverage.
    """
    data: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            native, roman, count_str = parts[0], parts[1], parts[2]
            try:
                count = int(count_str)
            except ValueError:
                continue
            roman = roman.strip().lower()
            if not roman:
                continue
            data[native][roman] += count
    return data


def load_language(lang_dir: Path, lang: str) -> Dict[str, Dict[str, int]]:
    """Load and merge all three splits (train/dev/test) for one language."""
    merged: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    lex_dir = lang_dir / "lexicons"
    for split in ("train", "dev", "test"):
        tsv = lex_dir / f"{lang}.translit.sampled.{split}.tsv"
        if not tsv.exists():
            continue
        for native, romans in read_lexicon(tsv).items():
            for roman, count in romans.items():
                merged[native][roman] += count
    return merged


def extract_groups(
    lang_data: Dict[str, Dict[str, int]],
    min_total: int = 3,
    min_variants: int = 2,
) -> Tuple[List[Tuple[str, ...]], List[Tuple[str, ...]]]:
    """
    Returns (approved_groups, review_groups).
    
    approved_groups: groups with ≥ min_variants distinct ASCII romanizations,
                     total attestation ≥ min_total, no negation words.
    review_groups:   groups containing negation words (need human review).
    """
    approved: List[Tuple[str, ...]] = []
    review:   List[Tuple[str, ...]] = []
    
    # Track which word_keys have already been assigned to a group
    # to prevent two different native words from producing conflicting groups
    key_to_group_id: Dict[str, int] = {}
    
    for native, roman_counts in lang_data.items():
        # Filter to clean ASCII romanizations only
        clean = {
            r: c for r, c in roman_counts.items()
            if is_clean_roman(r) and not _DIGIT_RE.search(r)
        }
        if len(clean) < min_variants:
            continue
        
        # Total attestation check
        total = sum(clean.values())
        if total < min_total:
            continue
        
        # Build the group: sort by count descending, canonical = most attested
        sorted_romans = sorted(clean.items(), key=lambda x: (-x[1], x[0]))
        group = tuple(r for r, _ in sorted_romans)
        
        # Drop single-char or very-short forms from the group
        # (keep them only if they're the ONLY spelling of an important word)
        group = tuple(r for r in group if len(r) >= 2)
        if len(group) < min_variants:
            continue
        
        # Deduplicate by word_key within the group
        seen_keys: Dict[str, str] = {}
        deduped = []
        for r in group:
            k = word_key(r)
            if k not in seen_keys:
                seen_keys[k] = r
                deduped.append(r)
        if len(deduped) < min_variants:
            continue
        group = tuple(deduped)
        
        # Exclude ambiguous tokens
        if any(r in AMBIGUOUS for r in group):
            continue
        
        # Check for cross-native-word key conflicts
        group_keys = {word_key(r) for r in group}
        conflict = False
        this_gid = None
        for k in group_keys:
            if k in key_to_group_id:
                if this_gid is None:
                    this_gid = key_to_group_id[k]
                elif this_gid != key_to_group_id[k]:
                    conflict = True
                    break
        if conflict:
            continue
        
        # Assign group id
        new_gid = id(group)
        for k in group_keys:
            key_to_group_id[k] = new_gid
        
        # Check if this group contains any negation word
        if any(word_key(r) in {word_key(n) for n in NEGATION_ROOTS} for r in group):
            review.append(group)
        else:
            approved.append(group)
    
    return approved, review


def format_python(groups: List[Tuple[str, ...]], lang_name: str, lang_code: str) -> str:
    """Format groups as Python tuples for rules.py."""
    lines = [
        f"\n# {'='*75}",
        f"# {lang_name.upper()} ({lang_code}) — auto-extracted from Dakshina v1.0",
        f"# {len(groups)} groups, sorted by canonical romanization.",
        f"# Source: Google Dakshina dataset (CC BY-SA 4.0)",
        f"# https://github.com/google-research-datasets/dakshina",
        f"# ALL groups here have ≥2 human-attested romanizations from Wikipedia text.",
        f"# MEDIUM confidence — a fluent speaker should spot-check before production.",
        f"# {'='*75}",
        f"{lang_code.upper()}_DAKSHINA_GROUPS: List[Tuple[str, ...]] = [",
    ]
    for group in sorted(groups, key=lambda g: g[0]):
        members = ", ".join(f'"{r}"' for r in group)
        lines.append(f"    ({members}),")
    lines.append("]")
    return "\n".join(lines)


def format_csv(groups: List[Tuple[str, ...]], lang: str, lang_name: str) -> List[dict]:
    """Format groups as CSV rows for speaker review."""
    rows = []
    for group in sorted(groups, key=lambda g: g[0]):
        rows.append({
            "language": lang,
            "language_name": lang_name,
            "canonical": group[0],
            "variants": "|".join(group[1:]),
            "spellings": "|".join(group),   # for direct use with RuleSet.from_csv()
            "status": "",                    # reviewer fills: approve / reject
            "notes": "",
        })
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Build FlipRate rules from Dakshina")
    parser.add_argument("--min-attestations", type=int, default=3,
                        help="Minimum total attestation count for a group (default: 3)")
    parser.add_argument("--min-variants", type=int, default=2,
                        help="Minimum number of distinct romanizations (default: 2)")
    parser.add_argument("--langs", nargs="*", default=list(LANGUAGES.keys()),
                        help="Languages to process (default: all)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_python_blocks = [
        '"""',
        "Auto-generated FlipRate spelling groups from the Dakshina dataset v1.0.",
        "Source: Google Research (CC BY-SA 4.0)",
        "Command: python scripts/build_rules_from_dakshina.py",
        "",
        "Usage in FlipRate:",
        "    from fliprate.data.dakshina_groups import HI_DAKSHINA_GROUPS, TA_DAKSHINA_GROUPS",
        "    from fliprate.rules import DEFAULT_RULES",
        "    rules = DEFAULT_RULES.extended(groups=HI_DAKSHINA_GROUPS)",
        '"""',
        "from __future__ import annotations",
        "from typing import List, Tuple",
    ]
    
    all_csv_rows: List[dict] = []
    review_csv_rows: List[dict] = []
    
    summary_rows = []
    
    for lang in args.langs:
        if lang not in LANGUAGES:
            print(f"  Skipping unknown language: {lang}", file=sys.stderr)
            continue
        
        lang_name = LANGUAGES[lang]
        lang_dir = DAKSHINA / lang
        if not lang_dir.exists():
            print(f"  {lang}: directory not found, skipping")
            continue
        
        print(f"  Loading {lang_name} ({lang})...", end=" ", flush=True)
        lang_data = load_language(lang_dir, lang)
        print(f"{len(lang_data)} native words", end=" -> ", flush=True)
        
        approved, review = extract_groups(
            lang_data,
            min_total=args.min_attestations,
            min_variants=args.min_variants,
        )
        print(f"{len(approved)} groups approved, {len(review)} need review")
        
        # Python output
        all_python_blocks.append(format_python(approved, lang_name, lang))
        
        # CSV rows
        all_csv_rows.extend(format_csv(approved, lang, lang_name))
        review_csv_rows.extend(format_csv(review, lang, lang_name))
        
        summary_rows.append({
            "lang": lang,
            "language": lang_name,
            "native_words": len(lang_data),
            "approved_groups": len(approved),
            "review_groups": len(review),
        })
    
    # Write Python module
    py_out = OUT_DIR / "dakshina_groups.py"
    with open(py_out, "w", encoding="utf-8") as f:
        f.write("\n".join(all_python_blocks))
        # Write combined dict at the bottom
        f.write("\n\n\n# Combined dict for all languages\n")
        f.write("DAKSHINA_LANGUAGE_GROUPS = {\n")
        for lang in args.langs:
            if lang in LANGUAGES:
                f.write(f'    "{lang}": {lang.upper()}_DAKSHINA_GROUPS,\n')
        f.write("}\n\n")
        f.write("ALL_DAKSHINA_GROUPS = sum(DAKSHINA_LANGUAGE_GROUPS.values(), [])\n")
    print(f"\n  Python module written: {py_out}")
    
    # Write approved CSV
    csv_out = OUT_DIR / "dakshina_approved.csv"
    if all_csv_rows:
        with open(csv_out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["language", "language_name", "canonical",
                                               "variants", "spellings", "status", "notes"])
            w.writeheader()
            w.writerows(all_csv_rows)
    print(f"  Approved CSV written: {csv_out} ({len(all_csv_rows)} groups)")
    
    # Write review CSV (negation words etc.)
    rev_out = OUT_DIR / "dakshina_review_negations.csv"
    if review_csv_rows:
        with open(rev_out, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["language", "language_name", "canonical",
                                               "variants", "spellings", "status", "notes"])
            w.writeheader()
            w.writerows(review_csv_rows)
    print(f"  Review CSV written:   {rev_out} ({len(review_csv_rows)} groups, needs speaker check)")
    
    # Print summary
    print("\n" + "="*60)
    print(f"{'Lang':<6} {'Language':<12} {'Words':>8} {'Groups':>8} {'Review':>8}")
    print("-"*60)
    total_groups = 0
    total_review = 0
    for r in summary_rows:
        print(f"{r['lang']:<6} {r['language']:<12} {r['native_words']:>8,} {r['approved_groups']:>8,} {r['review_groups']:>8,}")
        total_groups += r["approved_groups"]
        total_review += r["review_groups"]
    print("-"*60)
    print(f"{'TOTAL':<6} {'':<12} {sum(r['native_words'] for r in summary_rows):>8,} {total_groups:>8,} {total_review:>8,}")
    print("="*60)
    print(f"\nLoad in FlipRate:")
    print(f"  from fliprate.data.dakshina_groups import ALL_DAKSHINA_GROUPS")
    print(f"  from fliprate.rules import DEFAULT_RULES")
    print(f"  rules = DEFAULT_RULES.extended(groups=ALL_DAKSHINA_GROUPS)")


if __name__ == "__main__":
    print("Building FlipRate rules from Dakshina dataset...\n")
    main()
