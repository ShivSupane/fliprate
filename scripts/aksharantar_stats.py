"""Quick stats on the Aksharantar dataset."""
import json, os, sys
from pathlib import Path

BASE = Path(__file__).parent.parent

langs = [
    ("hin","Hindi"), ("guj","Gujarati"), ("tam","Tamil"), ("tel","Telugu"),
    ("kan","Kannada"), ("mal","Malayalam"), ("mar","Marathi"),
    ("pan","Punjabi"), ("ben","Bengali"), ("urd","Urdu"),
    ("ori","Odia"), ("asm","Assamese"), ("kas","Kashmiri"),
    ("mni","Manipuri"), ("brx","Bodo"), ("kok","Konkani"),
    ("mai","Maithili"), ("nep","Nepali"), ("san","Sanskrit"),
    ("sid","Siddhi"), ("doi","Dogri"),
]

total_pairs = 0
for code, name in langs:
    f = BASE / code / f"{code}_train.json"
    if not f.exists():
        print(f"{code}: NOT FOUND")
        continue
    count = 0
    sources = {}
    sample = []
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            count += 1
            s = d.get("source", "?")
            sources[s] = sources.get(s, 0) + 1
            if count <= 2:
                sample.append((d["native word"], d["english word"], s))
    top = sorted(sources.items(), key=lambda x: -x[1])[:3]
    total_pairs += count
    print(f"{code} ({name}): {count:,} pairs")
    print(f"  top sources: {top}")
    for nw, ew, src in sample:
        print(f"  '{nw}' -> '{ew}'  [{src}]")
    print()

print(f"TOTAL: {total_pairs:,} pairs across {len(langs)} languages")
