"""Quick end-to-end verification of both dataset integrations."""
import time, sys
sys.path.insert(0, "src")
import fliprate as fr
from fliprate.rules import word_key

print("=== DEFAULT_RULES ===")
print(f"Groups: {len(fr.DEFAULT_RULES.groups):,}  Keys: {len(fr.DEFAULT_RULES.key_to_canon):,}")

print("\n=== Dakshina per language (2-letter codes) ===")
for lang in ["hi","ta","te","kn","ml","mr","bn","gu","pa","ur"]:
    try:
        t0 = time.time()
        r = fr.dakshina_rules(lang)
        print(f"  {lang}   {len(r.groups):>7,} groups  {len(r.key_to_canon):>8,} keys  {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  {lang}   ERROR: {e}")

print("\n=== Aksharantar per language (3-letter codes) ===")
for lang in ["hin","tam","tel","kan","mal","mar","ben","pan","urd","guj"]:
    try:
        t0 = time.time()
        r = fr.aksharantar_rules(lang)
        print(f"  {lang}  {len(r.groups):>7,} groups  {len(r.key_to_canon):>8,} keys  {time.time()-t0:.2f}s")
    except Exception as e:
        print(f"  {lang}  ERROR: {e}")

print("\n=== Combined Dakshina + Aksharantar (Hindi) ===")
t0 = time.time()
dak = fr.load_dakshina_groups("hi")
aksh = fr.load_aksharantar_groups("hin")
from fliprate.rules import RuleSet, DEFAULT_RULES
combined = RuleSet(
    groups=list(DEFAULT_RULES.groups) + dak + aksh,
    lexicon=list(DEFAULT_RULES.lexicon),
    negations=DEFAULT_RULES.negations,
    ambiguous=dict(DEFAULT_RULES.ambiguous),
)
print(f"  {len(combined.groups):,} groups  {len(combined.key_to_canon):,} keys  {time.time()-t0:.2f}s")

print("\n=== All tests ===")
import subprocess, sys as _sys
r = subprocess.run([_sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"],
                   capture_output=True, text=True)
last = r.stdout.strip().split("\n")[-1]
print(f"  {last}")
