"""Find remaining TODOs and placeholders in the project."""
import pathlib

ROOT = pathlib.Path(__file__).parent.parent
SKIP = {".git", "__pycache__", ".pytest_cache", "dakshina_dataset_v1.0",
        "Aksharantar_All_Languages", "data", "dakshina_dataset_v1.0.tar",
        "Aksharantar_All_Languages.zip"}
PATTERNS = ["YOUR NAME", "YOUR-USERNAME", "your@example", "you@example",
            "AUTHOR NAME", "shiv@example.com", "TODO", "FIXME", "DRAFT"]
EXTS = {".py", ".toml", ".md", ".yml", ".yaml", ".txt"}

found = []
for f in ROOT.rglob("*"):
    if not f.is_file():
        continue
    if any(s in f.parts for s in SKIP):
        continue
    if f.suffix not in EXTS:
        continue
    try:
        content = f.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    for i, line in enumerate(content.splitlines(), 1):
        for p in PATTERNS:
            if p.lower() in line.lower():
                rel = f.relative_to(ROOT)
                found.append((str(rel), i, line.strip()))
                break

if not found:
    print("No remaining placeholders found!")
else:
    for path, ln, line in found:
        print(f"{path}:{ln}  {line}")
