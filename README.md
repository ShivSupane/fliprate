# FlipRate

**Does your NLP model change its answer when the same message is typed differently?**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://github.com/ShivSupane/fliprate/actions/workflows/ci.yml/badge.svg)](https://github.com/ShivSupane/fliprate/actions/workflows/ci.yml)

Built by **Shiv Supane**

---

## The problem

People type the same message in many spellings: `nahi` / `nhi` / `nahin`, `bahut` / `bohot`, `avadla` / `avdla` / `avadlaaa`.
Most models treat these as different inputs and change the answer. FlipRate measures that instability and helps protect the model.

---

## Install

```bash
pip install fliprate
```

From source:

```bash
git clone https://github.com/ShivSupane/fliprate.git
cd fliprate
pip install -e ".[dev,api]"
```

---

## Quick start

### Test a model

```python
import fliprate as fr

report = fr.test(my_model, "data/test.csv")
print(report.summary())
```

### Protect a model

```python
safe_model = fr.protect(my_model)
report2 = fr.test(safe_model, "data/test.csv")
```

### Quick gate check

```python
result = fr.gate(my_model, "mala product avadlaaa nahi")
print(result.decision)   # PASS / TRANSFORM / REVIEW
print(result.reason)
```

### Product-friendly wrappers

```python
import fliprate as fr

# one-line runtime gate
res = fr.guard(my_model, "mala order avadlaaa nahi")

# one-line audit
report = fr.audit(my_model, rows)

# mine candidate rules from real texts
cands = fr.learn_rules_from_texts([
    "mala avadla nahi",
    "mala avdla nahi",
    "mala avadlaa nahi",
])
```

---

## Web console and CLI

```bash
uvicorn fliprate.api:app --reload
```

Open `http://localhost:8000`.

Core CLI examples:

```bash
fliprate gate "mala product avadlaaa nahi"
fliprate normalize "mala product avadlaaa nhi"
fliprate test --data mydata.csv
```

---

## Real API

```python
from fliprate import normalize, gate, test, protect

clean = normalize("Mala order avadlaaa nahi", lowercase=False)
print(clean.normalized)

result = gate(my_model, "mala product avadlaaa nahi")
print(result.decision)
```

---

## Production notes

- `normalize(..., lowercase=False)` preserves case correctly.
- The library is safe for a single-process deployment.
- For multi-instance deployment behind a load balancer, the in-memory rate limiter and job store are process-local, so they do not share state across machines.

---

## License

MIT — see [LICENSE](LICENSE).
