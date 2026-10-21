# Release and deploy checklist

**Author: Shiv Supane**

---

## A. Before anything public

1. `pyproject.toml` is already set correctly:
   - `name = "Shiv Supane"`
   - `email = "shiv.supane777@gmail.com"`
   - `Homepage = "https://github.com/ShivSupane/fliprate"`

2. Check the PyPI name is free: open `https://pypi.org/project/fliprate/`.
   A 404 page means it is available.

3. Do **not** ship third-party datasets. The package ships only the 40-message demo sample.
   Dakshina (CC BY-SA 4.0) and Aksharantar (CC BY 4.0) must be downloaded separately.

4. Run `python -m pytest tests/ -q` and confirm 80 tests pass.

---

## B. Push to GitHub

```bash
# One-time: add the remote (use your actual GitHub username)
git remote add origin https://github.com/ShivSupane/fliprate.git
git push -u origin main
```

The CI workflow (`ci.yml`) runs automatically on every push across Python 3.9–3.12.

---

## C. Publish to PyPI (trusted publishing — no stored password)

**One-time setup — do this before creating any release:**

**Step 1 — Create the `pypi` environment on GitHub**
```
Your repo → Settings → Environments → New environment → name it: pypi
```
No secrets needed. The environment name must match `release.yml` exactly.

**Step 2 — Register a pending publisher on PyPI**
```
pypi.org → Your account → Publishing → Add new pending publisher
  Project name:      fliprate
  GitHub owner:      ShivSupane
  Repository name:   fliprate
  Workflow filename: release.yml
  Environment name:  pypi
```

**Step 3 — TestPyPI dry run (do this first)**
```bash
pip install build twine
python -m build
twine check dist/*
twine upload --repository testpypi dist/*

# Install and test in a clean venv
python -m venv /tmp/fr_test
/tmp/fr_test/bin/pip install --index-url https://test.pypi.org/simple/ --no-deps fliprate
/tmp/fr_test/bin/fliprate gate "mala product avadlaaa nahi"
```

**Step 4 — Real publish**
```
GitHub → Releases → New release → Tag: v0.2.1 → Publish release
```
The `release.yml` workflow runs automatically: installs the package, runs pytest, builds, and uploads.

**For every future release:** bump `version` in `pyproject.toml` and `src/fliprate/__init__.py`,
add a line to `CHANGELOG.md`, create a GitHub release with the matching tag.

---

## D. Deploy the live console

**Hugging Face Space (free, easiest)**
1. Create a Space → SDK: Docker
2. Push the repo to the Space
3. Set Space secrets:
   - `FLIPRATE_MODEL=fliprate.hf_models:xlmt_sentiment`
   - `FLIPRATE_API_KEY=<your-secret-key>`
   - `FLIPRATE_CORS=https://your-space-url.hf.space`

Note: The Dockerfile installs CPU-only PyTorch when the `hf` extra is used (saves ~1.8 GB vs the default wheel).
XLM-R downloads on first start (~1.3 GB). The HEALTHCHECK `--start-period=120s` accounts for this.

**Render / Railway / Fly.io**
Deploy the Dockerfile as a web service. These platforms set `$PORT` automatically.
The `--proxy-headers` flag in the CMD is already set — the rate limiter will see real client IPs.

**After deploying, verify:**
- `GET /health` → `{"status": "ok"}`
- `GET /ready` → 503 until model loads, then `{"status": "ready"}`
- Open the Playground, type a message, click Run gate
- Confirm events arrive one by one (if all arrive at once, the proxy is buffering SSE)

---

## E. Dataset attribution

If you publish work using the multi-language rules, cite:

**Dakshina:**
> Roark et al. (2020). "Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset."
> LREC 2020. https://aclanthology.org/2020.lrec-1.294

**Aksharantar:**
> Mittal et al. (2022). "Aksharantar: Towards Building Open Transliteration Tools for the Next Billion Users."
> https://arxiv.org/abs/2205.03018
