# Release and deploy checklist

**Author: Shiv Supane**

## A. Before anything public
1. Verify the two items in `pyproject.toml`:
   - `email = "shiv.supane777@gmail.com"` — update if your email differs
   - `Homepage = "https://github.com/ShivSupane/fliprate"` — confirm your exact GitHub username

   The release workflow will refuse to publish if any `example.com` address remains.

2. Add your name to `LICENSE` if you want it there (currently just says "MIT License").

3. Check the PyPI name is free: open `https://pypi.org/project/fliprate/`. A 404 page means it is available.
   If taken, rename in `pyproject.toml` (e.g. `fliprate-nlp`); the import name can stay `fliprate`.

4. Do **not** ship third-party datasets. The package ships only the 40-message demo sample.
   The Dakshina and Aksharantar datasets have their own licences (CC BY-SA 4.0 and CC BY 4.0)
   and must be downloaded separately.

5. Run `python -m pytest tests/ -q` (80 tests) and read the README "Limitations" section.

## B. Put the code on GitHub

```bash
git init
git add .
git commit -m "FlipRate v0.2.1 — multi-language Dakshina + Aksharantar rules"
# Create an empty repo named "fliprate" on github.com, then:
git remote add origin https://github.com/ShivSupane/fliprate.git
git branch -M main
git push -u origin main
```

**Before pushing, add to `.gitignore`:**
- `dakshina_dataset_v1.0/` — ~200 MB, not for public repo
- `Aksharantar_All_Languages/` — multi-GB, not for public repo
- `Aksharantar_All_Languages.zip`
- `dakshina_dataset_v1.0.tar`
- `data/rules_from_dakshina/`
- `data/rules_from_aksharantar/`

The `tests` workflow runs on every push (Python 3.9 to 3.12).

## C. Publish the library (PyPI)

Try TestPyPI first:
```bash
pip install build twine
python -m build
twine check dist/*
twine upload --repository testpypi dist/*
# install and test in a clean env:
python -m venv /tmp/t && /tmp/t/bin/pip install --index-url https://test.pypi.org/simple/ --no-deps fliprate
/tmp/t/bin/fliprate gate "mala product avadlaaa nahi"
```

Then the real publish via trusted publishing:
1. On pypi.org: your account → Publishing → add a *pending publisher*
   (project `fliprate`, GitHub owner `ShivSupane`, repo `fliprate`, workflow `release.yml`, environment `pypi`).
2. On GitHub: Releases → Draft a new release → tag `v0.2.1` → Publish.
   The `release.yml` workflow runs tests, builds the package and uploads automatically.

For every future release: bump `version` in `pyproject.toml` and `src/fliprate/__init__.py`,
add a line to `CHANGELOG.md`, create a new GitHub release with the matching tag.

## D. Deploy the live console

The FastAPI server serves the full web console at `/`.

**Hugging Face Space (free, easiest)**
1. Create a Space → SDK: Docker.
2. Set `FLIPRATE_MODEL=fliprate.hf_models:xlmt_sentiment` in Space secrets.
3. Optionally set `FLIPRATE_API_KEY` if the Space is public.
4. The Dockerfile is already in the repo — just push.

**Render / Railway / Fly.io**: deploy the Dockerfile as a web service; they set `$PORT` automatically.

After deploying, verify:
- `/health` → `{"status": "ok"}`
- `/ready` → 503 until model loads, then `{"status": "ready"}`
- Open the Playground and confirm events stream one by one (if they all arrive at once, a proxy is buffering).

## E. Dataset attribution

If you publish work using these datasets, cite:

**Dakshina:**
> Roark et al. (2020). "Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset."
> LREC 2020. https://aclanthology.org/2020.lrec-1.294

**Aksharantar:**
> Mittal et al. (2022). "Aksharantar: Towards Building Open Transliteration Tools for the Next Billion Users."
> https://arxiv.org/abs/2205.03018
