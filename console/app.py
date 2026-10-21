"""
FlipRate Console — the web UI is now served directly by the FastAPI server.

Start the server:
    uvicorn fliprate.api:app --reload

Then open http://localhost:8000 in your browser.

To use a real Hugging Face model:
    FLIPRATE_MODEL=fliprate.hf_models:xlmt_sentiment uvicorn fliprate.api:app

To use your own model:
    FLIPRATE_MODEL=my_package.my_module:my_function uvicorn fliprate.api:app

The Streamlit console has been removed. The built-in web console (src/fliprate/web/index.html)
provides the same Playground and Robustness Lab features, plus:
  - CSV upload directly from the browser
  - Per-kind flip rate bar chart
  - Flip examples table with answer-change pills
  - Dark/light mode, responsive layout
  - No extra dependencies (served by the existing FastAPI server)
"""

if __name__ == "__main__":
    import subprocess
    import sys
    print(__doc__)
    print("Starting server on http://localhost:8000 …")
    subprocess.run([sys.executable, "-m", "uvicorn", "fliprate.api:app", "--reload"])
