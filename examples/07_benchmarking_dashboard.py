"""07_benchmarking_dashboard.py — Model Robustness Benchmarking & Dashboard.

Demonstrates:
  - Benchmarking multiple candidate models side-by-side (fr.benchmark)
  - Baseline comparison ("Score vs Baseline")
  - Exporting a Markdown report and an interactive HTML quality dashboard
"""
import fliprate as fr


# Model A: Keyword-based baseline
def baseline_keyword_model(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in ["kharab", "bekar", "late", "nahi"]):
        return "negative"
    return "positive"


# Model B: Stem/subword-aware candidate model
def candidate_stem_model(text: str) -> str:
    lower = text.lower()
    # Handles phonetic prefixes and roots
    if any(w in lower for w in ["kharab", "bekar", "lat", "nhi", "nah"]):
        return "negative"
    return "positive"


def main():
    print("=" * 60)
    print("FlipRate — Robustness Benchmarking & Dashboard")
    print("=" * 60)

    dataset = [
        ("bahut accha product", "positive"),
        ("delivery time pe nahi aayi", "negative"),
        ("bohot badiya laga", "positive"),
        ("service ekdum kharab thi", "negative"),
        ("aacha laga mujhe", "positive"),
        ("paise barbad ho gaye", "negative"),
    ]

    print("\nRunning benchmark across candidate models...")
    suite = fr.benchmark(
        models={
            "keyword_baseline": baseline_keyword_model,
            "candidate_stem_model": candidate_stem_model,
        },
        dataset=dataset,
        baseline="keyword_baseline",
        n_variants=3,
        bootstrap=50,
    )

    # 1. Terminal Summary
    print("\n" + suite.summary())

    # 2. Markdown output (suitable for GitHub PR comments)
    print("\n--- Markdown Report ---")
    print(suite.to_markdown())

    # 3. Export to HTML dashboard
    out_html = "results/robustness_dashboard.html"
    suite.save(out_html)
    print(f"\nSaved interactive HTML dashboard to: {out_html}")


if __name__ == "__main__":
    main()
