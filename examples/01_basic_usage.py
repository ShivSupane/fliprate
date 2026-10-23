"""01_basic_usage.py — Beginner Quick Start with FlipRate.

Shows the simplest happy-path APIs:
  - quick_check()
  - normalize_text()
  - safe_predict()
  - check_model()
"""
import fliprate as fr


# 1. Define any model (a simple function text -> label)
def my_sentiment_model(text: str) -> str:
    lower = text.lower()
    if any(neg in lower for neg in ["nahi", "nhi", "nahin", "kharab", "bekar"]):
        return "negative"
    if any(pos in lower for pos in ["accha", "bahut", "badhiya", "mast"]):
        return "positive"
    return "neutral"


def main():
    print("=" * 60)
    print("FlipRate — Basic Usage & Happy-Path Demo")
    print("=" * 60)

    # 2. Quick 1-line check on a user message
    msg = "ye product bohot accha hai"
    result = fr.quick_check(my_sentiment_model, msg)
    print(f"\nMessage: '{msg}'")
    print(f"Safe to process: {result.is_safe}")
    print(f"Model prediction: {result.prediction}")
    print(f"Reason: {result.reason}")

    # 3. Clean messy romanized or Devanagari text directly
    messy = "Mala product avadlaaa nhiii!!!"
    clean = fr.normalize_text(messy, lowercase=False)
    print(f"\nMessy: '{messy}'")
    print(f"Clean: '{clean}'")

    # 4. Safe prediction (automatically handles spelling normalization)
    prediction = fr.safe_predict(my_sentiment_model, "ye delivery nhi aayi")
    print(f"\nSafe prediction on 'ye delivery nhi aayi': {prediction}")

    # 5. Check model robustness on a dataset
    dataset = [
        ("product bahut accha hai", "positive"),
        ("delivery nahi mili", "negative"),
        ("theek thaak laga", "neutral"),
        ("mala service aawadli", "positive"),
    ]
    print("\nAuditing model on dataset...")
    report = fr.check_model(my_sentiment_model, dataset, n_variants=3, bootstrap=50)
    print(report.summary())


if __name__ == "__main__":
    main()
