"""02_with_sklearn.py — Evaluating and Protecting a scikit-learn Model.

Shows:
  - Training a standard TF-IDF + LogisticRegression pipeline
  - Wrapping with fr.from_sklearn()
  - Measuring baseline flip rate
  - Protecting the model with fr.protect()
  - Measuring paired difference (before vs after)
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import fliprate as fr

# 1. Sample Hinglish training data
TRAIN_DATA = [
    ("bahut accha product hai", "positive"),
    ("badhiya service lagi", "positive"),
    ("ekdum mast experience", "positive"),
    ("aacha laga", "positive"),
    ("bohot pasand aaya", "positive"),
    ("kharab delivery thi", "negative"),
    ("product bilkul bekar hai", "negative"),
    ("paise barbad ho gaye", "negative"),
    ("delivery time pe nahi aayi", "negative"),
    ("refund nahi mila", "negative"),
]

# Held-out evaluation data
TEST_DATA = [
    ("bahut accha laga", "positive"),
    ("delivery time par nahi aayi", "negative"),
    ("badhiya laga mujhe", "positive"),
    ("paise barbad", "negative"),
]


def main():
    print("=" * 60)
    print("FlipRate — scikit-learn Pipeline Integration")
    print("=" * 60)

    # 2. Train a scikit-learn pipeline
    X_train = [t for t, l in TRAIN_DATA]
    y_train = [l for t, l in TRAIN_DATA]
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("clf", LogisticRegression()),
    ])
    pipe.fit(X_train, y_train)

    # 3. Wrap with FlipRate adapter
    model = fr.from_sklearn(pipe, name="tfidf_logistic")

    # 4. Measure robustness on raw model
    print("\nAuditing raw scikit-learn model...")
    raw_report = fr.test(model, TEST_DATA, n_variants=4, bootstrap=50)
    print(f"Raw Flip Rate: {raw_report.flip_rate:.1%}")

    # 5. Protect model with FlipRate normalizer
    print("\nProtecting model with FlipRate normalizer...")
    safe_model = fr.protect(model)
    safe_report = fr.test(safe_model, TEST_DATA, n_variants=4, bootstrap=50)
    print(f"Protected Flip Rate: {safe_report.flip_rate:.1%}")

    # 6. Statistical comparison (Paired Bootstrap)
    diff = fr.paired_difference(raw_report, safe_report)
    print(f"\nPaired Flip-Rate Difference: {diff['difference']:+.1%}")
    print(f"95% Confidence Interval: {diff['ci']}")


if __name__ == "__main__":
    main()
