"""03_with_hf.py — Testing Hugging Face Transformers.

Shows how to connect any Hugging Face sentiment or classification pipeline
using HFTextClassifier with batched inference and label normalization.
"""
import fliprate as fr


def main():
    print("=" * 60)
    print("FlipRate — Hugging Face Adapter Demo")
    print("=" * 60)

    # Note: Requires `pip install "fliprate[hf]"` (transformers + torch)
    # Lazy loaded — does not download until first predict call.
    try:
        clf = fr.HFTextClassifier(
            "cardiffnlp/twitter-xlm-roberta-base-sentiment",
            lowercase_labels=True,
            preprocess=fr.twitter_preprocess,
            batch_size=16,
        )
        print("HFTextClassifier initialized successfully.")
        print("Metadata:", clf.metadata())
    except ImportError:
        print("Note: transformers/torch not installed in current environment.")
        print("Install via: pip install 'fliprate[hf]'")
        return

    sample = "ye phone bohot accha hai"
    print(f"\nTesting gate on: '{sample}'")
    verdict = fr.quick_check(clf, sample)
    print(f"Decision: {verdict.decision} | Prediction: {verdict.prediction}")


if __name__ == "__main__":
    main()
