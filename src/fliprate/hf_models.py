"""Ready-made real models for deployment. Use with:  FLIPRATE_MODEL=fliprate.hf_models:xlmt_sentiment

Importing this module is cheap: the model itself downloads and loads only on first use (or at server warm-up).
Needs:  pip install "fliprate[hf]"
"""
from .adapters import HFTextClassifier, twitter_preprocess

# 3-class tweet sentiment (negative / neutral / positive). Labels are lowercased to match the SentiMix CSVs.
xlmt_sentiment = HFTextClassifier(
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    lowercase_labels=True,
    preprocess=twitter_preprocess,
)
