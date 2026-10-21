### Robustness by model

| Model | Mode | Acc (originals) | Acc (variants) | Macro-F1 (variants) | Flip rate (95% CI) |
|---|---|---|---|---|---|
| tfidf-word | raw | 65.9% | 65.4% | 0.654 | 4.2% (3.8% to 4.6%) |
| tfidf-word | protected | 65.6% | 65.1% | 0.651 | 3.2% (2.9% to 3.6%) |
| tfidf-char | raw | 64.1% | 64.0% | 0.643 | 6.4% (5.9% to 7.0%) |
| tfidf-char | protected | 64.3% | 64.0% | 0.643 | 4.3% (3.9% to 4.7%) |

### Effect of protection (normalization in front of the model), paired on the same messages

| Model | Flip rate raw -> protected | Change in flip rate (paired 95% CI) | Change in accuracy, originals (paired 95% CI) |
|---|---|---|---|
| tfidf-word | 4.2% -> 3.2% | -0.95 pts [-1.18, -0.71] | -0.30 pts [-0.60, +0.00] |
| tfidf-char | 6.4% -> 4.3% | -2.14 pts [-2.50, -1.80] | +0.20 pts [-0.03, +0.43] |

messages=2997 variants/message=5 seed=0 rules=7c63bddd679f dataset=616241c4fef4a4b8