### Robustness by model

| Model | Mode | Acc (originals) | Acc (variants) | Macro-F1 (variants) | Flip rate (95% CI) |
|---|---|---|---|---|---|
| tfidf-word | raw | 69.1% | 69.0% | 0.665 | 3.7% (3.1% to 4.4%) |
| tfidf-word | protected | 69.6% | 69.4% | 0.664 | 2.5% (2.0% to 3.0%) |
| tfidf-char | raw | 69.9% | 69.8% | 0.668 | 3.0% (2.4% to 3.6%) |
| tfidf-char | protected | 70.1% | 69.8% | 0.666 | 2.3% (1.7% to 2.8%) |

### Effect of protection (normalization in front of the model), paired on the same messages

| Model | Flip rate raw -> protected | Change in flip rate (paired 95% CI) | Change in accuracy, originals (paired 95% CI) |
|---|---|---|---|
| tfidf-word | 3.7% -> 2.5% | -1.24 pts [-1.73, -0.77] | +0.46 pts [-0.55, +1.47] |
| tfidf-char | 3.0% -> 2.3% | -0.74 pts [-1.09, -0.41] | +0.28 pts [-0.28, +0.83] |

messages=1085 variants/message=5 seed=0 rules=7c63bddd679f dataset=a75723158d0bc005