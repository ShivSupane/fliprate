```
Rows: 2845
Labels: negative=858, neutral=1055, positive=932
Exact duplicates: 13 | near-duplicate groups: 13 | groups with conflicting labels: 3
WARNING: 3 group(s) of near-identical messages have different labels; fix the labels
WARNING: 13 group(s) of near-duplicate messages (same words, different spelling); keep them in ONE split
```

Train/test overlap: 0 of 2845 test rows (0.0%) also appear in train; 0 with a different label.

### Robustness by model

| Model | Mode | Acc (originals) | Acc (variants) | Macro-F1 (variants) | Flip rate (95% CI) |
|---|---|---|---|---|---|
| tfidf-word | raw | 65.7% | 65.3% | 0.653 | 4.1% (3.8% to 4.6%) |
| tfidf-word | protected | 65.3% | 65.0% | 0.650 | 3.2% (2.9% to 3.6%) |
| tfidf-char | raw | 63.7% | 63.5% | 0.638 | 6.1% (5.6% to 6.6%) |
| tfidf-char | protected | 63.9% | 63.7% | 0.641 | 3.9% (3.5% to 4.4%) |

### Effect of protection (normalization in front of the model), paired on the same messages

| Model | Flip rate raw -> protected | Change in flip rate (paired 95% CI) | Change in accuracy, originals (paired 95% CI) |
|---|---|---|---|
| tfidf-word | 4.1% -> 3.2% | -0.93 pts [-1.20, -0.70] | -0.32 pts [-0.63, +0.00] |
| tfidf-char | 6.1% -> 3.9% | -2.12 pts [-2.49, -1.76] | +0.21 pts [-0.04, +0.46] |

messages=2845 variants/message=5 seed=0 rules=7c63bddd679f dataset=ac80b13b0464db64
