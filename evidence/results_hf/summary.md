### Robustness by model

| Model | Mode | Acc (originals) | Acc (variants) | Macro-F1 (variants) | Flip rate (95% CI) |
|---|---|---|---|---|---|
| cardiffnlp/twitter-xlm-roberta-base-sentiment | raw | 59.2% | 58.2% | 0.538 | 7.3% (6.7% to 7.9%) |
| cardiffnlp/twitter-xlm-roberta-base-sentiment | protected | 59.1% | 58.9% | 0.550 | 3.5% (3.1% to 3.9%) |

### Effect of protection (normalization in front of the model), paired on the same messages

| Model | Flip rate raw -> protected | Change in flip rate (paired 95% CI) | Change in accuracy, originals (paired 95% CI) |
|---|---|---|---|
| cardiffnlp/twitter-xlm-roberta-base-sentiment | 7.3% -> 3.5% | -3.80 pts [-4.22, -3.39] | -0.03 pts [-0.23, +0.17] |

messages=2997 variants/message=5 seed=0 rules=7c63bddd679f dataset=616241c4fef4a4b8