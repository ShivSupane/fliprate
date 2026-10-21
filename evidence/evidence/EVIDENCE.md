# FlipRate evidence report

Generated from saved result files by FlipRate 0.2.0 (Python 3.12.3). Every number below comes from a run, none is typed in by hand.
**Toy results from the bundled 40-message demo sample are not included anywhere in this report.**

## 1. What was tested

| | |
|---|---|
| Real data (main) | SentiMix Hinglish sentiment, a participant-processed copy (FinalTrainingOnly / FinalTest + Ty.txt), cleaned with `experiments/prepare_sentimix_processed.py` |
| Train / held-out test | 14,569 rows / 2,997 rows (negative 900, neutral 1,098, positive 999) |
| Secondary data | A Kaggle file (`output.csv`), 2,716 rows after trimming and de-duplication, split 1,631 / 1,085 with no leakage |
| Models | TF-IDF + logistic regression (word and character n-grams), trained here; plus the pretrained transformer `cardiffnlp/twitter-xlm-roberta-base-sentiment`, zero-shot (section 3b) |
| Variants | 5 per message: spelling swaps, stretched letters, punctuation, case. Seed 0 |
| Rules | Built-in spelling groups only (`--no-lexicon`), fingerprint `7c63bddd679f`. Written by the tool author, not reviewed by a fluent speaker |
| Confidence intervals | 95%, bootstrap over messages. Raw-versus-protected differences use a paired bootstrap on the same messages |

## 2. Problems found in the SentiMix copy (checked on the files)

- Label mapping 0 = negative, 1 = neutral, 2 = positive was verified: validation counts (890 / 1,128 / 982) and test counts from Ty.txt (900 / 1,100 / 1,000) match the published SentiMix numbers exactly.
- 43.7% of training rows end with a stray "nan" token (a stringified missing value). Validation and test have none. Removed from all splits.
- The "training-only" file contains 2,555 of the 3,000 validation tweets (same ids, same labels). Validation was therefore **not** used as held-out data. Only the test split was.
- Emoji or non-ASCII characters: 22% of train/valid rows, 2% of test rows. Removed from all splits.
- The text is lowercased, stopword-stripped and Porter-stemmed (for example "polic", "occupi", "movi"). This cannot be repaired, and it means this copy is **not natural user typing**.
- Label noise: 112 groups of near-identical training messages carry different labels; 152 test rows (5.1%) also appear in train, 45 of them with a different label.

## 3. Main results: SentiMix (processed copy), held-out test

#### All 2,997 test rows

| Model | Mode | Accuracy (originals) | Accuracy (variants) | Macro-F1 (originals) | Macro-F1 (variants) | Flip rate (95% CI) |
|---|---|---|---|---|---|---|
| tfidf-word | raw | 65.9% | 65.4% | 0.658 | 0.654 | 4.2% (3.8% to 4.6%) |
| tfidf-word | protected | 65.6% | 65.1% | 0.655 | 0.651 | 3.2% (2.9% to 3.6%) |
| tfidf-char | raw | 64.1% | 64.0% | 0.644 | 0.643 | 6.4% (5.9% to 7.0%) |
| tfidf-char | protected | 64.3% | 64.0% | 0.646 | 0.643 | 4.3% (3.9% to 4.7%) |

| Model | Change in flip rate (paired 95% CI) | Change in accuracy on originals (paired 95% CI) |
|---|---|---|
| tfidf-word | -0.95 pts [-1.18, -0.71] | -0.30 pts [-0.60, +0.00] |
| tfidf-char | -2.14 pts [-2.50, -1.80] | +0.20 pts [-0.03, +0.43] |

Flip rate by change type (share of variants whose answer changed):

| Model | Mode | case | punctuation | repetition (stretched letters) | spelling swap |
|---|---|---|---|---|---|
| tfidf-word | raw | 0.0% | 0.0% | 7.3% | 6.4% |
| tfidf-word | protected | 0.0% | 0.0% | 6.7% | 0.0% |
| tfidf-char | raw | 0.0% | 3.1% | 10.6% | 6.2% |
| tfidf-char | protected | 0.0% | 3.3% | 7.4% | 0.0% |

Sensitivity: dropping the 152 test rows that also appear in training (2,845 rows) changes nothing material (word 4.1% to 3.2%, char 6.1% to 3.9%, paired changes -0.93 and -2.12 points).

## 3b. Transformer: cardiffnlp/twitter-xlm-roberta-base-sentiment (zero-shot, run on a Colab GPU)

Run by the project owner in Google Colab (Python 3.13.15, FlipRate 0.2.0); I analyzed the saved result files and did not run it myself. The model was **not fine-tuned** on this data. Same 2,997 test messages, same seed and same variants as section 3 (dataset hash `616241c4fef4a4b8` and rules fingerprint `7c63bddd679f` match), so the comparisons below are paired. 17,955 model calls in 59 seconds.

| Mode | Accuracy (originals) | Accuracy (variants) | Macro-F1 (originals) | Macro-F1 (variants) | Flip rate (95% CI) |
|---|---|---|---|---|---|
| raw | 59.2% | 58.2% | 0.555 | 0.538 | 7.3% (6.7% to 7.9%) |
| protected | 59.1% | 58.9% | 0.555 | 0.550 | 3.5% (3.1% to 3.9%) |

Paired effect of the normalizer: flip rate -3.80 points [-4.22, -3.39]; accuracy on originals -0.03 points [-0.23, +0.17].

Flip rate by change type:

| Mode | case | punctuation | repetition (stretched letters) | spelling swap |
|---|---|---|---|---|
| raw | 11.4% | 5.0% | 7.3% | 3.6% |
| protected | 0.3% | 4.9% | 4.9% | 0.0% |

Compared with the scikit-learn models on the same messages (raw, paired): the transformer flips more than the word model (+3.08 points [+2.41, +3.78]) and slightly more than the character model (+0.87 points [+0.07, +1.62]), and its accuracy on originals is lower (-6.74 points vs word, -4.90 points vs char). After protection all three models land in a similar range (3.5% transformer, 3.2% word, 4.3% char).

## 4. Secondary results: Kaggle file (unverified labels, machine-transliterated text)

#### 1,085 held-out rows

| Model | Mode | Accuracy (originals) | Accuracy (variants) | Macro-F1 (originals) | Macro-F1 (variants) | Flip rate (95% CI) |
|---|---|---|---|---|---|---|
| tfidf-word | raw | 69.1% | 69.0% | 0.665 | 0.665 | 3.7% (3.1% to 4.4%) |
| tfidf-word | protected | 69.6% | 69.4% | 0.666 | 0.664 | 2.5% (2.0% to 3.0%) |
| tfidf-char | raw | 69.9% | 69.8% | 0.668 | 0.668 | 3.0% (2.4% to 3.6%) |
| tfidf-char | protected | 70.1% | 69.8% | 0.669 | 0.666 | 2.3% (1.7% to 2.8%) |

| Model | Change in flip rate (paired 95% CI) | Change in accuracy on originals (paired 95% CI) |
|---|---|---|
| tfidf-word | -1.24 pts [-1.73, -0.77] | +0.46 pts [-0.55, +1.47] |
| tfidf-char | -0.74 pts [-1.09, -0.41] | +0.28 pts [-0.28, +0.83] |

## 5. What these results do show

- Both classical models change their answer on a measurable share of meaning-preserving variants: about 4% to 6% on SentiMix and 3% to 4% on the Kaggle file.
- On SentiMix most flips come from stretched letters (word 7.3%, char 10.6% of those variants) and spelling swaps (about 6%). Case changes never flip either model, because both lowercase internally.
- On this data the character n-gram model flips more than the word model (6.4% vs 4.2%). I expected the opposite and have not investigated why.
- The transformer is the most case-sensitive model: 11.4% of its answers change when a message is upper-cased or capitalized, against 0.0% for the two TF-IDF models, which lowercase internally. Lowercasing in the normalizer removes this (0.3% left). This effect does not depend on the spelling rule list, so it is not circular. How much it matters depends on how real users type, and this dataset is already all lowercase.
- Punctuation changes still flip the transformer 4.9% of the time after protection, because the normalizer leaves punctuation alone.
- Putting the normalizer in front lowers the flip rate in all four model/data pairs, and every paired interval excludes zero. Accuracy on original messages does not change beyond noise (all paired intervals include or touch zero).
- Accuracy barely moves under variants (65.9% to 65.4% for the word model), so flip rate and accuracy measure different things: a flip can go toward the right answer or the wrong one.

## 6. What these results do not show

- **The spelling improvement is partly by construction.** The spelling variants and the normalizer use the same rule list, so normalizing undoes the swaps and the spelling flip rate falls to 0.0%. It only shows the normalizer covers its own rules, not that it fixes spelling variants it has never seen. A fair test needs variants from an independent source, such as real noisy messages.
- Stretched-letter flips are only partly fixed (word 7.3% to 6.7%, char 10.6% to 7.4%), because for words outside the rules the normalizer shortens runs to two letters rather than recovering the original word.
- The variants are rule-generated and no fluent speaker has judged whether they keep the meaning. That audit has not been done.
- The text is stemmed and lowercased, so these numbers say little about messages typed by real users.
- Only one transformer was tested, zero-shot. A fine-tuned transformer or an LLM may behave differently. The transformer's lower accuracy is probably partly because it saw stemmed, stopword-free text, which is unlike its training data.
- Baseline macro-F1 of about 0.65 is plausible for TF-IDF and no leakage warning fired, but I did not compare it with published scores, whose metric may differ.

## 7. Not done yet

1. Human audit of the variants (`fliprate audit`, then `fliprate audit-summary`).
2. The official unprocessed SentiMix files, which would give natural casing, punctuation and spelling.
3. A vocabulary-based normalizer built from training data, aimed at the stretched-letter flips that remain.
4. A fine-tuned transformer and an LLM, to see whether the flips shrink when a model has seen this kind of text.
