# Your labeled data goes here

FlipRate needs a CSV with two columns, `text` and `label`:

```
text,label
mera order cancel ho gaya bina bataye,order_issue
mala refund ajun milala nahi,payment_issue
```

## How to build a trustworthy dataset

1. Collect real messages (anonymize names, phone numbers, addresses). Never use someone's private chat without consent.
2. Aim for ~100 messages for development and ~200 held-out messages for the final test. Keep them in separate files (`dev.csv`, `test.csv`).
3. Ask a fluent Marathi/Hindi speaker to review every label.
4. Write down your labeling rules in one paragraph, so another person could reproduce your labels.
5. Never tune your normalizer rules on the test file. Look at it once, at the end.

The bundled `src/fliprate/data/sample_support.csv` (40 messages) is a **toy sample for demos**. It was written to
exercise the library and has not been reviewed by a fluent speaker. Do not publish results measured on it.
