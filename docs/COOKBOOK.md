# FlipRate Cookbook: Practical Real-World Patterns

A collection of copy-pasteable recipes for common NLP robustness and quality-guarding problems.

---

## Table of Contents
1. [Recipe 1: Sentiment Model Quality Gate](#recipe-1-sentiment-model-quality-gate)
2. [Recipe 2: Chatbot Input & Intent Guarding](#recipe-2-chatbot-input--intent-guarding)
3. [Recipe 3: Customer Support Ticket Classification](#recipe-3-customer-support-ticket-classification)
4. [Recipe 4: Adding Custom Domain Whitelists and Slang](#recipe-4-adding-custom-domain-whitelists-and-slang)
5. [Recipe 5: Multilingual Indic Evaluation (Hindi, Marathi, Tamil, Telugu)](#recipe-5-multilingual-indic-evaluation)
6. [Recipe 6: Explainability and Debugging Unstable Predictions](#recipe-6-explainability-and-debugging-unstable-predictions)

---

## Recipe 1: Sentiment Model Quality Gate

**Goal:** Ensure a customer sentiment model doesn't flip from positive to negative simply because of spelling variations.

```python
import fliprate as fr

# Wrap your trained classifier (sklearn, HF, or custom function)
safe_model = fr.protect(my_sentiment_classifier)

# In your API endpoint:
def analyze_sentiment(user_comment: str):
    # 1. Quick check
    verdict = fr.quick_check(safe_model, user_comment)
    
    if verdict.is_safe:
        return {"sentiment": verdict.prediction, "status": "confident"}
    else:
        # Route borderline/unstable inputs for human review
        return {
            "sentiment": verdict.prediction,
            "status": "flagged_for_review",
            "reason": verdict.reason
        }
```

---

## Recipe 2: Chatbot Input & Intent Guarding

**Goal:** Sanitize user messages before sending them to an LLM or intent classifier to prevent misrouting.

```python
import fliprate as fr

def handle_user_query(query: str):
    # 1. Validate length and content
    clean_query = fr.validate_text(query, max_length=500)
    
    # 2. Runtime guard
    guard_res = fr.runtime_guard(my_intent_classifier, clean_query, mode="lenient")
    
    if guard_res.should_route_to_human:
        return {
            "action": "transfer_to_agent",
            "clean_text": guard_res.clean_text,
            "reason": guard_res.reason
        }
    
    # Send standardized canonical text to your chatbot prompt
    bot_response = query_llm_agent(guard_res.clean_text)
    return {"action": "reply", "response": bot_response}
```

---

## Recipe 3: Customer Support Ticket Classification

**Goal:** Process incoming tickets in batch and generate an audit report of model stability.

```python
import fliprate as fr

tickets = [
    ("mera refund time pe nahi aaya", "billing"),
    ("delivery time par nahi mili", "logistics"),
    ("password reset link kaam nahi kar raha", "technical"),
]

# 1. Audit pipeline and save audit report
report = fr.evaluate_pipeline(
    my_ticket_classifier,
    tickets,
    save_path="results/weekly_ticket_audit.json",
    n_variants=5
)

print(report.summary())
print(f"Overall Flip Rate: {report.flip_rate:.1%}")
```

---

## Recipe 4: Adding Custom Domain Whitelists and Slang

**Goal:** Add industry-specific abbreviations or brand slang without touching library source code.

```python
import fliprate as fr

# Define custom spelling equivalents for e-commerce or fintech
ecommerce_rules = fr.add_custom_rule(
    variants=["cod", "c-o-d", "cashondelivery", "cash on delivery"],
    canonical="cod"
)
ecommerce_rules = fr.add_custom_rule(
    variants=["discnt", "dscnt", "disc", "discount"],
    canonical="discount",
    rules=ecommerce_rules
)

# Use in your gate or normalizer
clean_text = fr.normalize_text("order pe discnt nahi mila", rules=ecommerce_rules)
# -> "order pe discount nahi mila"
```

---

## Recipe 5: Multilingual Indic Evaluation

**Goal:** Test model resilience across Romanized Indic languages as well as native Devanagari script.

```python
import fliprate as fr

# Romanized Tamil
ta_rules = fr.load_rules("tamil")
report_ta = fr.check_model(my_model, "data/tamil_test.csv", rules=ta_rules)

# Native Devanagari (Hindi / Marathi)
dev_query = "मला डिलिव्हरी वेळेवर नाही मिळाली"
print(fr.detect_script(dev_query))  # "devanagari"

# Script-aware gate
result = fr.gate(my_model, dev_query)
print(f"Decision: {result.decision} | Cleaned: {result.text_used}")
```

---

## Recipe 6: Explainability and Debugging Unstable Predictions

**Goal:** Understand exactly why a message triggered `REVIEW` or `TRANSFORM`.

```python
import fliprate as fr

msg = "delivery nahi mili"

# 1. Plain-English explanation
explanation = fr.explain_gate_decision(msg, my_model)
print("Summary:", explanation["summary"])
print("Recommendation:", explanation["recommendation"])

# 2. View specific perturbations that flipped the model
flips = fr.show_why_it_flipped(msg, my_model)
for f in flips:
    print(f"Variant '{f['variant_text']}' flipped prediction from {f['original_prediction']} to {f['variant_prediction']}")
```
