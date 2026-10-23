"""04_with_llm.py — Testing LLM Classifiers (OpenAI, Anthropic, or local LLMs).

Shows how to wrap an LLM completion function using LLMClassifier with strict label parsing
and exponential backoff retries.
"""
import fliprate as fr


# Simulated LLM completion API
def mock_llm_completion(prompt: str) -> str:
    # In real usage, replace this with:
    # response = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0)
    # return response.choices[0].message.content
    lower = prompt.lower()
    if "refund" in lower or "cancel" in lower:
        return "billing"
    if "login" in lower or "password" in lower:
        return "account"
    return "general_query"


def main():
    print("=" * 60)
    print("FlipRate — LLM Classifier Integration")
    print("=" * 60)

    labels = ["billing", "account", "general_query"]

    # Wrap with LLMClassifier
    llm = fr.llm_classifier(
        mock_llm_completion,
        labels=labels,
        instruction="Classify this customer support query into exactly one label.",
    )

    # Wrap with retry decorator for production resilience
    safe_llm = fr.with_retries(llm, retries=3, backoff=0.5)

    test_message = "mujhe apna refund nahi mila abhi tak"
    print(f"\nUser Query: '{test_message}'")
    pred = safe_llm(test_message)
    print(f"LLM Prediction: {pred}")

    print("\nRunning quick_check on LLM...")
    check = fr.quick_check(safe_llm, test_message)
    print(f"Is safe: {check.is_safe} | Decision: {check.decision} | Cleaned: {check.suggested_text}")


if __name__ == "__main__":
    main()
