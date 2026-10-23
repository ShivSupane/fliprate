"""06_production_pipeline.py — Real-time Production Ingestion & Gating.

Demonstrates:
  - Input text validation (fr.validate_text)
  - Strict vs lenient runtime quality guarding (fr.runtime_guard)
  - Batch routing for incoming customer service tickets (fr.batch_guard)
"""
import fliprate as fr


def classifier_service(text: str) -> str:
    lower = text.lower()
    if any(w in lower for w in ["refund", "paisa", "paise", "cancel"]):
        return "billing_support"
    if any(w in lower for w in ["delivery", "aaya", "aayi", "track"]):
        return "logistics_support"
    return "general_help"


def process_customer_message(user_text: str, strict: bool = False):
    """Production handler for incoming user messages."""
    # Step 1: Validate input text
    try:
        clean_input = fr.validate_text(user_text, max_length=1000)
    except (TypeError, ValueError) as err:
        return {"status": "error", "message": str(err)}

    # Step 2: Runtime guard
    guard_res = fr.runtime_guard(
        classifier_service,
        clean_input,
        mode="strict" if strict else "lenient",
    )

    # Step 3: Decision routing
    if guard_res.should_route_to_human:
        return {
            "status": "routed_to_human",
            "reason": guard_res.reason,
            "raw_text": user_text,
            "suggested_text": guard_res.clean_text,
        }

    return {
        "status": "automated_success",
        "decision": guard_res.decision,
        "prediction": guard_res.prediction,
        "processed_text": guard_res.clean_text,
    }


def main():
    print("=" * 60)
    print("FlipRate — Production Pipeline Gating")
    print("=" * 60)

    incoming_messages = [
        "mera refund time pe nahi aaya",
        "delivery bohot late hai",
        "Mala refund nahi milalaaa",
    ]

    print("\nProcessing incoming customer messages:")
    for msg in incoming_messages:
        res = process_customer_message(msg)
        print(f"\nUser Query: '{msg}'")
        print(f"Outcome: {res}")


if __name__ == "__main__":
    main()
