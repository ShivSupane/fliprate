"""05_custom_rules_and_debugging.py — Rule Management & Explainability Tools.

Shows:
  - Loading pre-curated rules by language (fr.load_rules)
  - Adding domain-specific slang / acronym rules (fr.add_custom_rule)
  - Inspecting rule hits on a message (fr.print_rule_hits)
  - Explaining why a gate decision happened (fr.explain_gate_decision)
  - Showing which variants flipped the model (fr.show_why_it_flipped)
"""
import fliprate as fr


def brittle_model(text: str) -> str:
    # Model only recognizes exact spelling 'nahi'
    return "support_needed" if "nahi" in text.split() else "no_action"


def main():
    print("=" * 60)
    print("FlipRate — Rule Management & Diagnostics")
    print("=" * 60)

    # 1. Load language rules (Hindi, Marathi, Tamil, etc.)
    mr_rules = fr.load_rules("marathi")
    print(f"Loaded Marathi rules: {len(mr_rules.groups)} rule groups available.")

    # 2. Add custom domain-specific rules (e.g. e-commerce abbreviations)
    custom_rules = fr.add_custom_rule(
        variants=["discnt", "dscnt", "disc", "discount"],
        canonical="discount"
    )
    desc = fr.describe_rule_group("dscnt", rules=custom_rules)
    print("\nCustom Rule Group Details:")
    print(f"  Canonical word: {desc['canonical']}")
    print(f"  Known variants: {desc['variants']}")

    # 3. Preview generated perturbations
    test_msg = "order delivery nahi aayi"
    print(f"\nPreviewing perturbations for: '{test_msg}'")
    variants = fr.preview_variants(test_msg, n=5)
    for i, v in enumerate(variants, 1):
        print(f"  {i}. {v}")

    # 4. Print rule hits (which tokens matched our dictionaries)
    print(f"\nScanning matched tokens in '{test_msg}':")
    hits = fr.print_rule_hits(test_msg)
    for h in hits:
        print(f"  Token: '{h['token']}' -> Canonical: '{h['canonical']}' (Negation: {h['is_negation']})")

    # 5. Explain gate decision in plain English
    explanation = fr.explain_gate_decision(test_msg, brittle_model)
    print("\nDecision Explanation:")
    print(f"  Decision: {explanation['decision']}")
    print(f"  Summary: {explanation['summary']}")
    print(f"  Recommendation: {explanation['recommendation']}")

    # 6. Show exactly why and where the model flipped
    flips = fr.show_why_it_flipped(test_msg, brittle_model, n=4)
    print(f"\nFlips detected ({len(flips)}):")
    for f in flips:
        print(f"  - [{f['kind']}] Variant '{f['variant_text']}' changed output: '{f['original_prediction']}' -> '{f['variant_prediction']}'")


if __name__ == "__main__":
    main()
