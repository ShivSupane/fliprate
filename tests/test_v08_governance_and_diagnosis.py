"""Tests for FlipRate v0.8.0: Rule Governance, Protected Terms, and Non-Causal Diagnosis."""
import os
import tempfile
import unittest

import fliprate as fr
from fliprate import (
    Candidate,
    CandidateRuleSet,
    DEFAULT_RULES,
    FlipRate,
    FlipRateNormalizer,
    Report,
    RuleSet,
    assert_stable,
    explain_report_flips,
    gate,
    mask_protected_terms,
    mine_variants,
    normalize,
    perturb,
    show_why_it_flipped,
    test,
    unmask_protected_terms,
)


class TestProtectedTerms(unittest.TestCase):
    """8.1 Protected terms in normalization, perturbation, and gating."""

    def test_normalize_preserves_single_and_multiword_terms(self):
        # "Tata Motors" is protected; "bohot" should still normalize to "bahut"
        text = "Tata Motors bohot accha hai"
        res = normalize(text, protected_terms=["Tata Motors"])
        self.assertIn("Tata Motors", res.normalized)
        self.assertIn("bahut", res.normalized)
        # Even with lowercase=True, protected terms preserve exact casing
        self.assertTrue(res.normalized.startswith("Tata Motors"))

    def test_normalize_overlapping_terms_longest_first(self):
        # "Tata Motors" and "Tata" - multiword term must match before single word
        text = "Tata Motors is a division of Tata."
        res = normalize(text, protected_terms=["Tata", "Tata Motors"])
        self.assertEqual(res.normalized, "Tata Motors is a division of Tata.")

    def test_normalize_boundary_safety(self):
        # Protected "car" must not match inside "scar" or "card"
        text = "He bought a scar, a car, and a card."
        masked, masks = mask_protected_terms(text, ["car"])
        self.assertEqual(masks, ["car"])
        unmasked = unmask_protected_terms(masked, masks)
        self.assertEqual(unmasked, text)

    def test_normalize_punctuation_boundaries(self):
        text = "Contact Tata Motors, SBI, or ICICI!"
        res = normalize(text, protected_terms=["Tata Motors", "SBI", "ICICI"])
        self.assertEqual(res.normalized, "contact Tata Motors, SBI, or ICICI!")

    def test_normalize_devanagari_protected_terms(self):
        text = "टाटा मोटर्स बहुत अच्छा है"
        res = normalize(text, protected_terms=["टाटा मोटर्स"])
        self.assertIn("टाटा मोटर्स", res.normalized)

    def test_perturb_preserves_protected_terms_byte_for_byte(self):
        text = "Tata Motors bahut accha hai"
        variants = perturb(text, n=6, seed=42, protected_terms=["Tata Motors"])
        self.assertTrue(len(variants) > 0)
        for v in variants:
            self.assertIn(
                "Tata Motors",
                v.text,
                f"Protected term 'Tata Motors' was modified in variant: {v.text}",
            )

    def test_perturb_repetition_never_elongates_protected_vowels(self):
        text = "Tata Motors accha hai"
        # Run 10 variants with seed 123
        variants = perturb(text, n=10, seed=123, protected_terms=["Tata Motors"])
        for v in variants:
            self.assertNotIn("Moootors", v.text)
            self.assertNotIn("Taaata", v.text)

    def test_gate_and_engine_with_protected_terms(self):
        dummy_model = lambda t: "pos" if "accha" in t or "acha" in t else "neg"

        # Direct gate call with protected_terms
        dec = gate(dummy_model, "Tata Motors accha hai", n_variants=4, seed=1, protected_terms=["Tata Motors"])
        self.assertEqual(dec.decision, "PASS")

        # Engine configured with protected_terms
        engine = FlipRate(model=dummy_model, protected_terms=["Tata Motors"], n_variants=4, seed=1)
        res = engine.gate("Tata Motors accha hai")
        self.assertEqual(res.decision, "PASS")

        # Engine normalize
        norm = engine.normalize("Tata Motors bohot accha hai")
        self.assertTrue(norm.normalized.startswith("Tata Motors"))

        # Engine perturb
        vars_out = engine.perturb("Tata Motors accha hai")
        for v in vars_out:
            self.assertIn("Tata Motors", v.text)

    def test_sklearn_normalizer_with_protected_terms(self):
        normalizer = FlipRateNormalizer(protected_terms=["Tata Motors"])
        X = ["Tata Motors bohot accha hai", "SBI ka card"]
        out = normalizer.transform(X)
        self.assertIn("Tata Motors", out[0])
        self.assertIn("bahut", out[0])
        self.assertEqual(normalizer.get_params()["protected_terms"], ["Tata Motors"])


class TestRuleGovernanceWorkflow(unittest.TestCase):
    """8.2 Reviewed rule-mining workflow and CandidateRuleSet."""

    def test_candidate_dataclass_governance_fields(self):
        c = Candidate(
            spellings=["accha", "achha", "acha"],
            counts=[10, 5, 2],
            risk="contains frequent variants",
            language="indic-romanized",
            confidence=0.85,
        )
        self.assertEqual(c.canonical, "accha")
        self.assertEqual(c.variants, ["achha", "acha"])
        self.assertEqual(c.total, 17)
        self.assertEqual(c.reviewer_status, "pending")
        self.assertEqual(c.status, "pending")

        c.approve("verified by fluent Hindi speaker")
        self.assertEqual(c.reviewer_status, "approved")
        self.assertEqual(c.notes, "verified by fluent Hindi speaker")

        c.reject("false positive collision")
        self.assertEqual(c.reviewer_status, "rejected")

        d = c.to_dict()
        self.assertEqual(d["canonical"], "accha")
        self.assertEqual(d["reviewer_status"], "rejected")
        self.assertEqual(d["total_count"], 17)

    def test_mine_variants_returns_candidate_ruleset(self):
        texts = ["mala bhaji", "mla bhaji", "mala chai", "mla chai"] * 3
        cands = mine_variants(texts, min_count=2)
        self.assertIsInstance(cands, CandidateRuleSet)
        self.assertIsInstance(cands, list)  # Backward compatibility
        self.assertTrue(len(cands) >= 1)

        summary = cands.summary()
        self.assertIn("CandidateRuleSet", summary)
        self.assertIn("Pending", summary)

    def test_candidate_ruleset_approval_and_ruleset_conversion(self):
        cands = CandidateRuleSet([
            Candidate(spellings=["mala", "mla"], counts=[10, 5]),
            Candidate(spellings=["badhiya", "badiya"], counts=[8, 4]),
        ])
        # Initially both are pending
        self.assertEqual(len(cands.pending), 2)
        self.assertEqual(len(cands.approved), 0)

        # Unapproved ruleset should not include pending candidates by default
        base_rules = RuleSet()
        empty_rs = cands.to_ruleset(base=base_rules, approved_only=True)
        self.assertIsNone(empty_rs.group_of("mla"))

        # Approve one candidate
        cands.approve("mala", notes="approved for production")
        self.assertEqual(len(cands.approved), 1)
        self.assertEqual(len(cands.pending), 1)

        # Converted ruleset now activates the approved rule
        active_rs = cands.to_ruleset(base=base_rules, approved_only=True)
        self.assertIsNotNone(active_rs.group_of("mla"))
        self.assertEqual(active_rs.key_to_canon.get("mla"), "mala")
        # Unapproved candidate remains inactive
        self.assertIsNone(active_rs.group_of("badiya"))

    def test_save_reviewed_csv_roundtrip_with_ruleset(self):
        cands = CandidateRuleSet([
            Candidate(spellings=["mala", "mla"], counts=[10, 5]),
            Candidate(spellings=["kaha", "kahan"], counts=[4, 2]),
        ])
        cands.approve("mala")
        cands.reject("kaha")

        with tempfile.TemporaryDirectory() as d:
            csv_path = os.path.join(d, "reviewed_rules.csv")
            cands.save_reviewed_csv(csv_path)

            # RuleSet.from_csv only loads rows with status in ('approve', 'approved', 'yes', 'y')
            rs = RuleSet.from_csv(csv_path)
            self.assertIsNotNone(rs.group_of("mla"))
            self.assertIsNone(rs.group_of("kahan"))  # rejected -> ignored


class TestDiagnosisAndExplainability(unittest.TestCase):
    """8.3 Explainability and non-causal diagnosis."""

    def test_show_why_it_flipped_with_token_diffs(self):
        # Model that flips on 'acha' (missing 'c')
        def unstable_model(t):
            return "pos" if "accha" in t else "neg"

        flips = show_why_it_flipped("yeh bohot accha hai", unstable_model, n=5, seed=0)
        # Should have captured flips with exact token diffs
        if flips:
            f = flips[0]
            self.assertIn("original_text", f)
            self.assertIn("variant_text", f)
            self.assertIn("changed_tokens", f)
            self.assertIn("description", f)
            self.assertIn("not internal model reasoning", f["description"])

    def test_explain_report_flips_privacy_opt_in(self):
        # Create a report with flips
        dummy_flips = [
            {
                "original": "phone accha hai",
                "variant": "phone acha hai",
                "kind": "spelling",
                "original_pred": "pos",
                "variant_pred": "neg",
                "label": "pos",
            }
        ]
        rep = Report(
            n_messages=1,
            n_variants=1,
            accuracy_original=1.0,
            macro_f1_original=1.0,
            accuracy_variants=0.0,
            macro_f1_variants=0.0,
            flip_rate=1.0,
            flip_rate_ci=(0.5, 1.0),
            accuracy_drop=1.0,
            accuracy_drop_ci=(0.5, 1.0),
            by_kind={"spelling": {"variants": 1, "flips": 1, "flip_rate": 1.0, "accuracy": 0.0}},
            flips=dummy_flips,
            transitions={"pos -> neg": 1},
        )

        # Default: opt_in_examples=False -> privacy preserved, no raw text
        diag_private = explain_report_flips(rep, opt_in_examples=False)
        self.assertTrue(diag_private["examples_redacted"])
        self.assertEqual(len(diag_private["examples"]), 0)
        self.assertIn("privacy_note", diag_private)
        self.assertEqual(diag_private["total_flips"], 1)

        # Opt-in: opt_in_examples=True -> detailed inspection enabled
        diag_optin = explain_report_flips(rep, opt_in_examples=True)
        self.assertFalse(diag_optin["examples_redacted"])
        self.assertEqual(len(diag_optin["examples"]), 1)
        ex = diag_optin["examples"][0]
        self.assertEqual(ex["original_text"], "phone accha hai")
        self.assertIn("changed_tokens", ex)
        self.assertIn("not causal model reasoning", ex["description"])


class TestScriptAbstentionAndSchema(unittest.TestCase):
    """8.4 Script coverage abstention and report schema."""

    def test_unsupported_indic_scripts_abstain_to_review(self):
        dummy_model = lambda t: "pos"

        # Tamil text
        tamil_text = "இது ஒரு நல்ல தயாரிப்பு"
        dec_tamil = gate(dummy_model, tamil_text)
        self.assertEqual(dec_tamil.decision, "REVIEW")
        self.assertIn("unsupported script", dec_tamil.reason.lower())
        self.assertIn("tamil", dec_tamil.reason.lower())

        # Bengali text
        bengali_text = "এই পণ্যটি খুব ভাল"
        dec_bengali = gate(dummy_model, bengali_text)
        self.assertEqual(dec_bengali.decision, "REVIEW")
        self.assertIn("unsupported script", dec_bengali.reason.lower())
        self.assertIn("bengali", dec_bengali.reason.lower())

    def test_report_version_is_v080(self):
        dummy_model = lambda t: "pos"
        dataset = [("product accha hai", "pos"), ("bahut bura", "neg")]
        rep = test(dummy_model, dataset, n_variants=2, seed=0)
        self.assertEqual(rep.metadata["fliprate_version"], "0.8.0")
        self.assertEqual(rep.schema_version, "0.6.0")
        self.assertIn("protected_terms", rep.metadata)


if __name__ == "__main__":
    unittest.main()
