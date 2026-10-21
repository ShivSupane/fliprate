import unittest

import fliprate as fr
from fliprate.rules import ambiguous_tokens, is_safe_variant, negation_count


class PerturbTests(unittest.TestCase):
    def test_returns_distinct_variants_different_from_original(self):
        text = "mala order ajun milala nahi"
        vs = fr.perturb(text, n=8)
        self.assertTrue(0 < len(vs) <= 8)
        self.assertEqual(len({v.text for v in vs}), len(vs))
        self.assertNotIn(text, [v.text for v in vs])

    def test_deterministic_for_same_seed(self):
        a = fr.perturb("delivery bahut late hui", n=6, seed=3)
        b = fr.perturb("delivery bahut late hui", n=6, seed=3)
        self.assertEqual(a, b)

    def test_every_variant_is_safe(self):
        for text in ["mala product avadla nahi", "payment fail ho gaya", "I want to talk to a human"]:
            for v in fr.perturb(text, n=12, seed=1):
                self.assertTrue(is_safe_variant(text, v.text), (text, v.text))

    def test_negation_is_never_added_or_removed(self):
        text = "mala product avadla nahi"
        for v in fr.perturb(text, n=20):
            self.assertEqual(negation_count(v.text), negation_count(text))

    def test_ambiguous_nai_is_never_swapped(self):
        for v in fr.perturb("mujhe nai chahiye", n=20, kinds=["spelling"]):
            self.fail(f"spelling variant produced for text with only ambiguous words: {v.text}")

    def test_ambiguous_tokens_reported(self):
        self.assertEqual(ambiguous_tokens("mala he nai aavdla"), ["he", "nai"])

    def test_kind_filter_and_unknown_kind(self):
        self.assertTrue(all(v.kind == "case" for v in fr.perturb("Mala order", n=5, kinds=["case"])))
        with self.assertRaises(ValueError):
            fr.perturb("hello", kinds=["magic"])

    def test_empty_text(self):
        self.assertEqual(fr.perturb("   "), [])
        self.assertEqual(fr.perturb(""), [])

    def test_unsafe_change_detected(self):
        self.assertFalse(is_safe_variant("mala avadla nahi", "mala avadla"))
        self.assertFalse(is_safe_variant("order late", "order early"))


class NormalizeTests(unittest.TestCase):
    def test_known_words_and_repeats(self):
        r = fr.normalize("Mala ordeeer ajoon milala nhiii!!!")
        self.assertEqual(r.normalized, "mala order ajun milala nahi!")
        self.assertEqual(r.original, "Mala ordeeer ajoon milala nhiii!!!")

    def test_idempotent(self):
        once = fr.normalize("bohot achhaaa product hai!!").normalized
        self.assertEqual(fr.normalize(once).normalized, once)

    def test_never_changes_meaning(self):
        for text in ["mala product avadla nhi", "payment nahi hua", "nai chahiye mujhe"]:
            self.assertTrue(is_safe_variant(text, fr.normalize(text).normalized))

    def test_ambiguous_nai_untouched(self):
        self.assertIn("nai", fr.normalize("mujhe nai chahiye").normalized)

    def test_numbers_and_devanagari_untouched(self):
        self.assertEqual(fr.normalize("delivery 5000 din").normalized, "delivery 5000 din")
        self.assertIn("नहीं", fr.normalize("order नहीं aaya").normalized)


if __name__ == "__main__":
    unittest.main()
