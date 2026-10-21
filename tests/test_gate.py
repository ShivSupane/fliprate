import unittest

import fliprate as fr
from fliprate.demo import keyword_classifier


def strict_spelling_model(text):
    """Only understands canonical spellings, like a model trained on clean text."""
    return "neg" if "nahi" in text.split() else "pos"


class GateTests(unittest.TestCase):
    def test_pass_when_stable(self):
        r = fr.gate(lambda t: "same", "mala order ajun milala nahi")
        self.assertEqual(r.decision, "PASS")
        self.assertEqual(r.raw_stability, 1.0)

    def test_transform_when_normalization_fixes_instability(self):
        r = fr.gate(keyword_classifier, "mala product avadlaaa nahi", threshold=0.9)
        self.assertEqual(r.decision, "TRANSFORM", r.to_dict())
        self.assertEqual(r.original_text, "mala product avadlaaa nahi")
        self.assertEqual(r.text_used, "mala product avadla nahi")

    def test_review_when_non_latin(self):
        r = fr.gate(lambda t: "x", "मला ऑर्डर अजून मिळाली नाही")
        self.assertEqual(r.decision, "REVIEW")

    def test_review_when_empty(self):
        self.assertEqual(fr.gate(lambda t: "x", "  ").decision, "REVIEW")

    def test_review_when_unfixable(self):
        import random
        rng = random.Random(0)
        r = fr.gate(lambda t: rng.random() < 0.5, "mala order ajun milala nahi")
        self.assertEqual(r.decision, "REVIEW")

    def test_ambiguity_warning(self):
        r = fr.gate(lambda t: "same", "mala he nai aavdla")
        self.assertTrue(any("ambiguous" in w for w in r.warnings))

    def test_events_end_with_done_and_gate(self):
        ev = []
        fr.gate(lambda t: "same", "delivery late", on_event=ev.append)
        stages = [e["stage"] for e in ev]
        self.assertIn("gate", stages)
        self.assertEqual(stages[-1], "done")

    def test_sse_format(self):
        s = fr.to_sse(fr.make_event("x", a=1))
        self.assertTrue(s.startswith("data: ") and s.endswith("\n\n"))


if __name__ == "__main__":
    unittest.main()
