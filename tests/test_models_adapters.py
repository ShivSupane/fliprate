import unittest

import fliprate as fr


class NormalizeCaseTests(unittest.TestCase):
    def test_lowercase_false_preserves_original_case(self):
        out = fr.normalize("Mala order avadlaaa nahi", lowercase=False)
        self.assertEqual(out.normalized, "Mala order avadla nahi")

    def test_guard_wrapper_runs_gate(self):
        result = fr.guard(lambda t: "same", "mala order ajun milala nahi")
        self.assertEqual(result.decision, "PASS")

    def test_audit_wrapper_returns_report(self):
        rows = [("mala avadla nahi", "neg"), ("bahut accha", "pos")]
        report = fr.audit(lambda t: "pos" if "accha" in t else "neg", rows, n_variants=2, bootstrap=20)
        self.assertEqual(report.n_messages, 2)


if __name__ == "__main__":
    unittest.main()
