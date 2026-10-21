import unittest

import fliprate as fr
from fliprate.demo import keyword_classifier, sample_dataset
from fliprate.rules import WORD_RE, word_key


def robust_model(text):
    """Ignores case and stretched letters, so it should never flip on safe variants."""
    return "neg" if any(word_key(w) == "bad" for w in WORD_RE.findall(text)) else "pos"


def fragile_model(text):  # changes its answer whenever a letter is stretched
    return "weird" if any(a == b == c for a, b, c in zip(text, text[1:], text[2:])) else "ok"


DATA = [("bad delivery late", "neg"), ("order accha", "pos"), ("payment bad", "neg"), ("quality accha", "pos")]


class MetricTests(unittest.TestCase):
    def test_macro_f1_known_value(self):
        y_true = ["a", "a", "b", "b"]
        y_pred = ["a", "b", "b", "b"]
        # a: P=1 R=.5 F1=2/3 ; b: P=2/3 R=1 F1=.8 -> mean = 0.7333
        self.assertAlmostEqual(fr.macro_f1(y_true, y_pred), (2 / 3 + 0.8) / 2, places=6)

    def test_perfect_prediction(self):
        self.assertEqual(fr.macro_f1(["a", "b"], ["a", "b"]), 1.0)


class TestEngine(unittest.TestCase):
    def test_robust_model_has_zero_flip_rate(self):
        r = fr.test(robust_model, DATA, n_variants=6, bootstrap=50)
        self.assertEqual(r.flip_rate, 0.0)
        self.assertEqual(r.accuracy_drop, 0.0)
        self.assertEqual(r.accuracy, 1.0)

    def test_fragile_model_is_caught(self):
        r = fr.test(fragile_model, DATA, n_variants=8, kinds=["repetition"], bootstrap=50)
        self.assertGreater(r.flip_rate, 0.5)
        self.assertTrue(r.flips)
        self.assertIn("repetition", r.by_kind)

    def test_ci_contains_point_estimate(self):
        r = fr.test(keyword_classifier, sample_dataset(), n_variants=4, bootstrap=200)
        lo, hi = r.flip_rate_ci
        self.assertLessEqual(lo, r.flip_rate + 1e-9)
        self.assertGreaterEqual(hi, r.flip_rate - 1e-9)

    def test_events_stream_in_order(self):
        events = []
        fr.test(robust_model, DATA, n_variants=3, bootstrap=20, on_event=events.append)
        stages = [e["stage"] for e in events]
        self.assertEqual(stages[0], "started")
        self.assertEqual(stages[-1], "done")
        self.assertEqual(stages.count("progress"), len(DATA))
        self.assertEqual(events[-2]["done"], len(DATA))

    def test_reproducible(self):
        a = fr.test(keyword_classifier, sample_dataset(), n_variants=3, seed=7, bootstrap=20)
        b = fr.test(keyword_classifier, sample_dataset(), n_variants=3, seed=7, bootstrap=20)
        self.assertEqual(a.flip_rate, b.flip_rate)

    def test_dataset_loading_variants(self):
        self.assertEqual(fr.load_dataset([{"text": "a b", "label": "x"}]), [("a b", "x")])
        with self.assertRaises(ValueError):
            fr.load_dataset([])
        with self.assertRaises(ValueError):
            fr.read_csv_text("foo,bar\n1,2\n")

    def test_summary_mentions_caveat(self):
        r = fr.test(robust_model, DATA, n_variants=2, bootstrap=20)
        self.assertIn("not correctness", r.summary())


if __name__ == "__main__":
    unittest.main()
