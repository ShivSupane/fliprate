import tempfile
import unittest
from unittest.mock import MagicMock, patch

import fliprate as fr


class HappyPathApiTests(unittest.TestCase):
    def setUp(self):
        self.model = lambda t: "positive" if "accha" in t else "negative"

    def test_quick_check_returns_clean_verdict(self):
        res = fr.quick_check(self.model, "product bohot accha hai")
        self.assertTrue(res.is_safe)
        self.assertEqual(res.decision, "PASS")
        self.assertEqual(res.prediction, "positive")
        self.assertIn("SAFE", str(res))

    def test_normalize_text_returns_string(self):
        clean = fr.normalize_text("Mala product avadlaaa nhi", lowercase=False)
        self.assertEqual(clean, "Mala product avadla nahi")

    def test_safe_predict_transforms_unstable_text(self):
        # A brittle model that only matches canonical 'nahi'
        def brittle(t):
            return "neg" if "nahi" in t else "pos"

        pred = fr.safe_predict(brittle, "mala delivery nhi milali")
        self.assertEqual(pred, "neg")

    def test_check_model_runs_evaluation(self):
        rows = [("product accha hai", "pos"), ("delivery nahi aayi", "neg")]
        report = fr.check_model(self.model, rows, n_variants=2, bootstrap=10)
        self.assertEqual(report.n_messages, 2)
        self.assertGreaterEqual(report.accuracy_original, 0.0)


class ProductionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.model = lambda t: "pos" if "accha" in t else "neg"

    def test_runtime_guard_lenient_vs_strict(self):
        # Input that stabilizes under normalization
        text = "product bohot accha hai"
        res_lenient = fr.runtime_guard(self.model, text, mode="lenient")
        self.assertFalse(res_lenient.should_route_to_human)
        self.assertEqual(res_lenient.prediction, "pos")

        res_strict = fr.runtime_guard(self.model, text, mode="strict")
        # If PASS, neither flags route to human
        if res_strict.decision == "PASS":
            self.assertFalse(res_strict.should_route_to_human)

    def test_batch_guard_processes_multiple_texts(self):
        texts = ["product accha hai", "delivery nahi aayi"]
        results = fr.batch_guard(self.model, texts, n_variants=2)
        self.assertEqual(len(results), 2)
        self.assertIn(results[0].decision, ("PASS", "TRANSFORM", "REVIEW"))

    def test_evaluate_pipeline_saves_report(self):
        rows = [("product accha hai", "pos"), ("delivery nahi aayi", "neg")]
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        report = fr.evaluate_pipeline(self.model, rows, save_path=path, n_variants=2, bootstrap=10)
        self.assertEqual(report.n_messages, 2)
        loaded = fr.Report.load(path)
        self.assertEqual(loaded.n_messages, 2)

    def test_make_protected_model(self):
        seen = []
        def my_fn(t):
            seen.append(t)
            return "ok"
        protected = fr.make_protected_model(my_fn)
        protected("Mala ordeeer nhiii")
        self.assertEqual(seen, ["mala order nahi"])


class ModelAdapterTests(unittest.TestCase):
    def test_callable_adapter(self):
        adapter = fr.as_adapter(lambda t: "label_" + t[:3])
        self.assertEqual(adapter.predict("hello"), "label_hel")
        self.assertEqual(adapter("world"), "label_wor")
        self.assertEqual(adapter.predict_batch(["one", "two"]), ["label_one", "label_two"])
        self.assertEqual(adapter.metadata()["adapter_type"], "CallableAdapter")

    def test_as_adapter_idempotent(self):
        a1 = fr.CallableAdapter(lambda t: "ok")
        a2 = fr.as_adapter(a1)
        self.assertIs(a1, a2)

    @patch("urllib.request.urlopen")
    def test_http_service_adapter(self, mock_urlopen):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'{"prediction": "sentiment_pos"}'
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        http_adapter = fr.HttpServiceAdapter("http://fake-model.internal/predict")
        pred = http_adapter.predict("great service")
        self.assertEqual(pred, "sentiment_pos")
        self.assertEqual(http_adapter.metadata()["endpoint_url"], "http://fake-model.internal/predict")


class RuleManagementTests(unittest.TestCase):
    def test_load_rules_by_language_name_and_code(self):
        r_hi = fr.load_rules("hindi")
        self.assertGreater(len(r_hi.groups), 70)

        r_mr = fr.load_rules("mr")
        self.assertGreater(len(r_mr.groups), 70)

        with self.assertRaises(ValueError):
            fr.load_rules("unsupported_language_xyz")

    def test_add_custom_rule(self):
        custom = fr.add_custom_rule(["discnt", "dscnt", "discount"], canonical="discount")
        self.assertIn("dscnt", custom.key_to_canon)
        self.assertEqual(custom.key_to_canon["dscnt"], "discount")

    def test_preview_variants(self):
        variants = fr.preview_variants("delivery nahi aaya", n=5)
        self.assertIsInstance(variants, list)
        self.assertGreater(len(variants), 0)

    def test_describe_rule_group(self):
        desc = fr.describe_rule_group("nhi")
        self.assertIsNotNone(desc)
        self.assertEqual(desc["canonical"], "nahi")
        self.assertIn("nahi", desc["variants"])
        self.assertTrue(desc["is_negation"])

        self.assertIsNone(fr.describe_rule_group("completely_unknown_token_xyz"))


class ExplainabilityTests(unittest.TestCase):
    def test_explain_gate_decision(self):
        model = lambda t: "pos" if "accha" in t else "neg"
        explanation = fr.explain_gate_decision("product bohot accha hai", model, n_variants=3)
        self.assertIn("decision", explanation)
        self.assertIn("summary", explanation)
        self.assertIn("recommendation", explanation)
        self.assertIn(explanation["script"], ("latin", "devanagari", "mixed", "other"))

    def test_show_why_it_flipped(self):
        # Model sensitive to any perturbation
        def sensitive(t):
            return "original" if t == "delivery nahi mila" else "flipped"

        flips = fr.show_why_it_flipped("delivery nahi mila", sensitive, n=5)
        self.assertIsInstance(flips, list)
        if flips:
            self.assertEqual(flips[0]["original_prediction"], "original")
            self.assertEqual(flips[0]["variant_prediction"], "flipped")

    def test_preview_normalized_variants(self):
        previews = fr.preview_normalized_variants("delivery nahi mila", n=3)
        self.assertIsInstance(previews, list)
        self.assertGreater(len(previews), 0)
        self.assertIn("normalized", previews[0])

    def test_print_rule_hits(self):
        hits = fr.print_rule_hits("mala product avadlaaa nahi")
        tokens = [h["token"] for h in hits]
        self.assertIn("nahi", tokens)


class ValidationTests(unittest.TestCase):
    def test_validate_text(self):
        self.assertEqual(fr.validate_text("valid input"), "valid input")
        with self.assertRaises(TypeError):
            fr.validate_text(123)
        with self.assertRaises(ValueError):
            fr.validate_text("   ")
        with self.assertRaises(ValueError):
            fr.validate_text("abc", max_length=2)

    def test_validate_dataset(self):
        clean_data = [("text a", "pos"), ("text b", "neg")]
        res = fr.validate_dataset(clean_data)
        self.assertTrue(res["valid"])
        self.assertEqual(res["row_count"], 2)

        conflicting_data = [("text a", "pos"), ("text a", "neg")]
        res_conf = fr.validate_dataset(conflicting_data)
        self.assertFalse(res_conf["valid"])
        self.assertIn("text a", res_conf["label_conflicts"])


if __name__ == "__main__":
    unittest.main()
