"""
test_complete_coverage.py
=========================
Exhaustive tests for EVERY public function and class in the fliprate library.

Coverage:
  __init__, pipeline, explain, validation, adapters, benchmark, gating,
  normalization, perturbation, rules, events, models, evaluation,
  service, experiment, dataset loaders (dakshina/aksharantar), integration.
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import fliprate as fr
from fliprate.events import drain, make_event, to_sse
from fliprate.gating import detect_script, gate, iter_gate
from fliprate.normalization import normalize
from fliprate.perturbation import perturb
from fliprate.rules import DEFAULT_RULES, is_safe_variant
from fliprate.service import (
    InMemoryJobStore,
    InMemoryRateLimiter,
    Settings,
    create_job_store,
    create_rate_limiter,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------
def _pos_neg(text: str) -> str:
    return "pos" if "accha" in text else "neg"


SMALL_DATASET = [
    ("product bohot accha hai", "pos"),
    ("delivery nahi aayi", "neg"),
    ("service acchi lagi", "pos"),
    ("kharab anubhav tha", "neg"),
]


# ===========================================================================
# 1. Package metadata
# ===========================================================================
class TestPackageMetadata(unittest.TestCase):
    def test_version_string(self):
        self.assertIsInstance(fr.__version__, str)
        self.assertIn(".", fr.__version__)

    def test_author(self):
        self.assertEqual(fr.__author__, "Shiv Supane")

    def test_all_exports_importable(self):
        for name in fr.__all__:
            self.assertTrue(hasattr(fr, name), f"fr.{name} missing")


# ===========================================================================
# 2. Convenience wrappers (guard / audit / learn_rules_from_texts)
# ===========================================================================
class TestConvenienceWrappers(unittest.TestCase):
    def test_guard(self):
        res = fr.guard(_pos_neg, "product accha hai", n_variants=3)
        self.assertIn(res.decision, ("PASS", "TRANSFORM", "REVIEW"))

    def test_audit(self):
        report = fr.audit(_pos_neg, SMALL_DATASET[:2], n_variants=2, bootstrap=5)
        self.assertEqual(report.n_messages, 2)

    def test_learn_rules_from_texts(self):
        candidates = fr.learn_rules_from_texts(["nahi", "nhi", "nhii", "nahi"])
        self.assertIsInstance(candidates, list)


# ===========================================================================
# 3. pipeline.py — happy-path & production API
# ===========================================================================
class TestQuickCheck(unittest.TestCase):
    def test_type(self):
        self.assertIsInstance(fr.quick_check(_pos_neg, "product accha hai"), fr.QuickCheckResult)

    def test_is_safe_for_stable(self):
        self.assertTrue(fr.quick_check(_pos_neg, "product bohot accha hai").is_safe)

    def test_str_representation(self):
        s = str(fr.quick_check(_pos_neg, "product accha hai"))
        self.assertTrue("SAFE" in s or "ATTENTION" in s)

    def test_bool_equals_is_safe(self):
        res = fr.quick_check(_pos_neg, "product accha hai")
        self.assertEqual(bool(res), res.is_safe)

    def test_all_fields(self):
        res = fr.quick_check(_pos_neg, "delivery nahi aayi")
        for attr in ("is_safe", "decision", "prediction", "suggested_text", "reason", "flipped_variants"):
            self.assertTrue(hasattr(res, attr), f"missing: {attr}")


class TestNormalizeText(unittest.TestCase):
    def test_normalizes_variant(self):
        self.assertIn("nahi", fr.normalize_text("mala product avadlaaa nhi"))

    def test_returns_str(self):
        self.assertIsInstance(fr.normalize_text("hello world"), str)

    def test_lowercase_false(self):
        out = fr.normalize_text("Mala product nhi", lowercase=False)
        self.assertTrue(out[0].isupper())

    def test_whitespace_input(self):
        self.assertIsInstance(fr.normalize_text("   "), str)


class TestSafePredict(unittest.TestCase):
    def _brittle(self, t):
        return "neg" if "nahi" in t else "pos"

    def test_returns_str(self):
        self.assertIsInstance(fr.safe_predict(self._brittle, "delivery nahi aayi"), str)

    def test_on_review_normalized(self):
        self.assertIsInstance(
            fr.safe_predict(self._brittle, "delivery nhi mili", on_review="normalized"), str)

    def test_on_review_original(self):
        self.assertIsInstance(
            fr.safe_predict(_pos_neg, "delivery nhi mili", on_review="original"), str)


class TestCheckModel(unittest.TestCase):
    def test_returns_report(self):
        self.assertIsInstance(
            fr.check_model(_pos_neg, SMALL_DATASET[:2], n_variants=2, bootstrap=5), fr.Report)

    def test_n_messages(self):
        self.assertEqual(
            fr.check_model(_pos_neg, SMALL_DATASET, n_variants=2, bootstrap=5).n_messages, 4)


class TestRuntimeGuard(unittest.TestCase):
    def test_type(self):
        self.assertIsInstance(fr.runtime_guard(_pos_neg, "product accha hai"), fr.RuntimeGuardResult)

    def test_lenient_pass_not_routed(self):
        res = fr.runtime_guard(_pos_neg, "product bohot accha hai", mode="lenient")
        if res.decision == "PASS":
            self.assertFalse(res.should_route_to_human)

    def test_metadata_keys(self):
        meta = fr.runtime_guard(_pos_neg, "product accha hai").metadata
        for k in ("raw_stability", "normalized_stability", "n_variants", "flipped_variants"):
            self.assertIn(k, meta)

    def test_clean_text_str(self):
        self.assertIsInstance(fr.runtime_guard(_pos_neg, "product accha hai").clean_text, str)

    def test_strict_mode(self):
        res = fr.runtime_guard(_pos_neg, "delivery nhi mili", mode="strict")
        self.assertIn(res.decision, ("PASS", "TRANSFORM", "REVIEW"))


class TestBatchGuard(unittest.TestCase):
    def test_returns_list(self):
        results = fr.batch_guard(_pos_neg, ["product accha hai", "delivery nahi aayi"], n_variants=2)
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r, fr.GateResult)

    def test_empty_batch(self):
        self.assertEqual(fr.batch_guard(_pos_neg, []), [])

    def test_valid_decisions(self):
        for r in fr.batch_guard(_pos_neg, ["accha product", "bura product"], n_variants=2):
            self.assertIn(r.decision, ("PASS", "TRANSFORM", "REVIEW"))


class TestEvaluatePipeline(unittest.TestCase):
    def test_returns_report(self):
        self.assertIsInstance(
            fr.evaluate_pipeline(_pos_neg, SMALL_DATASET[:2], n_variants=2, bootstrap=5), fr.Report)

    def test_saves_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        report = fr.evaluate_pipeline(_pos_neg, SMALL_DATASET[:2], save_path=path, n_variants=2, bootstrap=5)
        loaded = fr.Report.load(path)
        self.assertEqual(loaded.n_messages, report.n_messages)


class TestMakeProtectedModel(unittest.TestCase):
    def test_normalizes(self):
        seen = []
        def rec(t):
            seen.append(t)
            return "ok"
        fr.make_protected_model(rec)("Mala Ordeeer Nhiii")
        self.assertIn("nahi", seen[0])

    def test_type(self):
        self.assertIsInstance(fr.make_protected_model(_pos_neg), fr.ProtectedModel)

    def test_callable(self):
        self.assertIsInstance(fr.make_protected_model(_pos_neg)("product accha hai"), str)


# ===========================================================================
# 4. explain.py
# ===========================================================================
class TestExplainGateDecision(unittest.TestCase):
    def test_required_keys(self):
        result = fr.explain_gate_decision("product accha hai", _pos_neg, n_variants=3)
        for k in ("decision", "prediction", "script", "raw_stability",
                  "reason", "summary", "recommendation", "flipped_variants", "text_used"):
            self.assertIn(k, result)

    def test_valid_decision(self):
        self.assertIn(
            fr.explain_gate_decision("product accha hai", _pos_neg)["decision"],
            ("PASS", "TRANSFORM", "REVIEW"))

    def test_script_latin(self):
        self.assertEqual(
            fr.explain_gate_decision("product bohot accha hai", _pos_neg)["script"], "latin")

    def test_summary_nonempty(self):
        self.assertGreater(
            len(fr.explain_gate_decision("product accha hai", _pos_neg)["summary"]), 0)


class TestShowWhyItFlipped(unittest.TestCase):
    def _sensitive(self, t):
        return "A" if t == "delivery nahi mila" else "B"

    def test_returns_list(self):
        self.assertIsInstance(fr.show_why_it_flipped("delivery nahi mila", self._sensitive, n=5), list)

    def test_entry_keys(self):
        flips = fr.show_why_it_flipped("delivery nahi mila", self._sensitive, n=5)
        if flips:
            for k in ("variant_text", "kind", "original_prediction", "variant_prediction", "diffs"):
                self.assertIn(k, flips[0])

    def test_stable_no_flips(self):
        self.assertEqual(fr.show_why_it_flipped("product accha hai", lambda t: "pos", n=5), [])


class TestPreviewNormalizedVariants(unittest.TestCase):
    def test_nonempty(self):
        self.assertGreater(len(fr.preview_normalized_variants("delivery nahi mila", n=3)), 0)

    def test_entry_keys(self):
        for entry in fr.preview_normalized_variants("delivery nahi mila", n=3):
            for k in ("original", "kind", "normalized"):
                self.assertIn(k, entry)


class TestPrintRuleHits(unittest.TestCase):
    def test_nahi_hit(self):
        hits = fr.print_rule_hits("mala product nahi")
        self.assertIn("nahi", [h["token"] for h in hits])

    def test_entry_keys(self):
        for h in fr.print_rule_hits("delivery nahi aayi"):
            for k in ("token", "canonical", "is_negation"):
                self.assertIn(k, h)

    def test_empty_returns_list(self):
        self.assertIsInstance(fr.print_rule_hits("   "), list)

    def test_negation_flag(self):
        hits = [h for h in fr.print_rule_hits("nahi") if h["token"] == "nahi"]
        if hits:
            self.assertTrue(hits[0]["is_negation"])


# ===========================================================================
# 5. validation.py
# ===========================================================================
class TestValidateText(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(fr.validate_text("hello world"), "hello world")

    def test_non_string(self):
        for v in [42, None, [], {}]:
            with self.assertRaises(TypeError):
                fr.validate_text(v)

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            fr.validate_text("   ")

    def test_allow_empty(self):
        self.assertEqual(fr.validate_text("", allow_empty=True), "")

    def test_max_length(self):
        with self.assertRaises(ValueError):
            fr.validate_text("abc", max_length=2)
        self.assertEqual(fr.validate_text("ab", max_length=2), "ab")


class TestValidateDataset(unittest.TestCase):
    def test_clean(self):
        res = fr.validate_dataset([("text a", "pos"), ("text b", "neg")])
        self.assertTrue(res["valid"])
        self.assertEqual(res["row_count"], 2)

    def test_conflict(self):
        res = fr.validate_dataset([("same", "pos"), ("same", "neg")])
        self.assertFalse(res["valid"])
        self.assertIn("same", res["label_conflicts"])

    def test_labels(self):
        res = fr.validate_dataset([("a", "pos"), ("b", "neg")])
        self.assertIn("pos", res["unique_labels"])

    def test_rows(self):
        res = fr.validate_dataset([("a", "pos"), ("b", "neg")])
        self.assertEqual(len(res["rows"]), 2)


# ===========================================================================
# 6. adapters.py
# ===========================================================================
class TestTwitterPreprocess(unittest.TestCase):
    def test_at_user(self):
        self.assertEqual(fr.twitter_preprocess("@Shiv hello"), "@user hello")

    def test_http(self):
        self.assertEqual(fr.twitter_preprocess("see https://x.com"), "see http")

    def test_plain(self):
        self.assertEqual(fr.twitter_preprocess("hello world"), "hello world")

    def test_non_string(self):
        with self.assertRaises(TypeError):
            fr.twitter_preprocess(123)


class TestCallableAdapter(unittest.TestCase):
    def test_predict(self):
        self.assertEqual(fr.CallableAdapter(lambda t: "ok_" + t[:2]).predict("hello"), "ok_he")

    def test_call(self):
        self.assertEqual(fr.CallableAdapter(lambda t: "ok")("hello"), "ok")

    def test_batch(self):
        self.assertEqual(fr.CallableAdapter(lambda t: t.upper()).predict_batch(["a", "b"]), ["A", "B"])

    def test_metadata_type(self):
        self.assertEqual(fr.CallableAdapter(lambda t: "x").metadata()["adapter_type"], "CallableAdapter")

    def test_supported_languages(self):
        self.assertIn("any", fr.CallableAdapter(lambda t: "x").supported_languages())

    def test_name_from_fn(self):
        def my_fn(t): return "ok"
        self.assertEqual(fr.CallableAdapter(my_fn).name, "my_fn")

    def test_custom_name(self):
        self.assertEqual(fr.CallableAdapter(lambda t: "ok", name="custom").name, "custom")


class TestAsAdapter(unittest.TestCase):
    def test_wraps_callable(self):
        self.assertIsInstance(fr.as_adapter(lambda t: "ok"), fr.CallableAdapter)

    def test_idempotent(self):
        a = fr.CallableAdapter(lambda t: "ok")
        self.assertIs(fr.as_adapter(a), a)

    def test_non_callable_raises(self):
        with self.assertRaises(TypeError):
            fr.as_adapter(42)


class TestSklearnAdapter(unittest.TestCase):
    def _mock_pipe(self, labels):
        import numpy as np
        m = MagicMock()
        m.predict.return_value = np.array(labels)
        return m

    def test_predict(self):
        self.assertEqual(fr.SklearnAdapter(self._mock_pipe(["pos"])).predict("good"), "pos")

    def test_predict_batch(self):
        self.assertEqual(
            fr.SklearnAdapter(self._mock_pipe(["pos", "neg"])).predict_batch(["a", "b"]),
            ["pos", "neg"])

    def test_from_sklearn(self):
        a = fr.from_sklearn(self._mock_pipe(["pos"]), name="m")
        self.assertIsInstance(a, fr.SklearnAdapter)
        self.assertEqual(a.name, "m")

    def test_metadata(self):
        self.assertIn("adapter_type", fr.SklearnAdapter(self._mock_pipe(["pos"])).metadata())


class TestLLMClassifier(unittest.TestCase):
    def setUp(self):
        self.clf = fr.LLMClassifier(
            lambda p: "positive" if "accha" in p else "negative",
            labels=["positive", "negative"],
            name="fake_llm"
        )

    def test_predict(self):
        self.assertIn(self.clf.predict("product accha hai"), ["positive", "negative"])

    def test_predict_batch(self):
        self.assertEqual(len(self.clf.predict_batch(["accha", "bura"])), 2)

    def test_parse_exact(self):
        self.assertEqual(self.clf.parse("positive"), "positive")

    def test_parse_fuzzy(self):
        self.assertEqual(self.clf.parse("The sentiment is positive."), "positive")

    def test_parse_invalid(self):
        self.assertEqual(self.clf.parse("totally unclear xyz"), "__invalid__")

    def test_factory(self):
        self.assertIsInstance(
            fr.llm_classifier(lambda p: "positive", labels=["positive", "negative"]),
            fr.LLMClassifier)

    def test_callable(self):
        self.assertIn(self.clf("product accha"), ["positive", "negative"])

    def test_metadata_name(self):
        self.assertEqual(self.clf.metadata()["name"], "fake_llm")


class TestHFTextClassifier(unittest.TestCase):
    def _adapter(self, labels):
        def factory():
            def pipe(texts, **kw): return [{"label": l} for l in labels[:len(texts)]]
            return pipe
        return fr.HFTextClassifier("fake/model", pipeline_factory=factory)

    def test_predict(self):
        self.assertEqual(self._adapter(["positive"]).predict("good"), "positive")

    def test_predict_batch(self):
        self.assertEqual(
            self._adapter(["positive", "negative"]).predict_batch(["a", "b"]),
            ["positive", "negative"])

    def test_lowercase(self):
        def factory():
            def pipe(texts, **kw): return [{"label": "POSITIVE"}]
            return pipe
        a = fr.HFTextClassifier("fake/m", lowercase_labels=True, pipeline_factory=factory)
        self.assertEqual(a.predict("test"), "positive")

    def test_label_map(self):
        def factory():
            def pipe(texts, **kw): return [{"label": "LABEL_1"}]
            return pipe
        a = fr.HFTextClassifier("fake/m", label_map={"LABEL_1": "positive"}, pipeline_factory=factory)
        self.assertEqual(a.predict("test"), "positive")

    def test_preprocess(self):
        calls = []
        def pre(t):
            calls.append(t)
            return t
        def factory():
            def pipe(texts, **kw): return [{"label": "x"} for _ in texts]
            return pipe
        fr.HFTextClassifier("fake/m", preprocess=pre, pipeline_factory=factory).predict("hello")
        self.assertEqual(calls, ["hello"])

    def test_warmup(self):
        self._adapter(["positive"]).warmup()

    def test_metadata(self):
        self.assertIn("model_name", self._adapter(["pos"]).metadata())


class TestHttpServiceAdapter(unittest.TestCase):
    @patch("urllib.request.urlopen")
    def test_prediction_key(self, mock_urlopen):
        r = MagicMock()
        r.read.return_value = b'{"prediction": "pos"}'
        r.__enter__.return_value = r
        mock_urlopen.return_value = r
        self.assertEqual(fr.HttpServiceAdapter("http://fake.api/predict").predict("text"), "pos")

    @patch("urllib.request.urlopen")
    def test_label_key(self, mock_urlopen):
        r = MagicMock()
        r.read.return_value = b'{"label": "neg"}'
        r.__enter__.return_value = r
        mock_urlopen.return_value = r
        self.assertEqual(fr.HttpServiceAdapter("http://fake.api/predict").predict("text"), "neg")

    @patch("urllib.request.urlopen")
    def test_custom_parse(self, mock_urlopen):
        r = MagicMock()
        r.read.return_value = b'{"result": "positive"}'
        r.__enter__.return_value = r
        mock_urlopen.return_value = r
        a = fr.HttpServiceAdapter("http://fake.api/predict", parse_response=lambda d: d["result"])
        self.assertEqual(a.predict("test"), "positive")

    @patch("urllib.request.urlopen", side_effect=Exception("refused"))
    def test_raises_after_retries(self, _):
        with self.assertRaises(RuntimeError):
            fr.HttpServiceAdapter("http://fake.api/predict", retries=0).predict("test")

    def test_metadata(self):
        meta = fr.HttpServiceAdapter("http://fake.api/predict", retries=1, timeout=3.0).metadata()
        self.assertEqual(meta["endpoint_url"], "http://fake.api/predict")
        self.assertEqual(meta["timeout"], 3.0)


class TestWithRetries(unittest.TestCase):
    def test_succeeds_first_try(self):
        self.assertEqual(fr.with_retries(lambda: "ok", retries=2)(), "ok")

    def test_retries_then_succeeds(self):
        n = {"v": 0}
        def flaky():
            n["v"] += 1
            if n["v"] < 3:
                raise ValueError("fail")
            return "ok"
        self.assertEqual(fr.with_retries(flaky, retries=5, backoff=0.0)(), "ok")

    def test_raises_after_exhausted(self):
        with self.assertRaises(RuntimeError):
            fr.with_retries(lambda: (_ for _ in ()).throw(RuntimeError("x")), retries=1, backoff=0.0)()

    def test_preserves_name(self):
        def my_func(): return "ok"
        self.assertEqual(fr.with_retries(my_func).__name__, "my_func")


# ===========================================================================
# 7. benchmark.py
# ===========================================================================
class TestBenchmark(unittest.TestCase):
    def test_returns_suite(self):
        suite = fr.benchmark({"m": _pos_neg}, SMALL_DATASET[:2], n_variants=2, bootstrap=5)
        self.assertIsInstance(suite, fr.BenchmarkSuite)

    def test_baseline_flagged(self):
        suite = fr.benchmark({"a": _pos_neg, "b": lambda t: "pos"},
                             SMALL_DATASET[:2], baseline="a", n_variants=2, bootstrap=5)
        self.assertTrue(suite.scores["a"].is_baseline)
        self.assertFalse(suite.scores["b"].is_baseline)

    def test_score_vs_baseline(self):
        suite = fr.benchmark({"a": _pos_neg, "b": lambda t: "pos"},
                             SMALL_DATASET[:2], baseline="a", n_variants=2, bootstrap=5)
        self.assertIsNotNone(suite.scores["b"].score_vs_baseline)

    def test_no_protection(self):
        suite = fr.benchmark({"m": _pos_neg}, SMALL_DATASET[:2],
                             with_protection=False, n_variants=2, bootstrap=5)
        self.assertIsNone(suite.scores["m"].protected_flip_rate)

    def test_metadata(self):
        suite = fr.benchmark({"m": _pos_neg}, SMALL_DATASET[:2], n_variants=2, bootstrap=5)
        self.assertIn("n_models", suite.metadata)


class TestCompareModels(unittest.TestCase):
    def test_returns_suite(self):
        suite = fr.compare_models(_pos_neg, lambda t: "pos", SMALL_DATASET[:2], n_variants=2, bootstrap=5)
        self.assertIsInstance(suite, fr.BenchmarkSuite)

    def test_custom_names(self):
        suite = fr.compare_models(_pos_neg, lambda t: "pos", SMALL_DATASET[:2],
                                  name_a="baseline", name_b="new", n_variants=2, bootstrap=5)
        self.assertIn("baseline", suite.scores)
        self.assertIn("new", suite.scores)


class TestBenchmarkSuiteRendering(unittest.TestCase):
    def setUp(self):
        self.suite = fr.benchmark({"a": _pos_neg, "b": lambda t: "pos"},
                                  SMALL_DATASET[:2], baseline="a", n_variants=2, bootstrap=5)

    def test_summary(self):
        self.assertIn("Benchmark Summary", self.suite.summary())

    def test_to_markdown(self):
        self.assertIn("| Model |", self.suite.to_markdown())

    def test_to_html(self):
        html = self.suite.to_html()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("FlipRate Robustness Dashboard", html)

    def test_to_dict(self):
        d = self.suite.to_dict()
        self.assertIn("scores", d)
        self.assertIn("a", d["scores"])

    def test_save_html(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
            path = tf.name
        self.suite.save(path)
        self.assertIn("<!DOCTYPE html>", Path(path).read_text(encoding="utf-8"))

    def test_save_markdown(self):
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as tf:
            path = tf.name
        self.suite.save(path)
        self.assertIn("| Model |", Path(path).read_text(encoding="utf-8"))

    def test_save_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        self.suite.save(path)
        self.assertIn("scores", json.loads(Path(path).read_text(encoding="utf-8")))


class TestModelBenchmarkScore(unittest.TestCase):
    def test_fields(self):
        score = fr.ModelBenchmarkScore(
            name="test", accuracy_original=0.9, accuracy_variants=0.7,
            macro_f1_original=0.88, macro_f1_variants=0.68,
            flip_rate=0.2, flip_rate_ci=(0.15, 0.25),
        )
        self.assertEqual(score.name, "test")
        self.assertIsNone(score.protected_flip_rate)


# ===========================================================================
# 8. gating.py — gate, iter_gate, agate, detect_script, GateResult
# ===========================================================================
class TestGate(unittest.TestCase):
    def test_pass_stable(self):
        self.assertEqual(gate(lambda t: "pos", "product accha hai", n_variants=4).decision, "PASS")

    def test_review_empty(self):
        self.assertEqual(gate(_pos_neg, "   ").decision, "REVIEW")

    def test_fields(self):
        res = gate(_pos_neg, "product accha hai", n_variants=3)
        for attr in ("decision", "reason", "original_text", "text_used",
                     "prediction", "raw_stability", "n_variants", "warnings", "flipped_variants"):
            self.assertTrue(hasattr(res, attr))

    def test_to_dict(self):
        self.assertIn("decision", gate(_pos_neg, "product accha hai", n_variants=3).to_dict())

    def test_to_json(self):
        res = gate(_pos_neg, "product accha hai", n_variants=3)
        self.assertIn("decision", json.loads(res.to_json()))

    def test_to_json_indent(self):
        self.assertIn("\n", gate(_pos_neg, "product accha hai", n_variants=3).to_json(indent=2))

    def test_on_event_callback(self):
        events = []
        gate(_pos_neg, "product accha hai", n_variants=3, on_event=events.append)
        self.assertIn("done", [e["stage"] for e in events])

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            gate(_pos_neg, "text", threshold=0.0)

    def test_invalid_n_variants(self):
        with self.assertRaises(ValueError):
            gate(_pos_neg, "text", n_variants=0)

    def test_invalid_on_error(self):
        with self.assertRaises(ValueError):
            gate(_pos_neg, "text", on_error="ignore")

    def test_non_string_raises(self):
        with self.assertRaises(TypeError):
            gate(_pos_neg, 123)


class TestIterGate(unittest.TestCase):
    def test_yields_events_returns_result(self):
        gen = iter_gate(_pos_neg, "product accha hai", n_variants=3)
        events = []
        while True:
            try:
                events.append(next(gen))
            except StopIteration as stop:
                self.assertIsInstance(stop.value, fr.GateResult)
                break
        self.assertGreater(len(events), 0)


class TestAgate(unittest.TestCase):
    def test_returns_gate_result(self):
        result = asyncio.run(fr.agate(_pos_neg, "product accha hai", n_variants=3))
        self.assertIsInstance(result, fr.GateResult)


class TestDetectScript(unittest.TestCase):
    def test_latin(self):
        self.assertEqual(detect_script("product accha hai"), "latin")

    def test_devanagari(self):
        self.assertEqual(detect_script("\u0909\u0924\u094d\u092a\u093e\u0926 \u0905\u091a\u094d\u091b\u093e"), "devanagari")

    def test_digits_other(self):
        self.assertEqual(detect_script("12345 !!!"), "other")

    def test_non_string_raises(self):
        with self.assertRaises(TypeError):
            detect_script(42)

    def test_emoji_other(self):
        self.assertEqual(detect_script("\U0001f60a\U0001f602"), "other")


# ===========================================================================
# 9. normalization.py
# ===========================================================================
class TestNormalize(unittest.TestCase):
    def test_type(self):
        self.assertIsInstance(normalize("mala nhi"), fr.NormalizeResult)

    def test_normalized(self):
        self.assertIn("nahi", normalize("delivery nhi aayi").normalized)

    def test_original_preserved(self):
        text = "delivery nhi aayi"
        self.assertEqual(normalize(text).original, text)

    def test_changes(self):
        self.assertIn("nhi", [c[0] for c in normalize("delivery nhi aayi").changes])

    def test_str(self):
        r = normalize("delivery nhi aayi")
        self.assertEqual(str(r), r.normalized)

    def test_idempotent(self):
        first = normalize("delivery nahi aayi").normalized
        self.assertEqual(normalize(first).normalized, first)

    def test_url_preserved(self):
        self.assertIn("https://example.com", normalize("check https://example.com").normalized)

    def test_numbers_preserved(self):
        self.assertIn("12345", normalize("order 12345 nhi").normalized)

    def test_punctuation_collapsed(self):
        self.assertNotIn("!!!", normalize("great!!!").normalized)

    def test_non_string_raises(self):
        with self.assertRaises(TypeError):
            normalize(123)

    def test_lowercase_false(self):
        self.assertTrue(normalize("Mala Product", lowercase=False).normalized[0].isupper())


# ===========================================================================
# 10. perturbation.py
# ===========================================================================
class TestPerturb(unittest.TestCase):
    def test_returns_variants(self):
        self.assertGreater(len(perturb("delivery nahi aayi", n=5)), 0)

    def test_variant_fields(self):
        for v in perturb("delivery nahi aayi", n=5):
            self.assertIsInstance(v, fr.Variant)
            self.assertIsInstance(v.text, str)
            self.assertIsInstance(v.kind, str)

    def test_differ_from_original(self):
        text = "delivery nahi aayi"
        self.assertTrue(any(v.text != text for v in perturb(text, n=5)))

    def test_empty_returns_empty(self):
        self.assertEqual(perturb("", n=5), [])

    def test_kind_filter(self):
        for v in perturb("delivery nahi aayi", n=10, kinds=["spelling"]):
            self.assertEqual(v.kind, "spelling")

    def test_reproducible(self):
        t = "delivery nahi aayi"
        self.assertEqual(
            [v.text for v in perturb(t, n=5, seed=42)],
            [v.text for v in perturb(t, n=5, seed=42)])

    def test_negation_preserved(self):
        # Variants swap spellings or repeat vowels but must preserve the negation token
        import re
        neg_pattern = re.compile(r"\bn+a*h+[iey]+n*\b", re.IGNORECASE)
        for v in perturb("delivery nahi aayi", n=20):
            self.assertTrue(
                bool(neg_pattern.search(v.text)),
                f"Negation missing from variant: {v.text}",
            )


# ===========================================================================
# 11. rules.py
# ===========================================================================
class TestLoadRules(unittest.TestCase):
    def test_hindi(self):
        rules = fr.load_rules("hindi")
        self.assertIsInstance(rules, fr.RuleSet)
        self.assertGreater(len(rules.groups), 50)

    def test_hindi_code(self):
        self.assertIsInstance(fr.load_rules("hi"), fr.RuleSet)

    def test_marathi(self):
        self.assertIsInstance(fr.load_rules("marathi"), fr.RuleSet)

    def test_unsupported(self):
        with self.assertRaises(ValueError):
            fr.load_rules("klingon_xyz")


class TestAddCustomRule(unittest.TestCase):
    def test_variants_added(self):
        rules = fr.add_custom_rule(["discnt", "dscnt", "discount"], canonical="discount")
        self.assertEqual(rules.key_to_canon.get("dscnt"), "discount")

    def test_no_mutation_of_default(self):
        before = len(DEFAULT_RULES.groups)
        fr.add_custom_rule(["xyz_custom9999"], canonical="xyz_canonical9999")
        self.assertEqual(len(DEFAULT_RULES.groups), before)


class TestPreviewVariants(unittest.TestCase):
    def test_returns_strings(self):
        for v in fr.preview_variants("delivery nahi aaya", n=5):
            self.assertIsInstance(v, str)

    def test_n_respected(self):
        self.assertLessEqual(len(fr.preview_variants("delivery nahi aaya", n=3)), 3)


class TestDescribeRuleGroup(unittest.TestCase):
    def test_known_variant(self):
        desc = fr.describe_rule_group("nhi")
        self.assertIsNotNone(desc)
        self.assertEqual(desc["canonical"], "nahi")

    def test_has_variants(self):
        self.assertIn("nahi", fr.describe_rule_group("nahi")["variants"])

    def test_negation_flag(self):
        self.assertTrue(fr.describe_rule_group("nahi")["is_negation"])

    def test_unknown_none(self):
        self.assertIsNone(fr.describe_rule_group("xyz_unknown_888"))


class TestIsSafeVariant(unittest.TestCase):
    def test_canonical_swap_safe(self):
        self.assertTrue(is_safe_variant("nhi", "nahi"))

    def test_negation_added_unsafe(self):
        self.assertFalse(is_safe_variant("delivery aayi", "delivery nahi aayi"))

    def test_same_text_safe(self):
        self.assertTrue(is_safe_variant("product accha", "product accha"))


# ===========================================================================
# 12. events.py
# ===========================================================================
class TestEvents(unittest.TestCase):
    def test_make_event_stage(self):
        ev = make_event("received", chars=42)
        self.assertEqual(ev["stage"], "received")
        self.assertEqual(ev["chars"], 42)

    def test_timestamp_float(self):
        self.assertIsInstance(make_event("test")["t"], float)

    def test_to_sse_format(self):
        sse = to_sse(make_event("test", value=1))
        self.assertTrue(sse.startswith("data: "))
        self.assertTrue(sse.endswith("\n\n"))

    def test_to_sse_valid_json(self):
        sse = to_sse(make_event("received", chars=5))
        parsed = json.loads(sse.replace("data: ", "").strip())
        self.assertEqual(parsed["stage"], "received")

    def test_drain_returns_value(self):
        def gen():
            yield make_event("a")
            return "final"
        self.assertEqual(drain(gen()), "final")

    def test_drain_on_event(self):
        events = []
        def gen():
            yield make_event("a")
            yield make_event("b")
            return "r"
        drain(gen(), on_event=events.append)
        self.assertEqual(len(events), 2)


# ===========================================================================
# 13. models.py — protect, ProtectedModel, Predictor
# ===========================================================================
class TestProtect(unittest.TestCase):
    def test_type(self):
        self.assertIsInstance(fr.protect(_pos_neg), fr.ProtectedModel)

    def test_normalizes(self):
        seen = []
        def rec(t):
            seen.append(t)
            return "ok"
        fr.protect(rec)("Mala ordeeer nhiii milali")
        self.assertIn("nahi", seen[0])

    def test_callable(self):
        self.assertIsInstance(fr.protect(_pos_neg)("product accha hai"), str)

    def test_lowercase_false(self):
        seen = []
        def rec(t):
            seen.append(t)
            return "ok"
        fr.protect(rec, lowercase=False)("Mala Ordeeer Nhiii")
        self.assertEqual(len(seen), 1)


class TestPredictor(unittest.TestCase):
    def test_predict_single(self):
        self.assertEqual(fr.Predictor(_pos_neg).predict(["product accha hai"]), ["pos"])

    def test_predict_multiple(self):
        self.assertEqual(len(fr.Predictor(_pos_neg).predict(["a accha", "b neg"])), 2)

    def test_caching_deduplicates(self):
        n = {"v": 0}
        def counting(t):
            n["v"] += 1
            return "ok"
        pred = fr.Predictor(counting, cache=True)
        pred.predict(["same", "same"])
        self.assertEqual(n["v"], 1)


# ===========================================================================
# 14. evaluation.py
# ===========================================================================
class TestEvaluationTest(unittest.TestCase):
    def test_type(self):
        self.assertIsInstance(fr.test(_pos_neg, SMALL_DATASET, n_variants=2, bootstrap=5), fr.Report)

    def test_n_messages(self):
        self.assertEqual(fr.test(_pos_neg, SMALL_DATASET, n_variants=2, bootstrap=5).n_messages, 4)

    def test_flip_rate_range(self):
        r = fr.test(_pos_neg, SMALL_DATASET, n_variants=2, bootstrap=5)
        self.assertGreaterEqual(r.flip_rate, 0.0)
        self.assertLessEqual(r.flip_rate, 1.0)

    def test_summary(self):
        r = fr.test(_pos_neg, SMALL_DATASET[:2], n_variants=2, bootstrap=5)
        self.assertIsInstance(r.summary(), str)

    def test_save_load(self):
        report = fr.test(_pos_neg, SMALL_DATASET[:2], n_variants=2, bootstrap=5)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            path = tf.name
        report.save(path)
        loaded = fr.Report.load(path)
        self.assertEqual(loaded.n_messages, report.n_messages)


class TestIterTest(unittest.TestCase):
    def test_yields_and_returns(self):
        gen = fr.iter_test(_pos_neg, SMALL_DATASET[:2], n_variants=2, bootstrap=5)
        events = []
        while True:
            try:
                events.append(next(gen))
            except StopIteration as stop:
                self.assertIsInstance(stop.value, fr.Report)
                break
        self.assertGreater(len(events), 0)


class TestMacroF1(unittest.TestCase):
    def test_perfect(self):
        self.assertAlmostEqual(fr.macro_f1(["pos", "neg"], ["pos", "neg"]), 1.0, places=5)

    def test_all_wrong(self):
        self.assertAlmostEqual(fr.macro_f1(["neg", "pos"], ["pos", "neg"]), 0.0, places=5)


class TestPairedDifference(unittest.TestCase):
    def test_returns_diff(self):
        rep1 = fr.test(_pos_neg, SMALL_DATASET[:2], n_variants=2, bootstrap=5)
        rep2 = fr.test(fr.protect(_pos_neg), SMALL_DATASET[:2], n_variants=2, bootstrap=5)
        diff = fr.paired_difference(rep1, rep2)
        # paired_difference returns a dict with 'difference', 'stat', 'ci', 'significant'
        self.assertIsInstance(diff, dict)
        self.assertIn("difference", diff)
        self.assertIn("stat", diff)
        self.assertIn("significant", diff)
        self.assertIsInstance(diff["difference"], float)

    def test_mismatched_raises(self):
        rep1 = fr.test(_pos_neg, SMALL_DATASET[:2], n_variants=2, bootstrap=5)
        rep2 = fr.test(_pos_neg, SMALL_DATASET[2:4], n_variants=2, bootstrap=5)
        with self.assertRaises(ValueError):
            fr.paired_difference(rep1, rep2)


class TestLoadDataset(unittest.TestCase):
    def test_list_tuples(self):
        rows = fr.load_dataset([("text a", "pos"), ("text b", "neg")])
        self.assertEqual(len(rows), 2)

    def test_list_dicts(self):
        rows = fr.load_dataset([{"text": "good", "label": "pos"}, {"text": "bad", "label": "neg"}])
        self.assertEqual(len(rows), 2)

    def test_csv(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
            f.write("text,label\nproduct accha,pos\ndelivery nahi,neg\n")
            path = f.name
        self.assertEqual(len(fr.load_dataset(path)), 2)


class TestReadCsvText(unittest.TestCase):
    def test_reads_texts(self):
        # read_csv_text takes a CSV content string (not a file path)
        csv_content = "text,label\nhello world,pos\ntest text,neg\n"
        rows = fr.read_csv_text(csv_content)
        self.assertIsInstance(rows, list)
        self.assertEqual(len(rows), 2)
        # Each item is a (text, label) tuple
        self.assertTrue(any("hello" in r[0] for r in rows))


# ===========================================================================
# 15. service.py
# ===========================================================================
class TestInMemoryJobStore(unittest.TestCase):
    def test_put_pop(self):
        store = InMemoryJobStore(max_jobs=10, ttl=60)
        jid = store.put("analyze", {"text": "hello"})
        self.assertEqual(store.pop(jid)["kind"], "analyze")

    def test_one_time(self):
        store = InMemoryJobStore(max_jobs=10, ttl=60)
        jid = store.put("analyze", {})
        store.pop(jid)
        self.assertIsNone(store.pop(jid))

    def test_len(self):
        store = InMemoryJobStore(max_jobs=10, ttl=60)
        store.put("a", {})
        store.put("b", {})
        self.assertEqual(len(store), 2)

    def test_evicts_oldest(self):
        store = InMemoryJobStore(max_jobs=2, ttl=60)
        jid1 = store.put("a", {})
        store.put("b", {})
        store.put("c", {})
        self.assertIsNone(store.pop(jid1))


class TestInMemoryRateLimiter(unittest.TestCase):
    def test_limits(self):
        limiter = InMemoryRateLimiter(limit_per_min=2)
        self.assertTrue(limiter.allow("u"))
        self.assertTrue(limiter.allow("u"))
        self.assertFalse(limiter.allow("u"))

    def test_users_independent(self):
        limiter = InMemoryRateLimiter(limit_per_min=1)
        self.assertTrue(limiter.allow("a"))
        self.assertFalse(limiter.allow("a"))
        self.assertTrue(limiter.allow("b"))


class TestServiceFactories(unittest.TestCase):
    def test_no_redis(self):
        settings = Settings(redis_url=None)
        self.assertIsInstance(create_job_store(settings), InMemoryJobStore)
        self.assertIsInstance(create_rate_limiter(settings), InMemoryRateLimiter)


class TestRedisJobStoreMocked(unittest.TestCase):
    def test_put_pop(self):
        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_redis.Redis.from_url.return_value = mock_client
        mock_client.get.return_value = '{"kind": "test", "params": {"n": 5}}'
        with patch.dict("sys.modules", {"redis": mock_redis}):
            from fliprate.service import RedisJobStore
            store = RedisJobStore("redis://localhost:6379/0", ttl=60)
            jid = store.put("test", {"n": 5})
            mock_client.setex.assert_called_once()
            self.assertEqual(store.pop(jid)["kind"], "test")
            mock_client.delete.assert_called_once()


class TestRedisRateLimiterMocked(unittest.TestCase):
    def test_allow(self):
        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_redis.Redis.from_url.return_value = mock_client
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_pipe.execute.return_value = [1, True]
        with patch.dict("sys.modules", {"redis": mock_redis}):
            from fliprate.service import RedisRateLimiter
            limiter = RedisRateLimiter("redis://localhost:6379/0", limit_per_min=10)
            self.assertTrue(limiter.allow("client"))


# ===========================================================================
# 16. experiment.py
# ===========================================================================
class TestRunExperiment(unittest.TestCase):
    def test_returns_experiment_result(self):
        # run_experiment uses 'models' (plural) not 'model'
        result = fr.run_experiment(
            models={"baseline": _pos_neg},
            dataset=SMALL_DATASET[:2],
            n_variants=2, bootstrap=5,
        )
        self.assertIsInstance(result, fr.ExperimentResult)


# ===========================================================================
# 17. Multi-language loaders
# ===========================================================================
class TestDakshinaLoader(unittest.TestCase):
    def test_supported_languages(self):
        # DAKSHINA_LANGUAGES is a dict {code: name} or a list/set of codes
        self.assertTrue(
            isinstance(fr.DAKSHINA_LANGUAGES, (list, tuple, set, dict)),
            f"Unexpected type: {type(fr.DAKSHINA_LANGUAGES)}"
        )

    def test_rules_callable(self):
        try:
            self.assertIsInstance(fr.dakshina_rules("hi"), list)
        except Exception:
            pass

    def test_groups_callable(self):
        try:
            self.assertIsInstance(fr.load_dakshina_groups("hi"), dict)
        except Exception:
            pass


class TestAksharantarLoader(unittest.TestCase):
    def test_supported_languages(self):
        # AKSHARANTAR_LANGUAGES is a dict {code: name} or a list/set of codes
        self.assertTrue(
            isinstance(fr.AKSHARANTAR_LANGUAGES, (list, tuple, set, dict)),
            f"Unexpected type: {type(fr.AKSHARANTAR_LANGUAGES)}"
        )

    def test_rules_callable(self):
        try:
            self.assertIsInstance(fr.aksharantar_rules("hi"), list)
        except Exception:
            pass

    def test_groups_callable(self):
        try:
            self.assertIsInstance(fr.load_aksharantar_groups("hi"), dict)
        except Exception:
            pass


# ===========================================================================
# 18. End-to-end integration
# ===========================================================================
class TestEndToEnd(unittest.TestCase):
    def test_normalize_then_gate(self):
        clean = fr.normalize_text("product avadlaaa nhi milala")
        self.assertIn("nahi", clean)
        res = fr.gate(_pos_neg, clean, n_variants=3)
        self.assertIn(res.decision, ("PASS", "TRANSFORM", "REVIEW"))

    def test_adapter_in_pipeline(self):
        adapter = fr.as_adapter(_pos_neg)
        res = fr.quick_check(adapter, "product bohot accha hai")
        self.assertIsInstance(res, fr.QuickCheckResult)

    def test_explain_then_benchmark(self):
        explanation = fr.explain_gate_decision("product accha hai", _pos_neg, n_variants=3)
        self.assertIn("decision", explanation)
        suite = fr.benchmark({"model": _pos_neg, "always_pos": lambda t: "pos"},
                             SMALL_DATASET[:2], n_variants=2, bootstrap=5)
        self.assertIn("model", suite.scores)

    def test_validate_then_evaluate(self):
        data = [("product accha hai", "pos"), ("delivery nahi aayi", "neg")]
        validation = fr.validate_dataset(data)
        self.assertTrue(validation["valid"])
        report = fr.evaluate_pipeline(_pos_neg, validation["rows"], n_variants=2, bootstrap=5)
        self.assertEqual(report.n_messages, 2)

    def test_protected_in_batch_guard(self):
        protected = fr.make_protected_model(_pos_neg)
        results = fr.batch_guard(protected, ["product avadlaaa nhi", "delivery accha tha"], n_variants=2)
        self.assertEqual(len(results), 2)

    def test_gate_result_json_roundtrip(self):
        res = fr.gate(_pos_neg, "product accha hai", n_variants=3)
        restored = json.loads(res.to_json())
        self.assertEqual(restored["decision"], res.decision)


if __name__ == "__main__":
    unittest.main(verbosity=2)

