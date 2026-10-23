import unittest

import fliprate as fr
from fliprate.adapters import HFTextClassifier, LLMClassifier, twitter_preprocess, with_retries
from fliprate.models import Predictor, batch_predict


class CountingModel:
    def __init__(self):
        self.single_calls = 0
        self.batch_calls = []

    def __call__(self, text):
        self.single_calls += 1
        return "x"

    def predict_batch(self, texts):
        self.batch_calls.append(list(texts))
        return ["x" for _ in texts]


class PredictorTests(unittest.TestCase):
    def test_dedupes_and_caches(self):
        m = CountingModel()
        p = Predictor(m)
        out = p.predict(["a", "b", "a"])
        self.assertEqual(out, ["x", "x", "x"])
        self.assertEqual(m.batch_calls, [["a", "b"]])
        p.predict(["a", "c"])
        self.assertEqual(m.batch_calls[-1], ["c"])
        self.assertEqual(p.cache_hits, 1)

    def test_uses_predict_batch_when_available(self):
        m = CountingModel()
        batch_predict(m, ["a", "b"])
        self.assertEqual(m.single_calls, 0)

    def test_batch_size_respected(self):
        m = CountingModel()
        Predictor(m, batch_size=2).predict(["a", "b", "c", "d", "e"])
        self.assertEqual([len(b) for b in m.batch_calls], [2, 2, 1])

    def test_wrong_length_from_model_is_an_error(self):
        class Bad:
            def __call__(self, t): return 1
            def predict_batch(self, texts): return [1]
        with self.assertRaises(ValueError):
            batch_predict(Bad(), ["a", "b"])

    def test_cache_disabled(self):
        m = CountingModel()
        p = Predictor(m, cache=False)
        p.predict(["a"]); p.predict(["a"])
        self.assertEqual(len(m.batch_calls), 2)


class ProtectTests(unittest.TestCase):
    def test_protect_normalizes_before_model(self):
        seen = []
        def model(t):
            seen.append(t)
            return "ok"
        pm = fr.protect(model)
        pm("Mala ordeeer nhiii!!!")
        self.assertEqual(seen, ["mala order nahi!"])
        self.assertTrue(pm.name.startswith("protected("))

    def test_protection_reduces_flip_rate_for_brittle_model(self):
        def brittle(t):  # only understands exact canonical spelling
            return "neg" if "nahi" in t.split() else "pos"
        data = [("mala avadla nahi", "neg"), ("mala avadla", "pos")] * 5
        raw = fr.test(brittle, data, n_variants=8, kinds=["spelling"], bootstrap=20)
        prot = fr.test(fr.protect(brittle), data, n_variants=8, kinds=["spelling"], bootstrap=20)
        self.assertGreater(raw.flip_rate, 0)
        self.assertEqual(prot.flip_rate, 0.0)


class HFAdapterTests(unittest.TestCase):
    def fake_pipe(self):
        calls = []
        def pipe(texts, batch_size, truncation, max_length):
            calls.append((len(texts), batch_size, truncation, max_length))
            return [{"label": "Negative" if "nahi" in t else "Positive", "score": .9} for t in texts]
        return pipe, calls

    def test_batches_and_lowercases_labels(self):
        pipe, calls = self.fake_pipe()
        clf = HFTextClassifier("fake/model", lowercase_labels=True, batch_size=16, pipeline_factory=lambda: pipe)
        self.assertEqual(clf.predict_batch(["acha hai", "ye nahi chahiye"]), ["positive", "negative"])
        self.assertEqual(calls, [(2, 16, True, 256)])
        self.assertEqual(clf("acha hai"), "positive")

    def test_label_map_and_preprocess(self):
        pipe, _ = self.fake_pipe()
        clf = HFTextClassifier("fake", label_map={"Positive": "pos"}, preprocess=twitter_preprocess,
                               pipeline_factory=lambda: pipe)
        self.assertEqual(clf("@bob acha"), "pos")
        self.assertEqual(twitter_preprocess("hi @bob see https://x.co"), "hi @user see http")

    def test_works_inside_fliprate_test(self):
        pipe, _ = self.fake_pipe()
        clf = HFTextClassifier("fake", lowercase_labels=True, pipeline_factory=lambda: pipe)
        r = fr.test(clf, [("mala avadla nahi", "negative"), ("bahut accha", "positive")], n_variants=4, bootstrap=20)
        self.assertEqual(r.n_messages, 2)
        self.assertEqual(r.metadata["model"], "fake")


class LLMAdapterTests(unittest.TestCase):
    LABELS = ["order_issue", "payment_issue"]

    def test_parsing(self):
        c = LLMClassifier(lambda p: "", self.LABELS)
        self.assertEqual(c.parse("payment_issue"), "payment_issue")
        self.assertEqual(c.parse("The label is: Order_Issue."), "order_issue")
        self.assertEqual(c.parse("order_issue or payment_issue"), "__invalid__")
        self.assertEqual(c.parse("no idea"), "__invalid__")
        self.assertEqual(c.n_invalid, 2)

    def test_prompt_contains_labels_and_message(self):
        seen = []
        c = LLMClassifier(lambda p: seen.append(p) or "order_issue", self.LABELS)
        c("mera order cancel")
        self.assertIn("order_issue, payment_issue", seen[0])
        self.assertIn("mera order cancel", seen[0])


class RetryTests(unittest.TestCase):
    def test_retries_then_succeeds(self):
        state = {"n": 0}
        def flaky():
            state["n"] += 1
            if state["n"] < 3:
                raise RuntimeError("boom")
            return "ok"
        self.assertEqual(with_retries(flaky, retries=3, backoff=0)(), "ok")
        self.assertEqual(state["n"], 3)

    def test_gives_up(self):
        def always():
            raise RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            with_retries(always, retries=2, backoff=0)()


class DeploymentModelTests(unittest.TestCase):
    def test_hf_models_import_is_cheap_and_has_batch_api(self):
        from fliprate import hf_models
        self.assertTrue(hasattr(hf_models.xlmt_sentiment, "predict_batch"))
        self.assertIsNone(hf_models.xlmt_sentiment._pipe)  # nothing downloaded at import time
        from fliprate.demo import load_model
        self.assertIs(load_model("fliprate.hf_models:xlmt_sentiment"), hf_models.xlmt_sentiment)


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
