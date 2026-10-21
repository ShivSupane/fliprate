import tempfile
import threading
import unittest

import fliprate as fr
from fliprate.demo import keyword_classifier, sample_dataset
from fliprate.service import JobStore, Metrics, RateLimiter, Settings, key_ok, run_job_events


class GateRegressionTests(unittest.TestCase):
    def test_screenshot_case_is_transform_not_pass(self):
        """One variant ("pymnt") flips the model. Normalization fixes it, so the gate must say TRANSFORM."""
        def model(t):
            return "payment_issue" if "payment" in t.lower().replace(".", "").split() else "human_support"
        r = fr.gate(model, "i need the payment back. Aapki servises good nahi he ")
        self.assertEqual(r.decision, "TRANSFORM")
        self.assertLess(r.raw_stability, 1.0)
        self.assertEqual(r.normalized_stability, 1.0)
        self.assertTrue(r.flipped_variants)

    def test_mostly_stable_passes_with_warning_when_normalization_cannot_help(self):
        def model(t):  # flips when the text ends with a dot; normalization cannot remove that difference
            return "b" if t.rstrip().endswith(".") else "a"
        r = fr.gate(model, "mala order late", n_variants=12, threshold=0.5)
        self.assertEqual(r.decision, "PASS", r.to_dict())
        self.assertLess(r.raw_stability, 1.0)
        self.assertTrue(any("did not improve" in w for w in r.warnings))

    def test_review_when_unstable_and_unfixable(self):
        def model(t):  # answer depends on punctuation, which normalization keeps
            return "b" if t.rstrip().endswith((".", "!")) else "a"
        r = fr.gate(model, "mala order late", n_variants=12, threshold=0.99)
        self.assertEqual(r.decision, "REVIEW")


class SettingsServiceTests(unittest.TestCase):
    def test_settings_from_env(self):
        s = Settings.from_env({"FLIPRATE_API_KEY": "k", "FLIPRATE_CORS": "https://a.com, https://b.com", "FLIPRATE_RATE_LIMIT": "5"})
        self.assertEqual((s.api_key, s.cors_origins, s.rate_limit_per_min), ("k", ["https://a.com", "https://b.com"], 5))

    def test_key_ok(self):
        self.assertTrue(key_ok(None, None))
        self.assertTrue(key_ok("secret", "secret"))
        self.assertFalse(key_ok("nope", "secret"))
        self.assertFalse(key_ok(None, "secret"))

    def test_job_store_is_one_time_and_expires(self):
        t = [0.0]
        store = JobStore(max_jobs=2, ttl=10, clock=lambda: t[0])
        a = store.put("analyze", {})
        self.assertIsNotNone(store.pop(a))
        self.assertIsNone(store.pop(a))
        b = store.put("analyze", {})
        t[0] = 11
        self.assertIsNone(store.pop(b))

    def test_job_store_evicts_oldest(self):
        store = JobStore(max_jobs=2)
        ids = [store.put("x", {}) for _ in range(3)]
        self.assertIsNone(store.pop(ids[0]))
        self.assertIsNotNone(store.pop(ids[2]))

    def test_rate_limiter_window(self):
        t = [0.0]
        rl = RateLimiter(2, clock=lambda: t[0])
        self.assertTrue(rl.allow("ip")); self.assertTrue(rl.allow("ip")); self.assertFalse(rl.allow("ip"))
        self.assertTrue(rl.allow("other"))
        t[0] = 61
        self.assertTrue(rl.allow("ip"))
        self.assertTrue(RateLimiter(0).allow("x"))

    def test_metrics_render(self):
        m = Metrics(); m.inc("jobs_started_total"); m.inc("jobs_started_total")
        self.assertIn("fliprate_jobs_started_total 2.0", m.render())

    def test_run_job_events_analyze_and_never_leaks_exception_text(self):
        sem, met = threading.BoundedSemaphore(1), Metrics()
        job = {"kind": "analyze", "params": {"text": "mala order nahi milala", "n_variants": 4}}
        stages = [e["stage"] for e in run_job_events(job, keyword_classifier, sample_dataset, sem, met)]
        self.assertEqual(stages[-1], "end")
        self.assertIn("gate", stages)

        def boom(_):
            raise RuntimeError("SECRET-USER-TEXT")
        evs = list(run_job_events(job, boom, sample_dataset, sem, met))
        err = next(e for e in evs if e["stage"] == "error")
        self.assertNotIn("SECRET", err["message"])
        self.assertEqual(met._c["jobs_failed_total"], 1)

    def test_busy_server_rejects(self):
        sem = threading.BoundedSemaphore(1)
        sem.acquire()
        evs = list(run_job_events({"kind": "test", "params": {"n_variants": 2}}, keyword_classifier, sample_dataset, sem, Metrics()))
        self.assertEqual(evs[0]["stage"], "error")

    def test_test_job_streams_progress(self):
        sem = threading.BoundedSemaphore(1)
        job = {"kind": "test", "params": {"n_variants": 2, "rows": [{"text": "mala order nahi", "label": "order_issue"}]}}
        stages = [e["stage"] for e in run_job_events(job, keyword_classifier, sample_dataset, sem, Metrics())]
        self.assertEqual(stages, ["started", "progress", "done", "end"])


class ReportPersistenceTests(unittest.TestCase):
    def test_save_load_roundtrip_with_metadata(self):
        r = fr.test(keyword_classifier, sample_dataset(), n_variants=2, bootstrap=20)
        with tempfile.TemporaryDirectory() as d:
            r.save(d + "/r.json")
            r2 = fr.Report.load(d + "/r.json")
        self.assertEqual(r2.flip_rate, r.flip_rate)
        self.assertEqual(r2.metadata["rules_fingerprint"], fr.DEFAULT_RULES.fingerprint())
        self.assertEqual(len(r2.metadata["dataset_sha256_16"]), 16)
        self.assertIn("| Flip rate |", r.to_markdown())


class PairedTests(unittest.TestCase):
    def test_paired_difference_detects_protection_effect(self):
        def brittle(t):
            return "neg" if "nahi" in t.split() else "pos"
        data = [("mala avadla nahi", "neg"), ("mala avadla", "pos")] * 10
        raw = fr.test(brittle, data, n_variants=6, bootstrap=20)
        prot = fr.test(fr.protect(brittle), data, n_variants=6, bootstrap=20)
        d = fr.paired_difference(raw, prot, "flip_rate", boots=200)
        self.assertLess(d["difference"], 0)
        self.assertTrue(d["significant"])
        self.assertEqual(fr.paired_difference(raw, raw, "flip_rate", boots=50)["difference"], 0.0)

    def test_refuses_mismatched_runs(self):
        a = fr.test(lambda t: "x", [("a b c", "x"), ("d e f", "x")], n_variants=2, bootstrap=10)
        b = fr.test(lambda t: "x", [("a b c", "x"), ("g h i", "x")], n_variants=2, bootstrap=10)
        with self.assertRaises(ValueError):
            fr.paired_difference(a, b)

    def test_save_load_keeps_message_stats(self):
        r = fr.test(lambda t: "x", [("a b c", "x")], n_variants=2, bootstrap=10)
        with tempfile.TemporaryDirectory() as d:
            r.save(d + "/r.json")
            self.assertEqual(fr.Report.load(d + "/r.json").message_stats, r.message_stats)


class ExperimentTests(unittest.TestCase):
    def test_run_experiment_raw_vs_protected(self):
        def brittle(t):
            return "neg" if "nahi" in t.split() else "pos"
        data = [("mala avadla nahi", "neg"), ("mala avadla", "pos")] * 6
        res = fr.run_experiment({"brittle": brittle}, data, n_variants=6, bootstrap=20)
        self.assertEqual(set(res.reports["brittle"]), {"raw", "protected"})
        self.assertGreaterEqual(res.reports["brittle"]["raw"].flip_rate, res.reports["brittle"]["protected"].flip_rate)
        md = res.markdown()
        self.assertIn("Effect of protection", md)
        with tempfile.TemporaryDirectory() as d:
            out = res.save(d + "/run")
            self.assertTrue((out / "summary.md").exists())
            self.assertTrue((out / "brittle__raw.json").exists())

    def test_sklearn_baselines_run_through_fliprate(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")
        from fliprate.baselines import train_tfidf
        rows = sample_dataset()
        models = {"word": train_tfidf(rows, "word"), "char": train_tfidf(rows, "char_wb")}
        res = fr.run_experiment(models, rows, n_variants=3, bootstrap=20, with_protection=False)
        for name in models:
            self.assertEqual(res.reports[name]["raw"].n_messages, len(rows))


if __name__ == "__main__":
    unittest.main()
