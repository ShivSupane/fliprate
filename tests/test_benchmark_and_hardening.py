import tempfile
import unittest
from unittest.mock import MagicMock, patch

import fliprate as fr
from fliprate.service import (
    InMemoryJobStore,
    InMemoryRateLimiter,
    RedisJobStore,
    RedisRateLimiter,
    Settings,
    create_job_store,
    create_rate_limiter,
)


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.dataset = [
            ("bahut accha product", "pos"),
            ("delivery nahi aayi", "neg"),
            ("service acchi lagi", "pos"),
            ("kharab laga", "neg"),
        ]
        self.model_a = lambda t: "pos" if "accha" in t or "acchi" in t else "neg"
        self.model_b = lambda t: "pos" if "bahut" in t else "neg"

    def test_benchmark_multi_model_and_baseline(self):
        suite = fr.benchmark(
            {"model_a": self.model_a, "model_b": self.model_b},
            self.dataset,
            baseline="model_a",
            n_variants=2,
            bootstrap=10,
        )
        self.assertIn("model_a", suite.scores)
        self.assertIn("model_b", suite.scores)
        self.assertTrue(suite.scores["model_a"].is_baseline)
        self.assertFalse(suite.scores["model_b"].is_baseline)
        self.assertIsNotNone(suite.scores["model_b"].score_vs_baseline)

    def test_benchmark_rendering_and_save(self):
        suite = fr.compare_models(self.model_a, self.model_b, self.dataset, n_variants=2, bootstrap=10)
        summary = suite.summary()
        self.assertIn("Benchmark Summary", summary)

        md = suite.to_markdown()
        self.assertIn("| Model |", md)

        html = suite.to_html()
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("FlipRate Robustness Dashboard", html)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tf:
            path_html = tf.name
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf2:
            path_json = tf2.name

        suite.save(path_html)
        suite.save(path_json)
        self.assertTrue(len(open(path_html, encoding="utf-8").read()) > 0)
        self.assertTrue(len(open(path_json, encoding="utf-8").read()) > 0)


class StorageAndHardeningTests(unittest.TestCase):
    def test_in_memory_job_store(self):
        store = InMemoryJobStore(max_jobs=10, ttl=60)
        job_id = store.put("analyze", {"text": "hello"})
        self.assertEqual(len(store), 1)
        job = store.pop(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["kind"], "analyze")
        # second pop is None (one-time token)
        self.assertIsNone(store.pop(job_id))

    def test_in_memory_rate_limiter(self):
        limiter = InMemoryRateLimiter(limit_per_min=2)
        self.assertTrue(limiter.allow("user1"))
        self.assertTrue(limiter.allow("user1"))
        self.assertFalse(limiter.allow("user1"))
        # different user is allowed
        self.assertTrue(limiter.allow("user2"))

    def test_factories_fallback_without_redis(self):
        settings = Settings(redis_url=None)
        store = create_job_store(settings)
        limiter = create_rate_limiter(settings)
        self.assertIsInstance(store, InMemoryJobStore)
        self.assertIsInstance(limiter, InMemoryRateLimiter)

    def test_redis_job_store(self):
        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_redis.Redis.from_url.return_value = mock_client
        mock_client.get.return_value = '{"kind": "test", "params": {"n": 5}}'

        with patch.dict("sys.modules", {"redis": mock_redis}):
            store = RedisJobStore("redis://localhost:6379/0", ttl=60)
            job_id = store.put("test", {"n": 5})
            mock_client.setex.assert_called_once()

            job = store.pop(job_id)
            self.assertEqual(job["kind"], "test")
            mock_client.delete.assert_called_once()

    def test_redis_rate_limiter(self):
        mock_redis = MagicMock()
        mock_client = MagicMock()
        mock_redis.Redis.from_url.return_value = mock_client
        mock_pipe = MagicMock()
        mock_client.pipeline.return_value = mock_pipe
        mock_pipe.execute.return_value = [1, True]  # 1st request

        with patch.dict("sys.modules", {"redis": mock_redis}):
            limiter = RedisRateLimiter("redis://localhost:6379/0", limit_per_min=10)
            allowed = limiter.allow("client_abc")
            self.assertTrue(allowed)


class ApiCorrelationIdTests(unittest.TestCase):
    def test_api_correlation_id_middleware(self):
        try:
            from fastapi.testclient import TestClient
            from fliprate.api import app
            client = TestClient(app)
            response = client.get("/health", headers={"X-Request-ID": "custom-trace-id-123"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers.get("X-Request-ID"), "custom-trace-id-123")
        except ImportError:
            pass  # optional api extra


if __name__ == "__main__":
    unittest.main()
