import unittest
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from fliprate.api import app, SETTINGS, READY, LIMITER


class ApiHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        # Reset original settings
        self._orig_api_key = SETTINGS.api_key
        self._orig_ready = READY.get("model", False)

    def tearDown(self):
        SETTINGS.api_key = self._orig_api_key
        READY["model"] = self._orig_ready

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("version", data)

    def test_ready_endpoint(self):
        READY["model"] = False
        resp = self.client.get("/ready")
        self.assertEqual(resp.status_code, 503)

        READY["model"] = True
        resp = self.client.get("/ready")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"status": "ready"})

    def test_console_and_metrics(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])

        resp = self.client.get("/metrics")
        self.assertEqual(resp.status_code, 200)

    def test_auth_enforcement_401(self):
        SETTINGS.api_key = "test_secret_123"

        # Missing key
        resp = self.client.post("/jobs/analyze", json={"text": "kahi tari problem ahe", "n_variants": 2})
        self.assertEqual(resp.status_code, 401)

        # Wrong key
        resp = self.client.post(
            "/jobs/analyze",
            json={"text": "kahi tari problem ahe", "n_variants": 2},
            headers={"X-API-Key": "wrong_key"},
        )
        self.assertEqual(resp.status_code, 401)

        # Correct key
        resp = self.client.post(
            "/jobs/analyze",
            json={"text": "kahi tari problem ahe", "n_variants": 2},
            headers={"X-API-Key": "test_secret_123"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("job_id", resp.json())

    def test_rate_limit_429(self):
        SETTINGS.api_key = None
        orig_limit = LIMITER.limit
        try:
            LIMITER._hits.clear()
            LIMITER._last_seen.clear()
            LIMITER.limit = 1

            # First request passes
            resp1 = self.client.post(
                "/jobs/analyze",
                json={"text": "test rate limit", "n_variants": 2},
            )
            self.assertEqual(resp1.status_code, 200)

            # Second immediate request gets rate limited
            resp2 = self.client.post(
                "/jobs/analyze",
                json={"text": "test rate limit", "n_variants": 2},
            )
            self.assertEqual(resp2.status_code, 429)
        finally:
            LIMITER.limit = orig_limit

    def test_post_analyze_and_stream_flow(self):
        SETTINGS.api_key = None
        resp = self.client.post("/jobs/analyze", json={"text": "delivery nahi mila", "n_variants": 2})
        self.assertEqual(resp.status_code, 200)
        job_id = resp.json().get("job_id")
        self.assertIsNotNone(job_id)

        # Stream SSE events
        stream_resp = self.client.get(f"/jobs/{job_id}/stream")
        self.assertEqual(stream_resp.status_code, 200)
        self.assertIn("text/event-stream", stream_resp.headers["content-type"])
        self.assertIn("data: ", stream_resp.text)
        self.assertIn('"stage": "done"', stream_resp.text)

        # Stream once consumed cannot be re-streamed (one-time secret token)
        second_resp = self.client.get(f"/jobs/{job_id}/stream")
        self.assertEqual(second_resp.status_code, 404)
