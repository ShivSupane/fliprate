import unittest
import pytest

fastapi = pytest.importorskip("fastapi")


class ApiRouteTests(unittest.TestCase):
    def test_app_and_endpoints(self):
        from fliprate.api import app, health, ready, metrics, READY, console

        self.assertEqual(app.title, "FlipRate API")
        
        h = health()
        self.assertEqual(h["status"], "ok")
        self.assertIn("version", h)

        READY["model"] = True
        r = ready()
        self.assertEqual(r["status"], "ready")

        m = metrics()
        self.assertIsInstance(m, str)

        c = console()
        self.assertEqual(c.media_type, "text/html; charset=utf-8")
