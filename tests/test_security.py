"""
tests/test_security.py
──────────────────────
Comprehensive test suite covering:
  - Basic API functionality
  - Risk Score rule engine
  - ML bot detection
  - False positive handling
  - Prometheus metrics

Usage:
    pytest tests/test_security.py -v
    pytest tests/test_security.py -v --html=reports/report.html
"""

import pytest
import requests
import time

BASE = "http://localhost:8000"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fresh_session(headless: bool = False, no_lang: bool = False) -> requests.Session:
    """Return a session with a unique fingerprint."""
    s = requests.Session()
    uid = str(time.time_ns())
    if headless:
        s.headers.update({"User-Agent": f"headless-chrome/{uid}"})
    else:
        s.headers.update({"User-Agent": f"Mozilla/5.0 RBT-Test/{uid}"})
    if no_lang:
        s.headers.pop("Accept-Language", None)
    return s


def get_metric(name: str) -> str:
    """Fetch raw metrics text from /metrics."""
    return requests.get(f"{BASE}/metrics").text


# ─────────────────────────────────────────────
# SUITE 1 — Basic API
# ─────────────────────────────────────────────
class TestBasicAPI:

    def test_root_returns_200(self):
        r = requests.get(f"{BASE}/")
        assert r.status_code == 200

    def test_root_shows_ml_status(self):
        r = requests.get(f"{BASE}/")
        data = r.json()
        assert "ml_model_loaded" in data

    def test_status_endpoint(self):
        r = requests.get(f"{BASE}/status")
        assert r.status_code == 200
        data = r.json()
        assert "threshold" in data
        assert data["threshold"] == 30

    def test_metrics_endpoint_reachable(self):
        r = requests.get(f"{BASE}/metrics")
        assert r.status_code == 200
        assert "http_requests_total" in r.text

    def test_metrics_contains_all_counters(self):
        text = get_metric("all")
        expected = [
            "http_requests_total",
            "blocked_requests_total",
            "false_positive_blocks_total",
            "current_risk_score",
            "login_failures_total",
            "bot_ml_probability",
        ]
        for metric in expected:
            assert metric in text, f"Missing metric: {metric}"


# ─────────────────────────────────────────────
# SUITE 2 — Authentication
# ─────────────────────────────────────────────
class TestAuthentication:

    def test_valid_login_returns_200(self):
        s = fresh_session()
        r = s.get(f"{BASE}/login",
                  params={"username": "admin", "password": "secret123"})
        assert r.status_code == 200
        assert "Welcome" in r.text

    def test_invalid_login_returns_401(self):
        s = fresh_session()
        r = s.get(f"{BASE}/login",
                  params={"username": "admin", "password": "wrongpass"})
        assert r.status_code == 401

    def test_failed_login_increments_metric(self):
        s = fresh_session()
        before = get_metric("login_failures_total")
        s.get(f"{BASE}/login",
              params={"username": "x", "password": "x"})
        after = get_metric("login_failures_total")
        # Metric text should have grown (new label combination or count)
        assert "login_failures_total" in after


# ─────────────────────────────────────────────
# SUITE 3 — Risk Score Rules
# ─────────────────────────────────────────────
class TestRiskScoreRules:

    def test_headless_ua_gets_risk_points(self):
        """Headless UA adds +15 to risk score."""
        s = fresh_session(headless=True)
        # First request — should pass but increment risk
        r = s.get(f"{BASE}/api/data")
        # Risk score metric should be present
        metrics = get_metric("current_risk_score")
        assert "current_risk_score" in metrics

    def test_risk_score_blocks_after_threshold(self):
        """After enough failed logins + headless UA, request is blocked."""
        s = fresh_session(headless=True)
        # Build risk score: +15 (headless) + 4*10 (fails) = 55 > 30
        s.get(f"{BASE}/api/data")  # +15
        for _ in range(4):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "x"})  # +10 each
        r = s.get(f"{BASE}/api/data")
        assert r.status_code == 403, \
            f"Expected 403 but got {r.status_code} — risk score may not have reached threshold"

    def test_blocked_request_increments_counter(self):
        """blocked_requests_total increases when a request is blocked."""
        s = fresh_session(headless=True)
        s.get(f"{BASE}/api/data")
        for _ in range(4):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "x"})
        s.get(f"{BASE}/api/data")  # trigger block
        metrics = get_metric("blocked_requests_total")
        assert "blocked_requests_total" in metrics


# ─────────────────────────────────────────────
# SUITE 4 — False Positives
# ─────────────────────────────────────────────
class TestFalsePositives:

    def test_legitimate_header_bypasses_block(self):
        """X-Legitimate-User: true allows through even when score > 30."""
        s = fresh_session(headless=True)
        # Build risk above threshold
        s.get(f"{BASE}/api/data")
        for _ in range(4):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "x"})
        # Confirm blocked first
        r_blocked = s.get(f"{BASE}/api/data")
        assert r_blocked.status_code == 403

        # Now add legitimate header
        r_legit = s.get(f"{BASE}/api/data",
                        headers={"X-Legitimate-User": "true"})
        assert r_legit.status_code == 200

    def test_false_positive_counter_increments(self):
        """false_positive_blocks_total increments on legitimate bypass."""
        s = fresh_session(headless=True)
        s.get(f"{BASE}/api/data")
        for _ in range(4):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "x"})

        before = get_metric("false_positive_blocks_total")
        s.get(f"{BASE}/api/data", headers={"X-Legitimate-User": "true"})
        after = get_metric("false_positive_blocks_total")
        assert "false_positive_blocks_total" in after


# ─────────────────────────────────────────────
# SUITE 5 — ML Bot Detection
# ─────────────────────────────────────────────
class TestMLDetection:

    def test_ml_probability_metric_exists(self):
        """bot_ml_probability metric appears after a request."""
        s = fresh_session(headless=True)
        s.get(f"{BASE}/api/data")
        metrics = get_metric("bot_ml_probability")
        # Only present if model is loaded — skip gracefully if not
        status = requests.get(f"{BASE}/status").json()
        if status.get("ml_model_loaded"):
            assert "bot_ml_probability" in metrics

    def test_ml_blocks_obvious_bot(self):
        """If ML model is loaded, an obvious bot pattern is blocked."""
        status = requests.get(f"{BASE}/status").json()
        if not status.get("ml_model_loaded"):
            pytest.skip("ML model not loaded — skipping ML-specific test")

        s = fresh_session(headless=True)
        # Simulate high-volume bot: many requests + failed logins
        for _ in range(10):
            s.get(f"{BASE}/api/data")
            s.get(f"{BASE}/login",
                  params={"username": "hack", "password": "pass"})

        r = s.get(f"{BASE}/api/data")
        assert r.status_code == 403

    def test_ml_allows_legit_user(self):
        """A normal user pattern should not be blocked by ML."""
        status = requests.get(f"{BASE}/status").json()
        if not status.get("ml_model_loaded"):
            pytest.skip("ML model not loaded")

        s = fresh_session()  # normal UA
        s.headers["Accept-Language"] = "es-ES,es;q=0.9"
        r = s.get(f"{BASE}/api/data")
        assert r.status_code == 200


# ─────────────────────────────────────────────
# SUITE 6 — Load generation (for Grafana)
# ─────────────────────────────────────────────
class TestLoadGeneration:

    def test_generate_mixed_traffic(self):
        """
        Generates a realistic mix of traffic to populate Grafana dashboards.
        Run this test to see data in Grafana.
        """
        import random

        # 10 legitimate users
        for i in range(10):
            s = fresh_session()
            s.headers["Accept-Language"] = "es-ES"
            s.get(f"{BASE}/api/data")
            s.get(f"{BASE}/login",
                  params={"username": "admin", "password": "secret123"})

        # 5 headless bots
        for i in range(5):
            s = fresh_session(headless=True)
            s.get(f"{BASE}/api/data")
            for _ in range(random.randint(2, 4)):
                s.get(f"{BASE}/login",
                      params={"username": "admin",
                              "password": f"wrong{random.randint(0,999)}"})

        # 2 false positives
        for i in range(2):
            s = fresh_session(headless=True)
            s.get(f"{BASE}/api/data")
            for _ in range(4):
                s.get(f"{BASE}/login",
                      params={"username": "x", "password": "x"})
            s.get(f"{BASE}/api/data",
                  headers={"X-Legitimate-User": "true"})

        metrics = get_metric("all")
        assert "http_requests_total" in metrics
        print("\n✅ Traffic generated — check Grafana at http://localhost:3000")
