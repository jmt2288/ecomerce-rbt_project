"""
tests/test_unit_deep.py
=======================
Deep unit tests for every function in main.py.
No API, no Docker — pure unit tests with mocks.

Run:
    pytest tests/test_unit_deep.py -v
"""

import hashlib
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════
# TestGetFingerprint
# ══════════════════════════════════════════════
class TestGetFingerprint:

    def _req(self, ua="", lang="", enc=""):
        req = MagicMock()
        # get_fingerprint uses: .get("User-Agent",""), .get("Accept-Language",""), .get("Accept-Encoding","")
        req.headers = {
            "User-Agent":      ua,
            "Accept-Language": lang,
            "Accept-Encoding": enc,
        }
        return req

    def test_known_ua_produces_correct_md5(self):
        """MD5(UA|lang|enc) — known values produce known hash."""
        from main import get_fingerprint
        req = self._req("Mozilla/5.0", "en-US", "gzip")
        expected = hashlib.md5("Mozilla/5.0|en-US|gzip".encode()).hexdigest()
        assert get_fingerprint(req) == expected

    def test_missing_headers_fallback_to_empty_string(self):
        """Missing headers default to empty string in the hash."""
        from main import get_fingerprint
        req = MagicMock()
        req.headers = {}
        # Function does .get("User-Agent","") etc — all default to ""
        expected = hashlib.md5("||".encode()).hexdigest()
        assert get_fingerprint(req) == expected

    def test_returns_32_hex_chars(self):
        """Fingerprint is always exactly 32 hex characters (MD5)."""
        from main import get_fingerprint
        req = self._req("TestBrowser/1.0", "es-ES", "gzip")
        fp = get_fingerprint(req)
        assert len(fp) == 32
        assert all(c in "0123456789abcdef" for c in fp)

    def test_different_uas_produce_different_fingerprints(self):
        from main import get_fingerprint
        req1 = self._req("Mozilla/5.0", "en", "gzip")
        req2 = self._req("headless-chrome", "en", "gzip")
        assert get_fingerprint(req1) != get_fingerprint(req2)

    def test_same_headers_produce_same_fingerprint(self):
        from main import get_fingerprint
        req1 = self._req("Mozilla/5.0", "en", "gzip")
        req2 = self._req("Mozilla/5.0", "en", "gzip")
        assert get_fingerprint(req1) == get_fingerprint(req2)


# ══════════════════════════════════════════════
# TestGetIdentifier
# ══════════════════════════════════════════════
class TestGetIdentifier:

    def test_format_is_fingerprint_colon_ip(self):
        """identifier = fingerprint:ip — colon-separated."""
        from main import get_identifier
        req = MagicMock()
        req.headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en", "Accept-Encoding": "gzip"}
        req.client = MagicMock(); req.client.host = "10.0.0.1"
        ident = get_identifier(req)
        assert ":" in ident
        fp, ip = ident.split(":", 1)
        assert len(fp) == 32
        assert ip == "10.0.0.1"

    def test_fingerprint_part_is_32_hex(self):
        from main import get_identifier
        req = MagicMock()
        req.headers = {"User-Agent": "TestBrowser"}
        req.client = MagicMock(); req.client.host = "1.2.3.4"
        ident = get_identifier(req)
        fp = ident.split(":")[0]
        assert len(fp) == 32

    def test_uses_client_host_not_forwarded_for(self):
        """get_identifier uses request.client.host — not X-Forwarded-For.
        X-Forwarded-For is NOT used by this function (spoofing protection).
        """
        from main import get_identifier
        req = MagicMock()
        req.headers = {
            "User-Agent":       "Mozilla/5.0",
            "X-Forwarded-For":  "203.0.113.5",   # this header is IGNORED
        }
        req.client = MagicMock(); req.client.host = "10.0.0.1"
        ident = get_identifier(req)
        # The IP in the identifier is client.host, not X-Forwarded-For
        assert "10.0.0.1" in ident
        assert "203.0.113.5" not in ident

    def test_different_ips_produce_different_identifiers(self):
        from main import get_identifier
        req1 = MagicMock()
        req1.headers = {"User-Agent": "Mozilla/5.0"}
        req1.client = MagicMock(); req1.client.host = "1.1.1.1"
        req2 = MagicMock()
        req2.headers = {"User-Agent": "Mozilla/5.0"}
        req2.client = MagicMock(); req2.client.host = "2.2.2.2"
        assert get_identifier(req1) != get_identifier(req2)


# ══════════════════════════════════════════════
# TestAnalyzeBehavioralAI
# ══════════════════════════════════════════════
class TestAnalyzeBehavioralAI:
    """
    analyze_behavioral_ai() returns a TUPLE: (score: float, events: list)
    Always use [0] to get the score.

    Rules:
      - Headless UA keywords: headless, selenium, puppeteer, playwright
      - Headless UA present:     +15 pts
      - Missing accept-language: +5  pts
      - Both signals:            +20 pts (not +20 — BHV-003 covers both)
    """

    def _req(self, ua, lang=None):
        req = MagicMock()
        # User-Agent: mixed case (matches .get("User-Agent",""))
        # accept-language: lowercase (matches "accept-language" not in headers)
        h = {"User-Agent": ua}
        if lang is not None:
            h["accept-language"] = lang
        req.headers = h
        return req

    def test_normal_browser_returns_zero(self):
        """Normal UA + Accept-Language -> 0.0 pts."""
        from main import analyze_behavioral_ai
        req = self._req("Mozilla/5.0 Chrome/99", lang="en-US")
        assert analyze_behavioral_ai(req)[0] == 0.0

    def test_headless_ua_returns_15(self):
        """Headless UA + Accept-Language present -> exactly +15 pts."""
        from main import analyze_behavioral_ai
        for ua in ["headless-chrome", "selenium-webdriver/4.0",
                   "puppeteer/21", "playwright/1.40"]:
            req = self._req(ua, lang="en")
            score = analyze_behavioral_ai(req)[0]
            assert score == 15.0, f"Failed for UA: {ua}, got {score}"

    def test_no_accept_language_adds_5(self):
        """Normal UA + NO Accept-Language -> exactly +5 pts."""
        from main import analyze_behavioral_ai
        req = self._req("Mozilla/5.0")  # no lang
        assert analyze_behavioral_ai(req)[0] == 5.0

    def test_headless_plus_no_language_returns_20(self):
        """Headless UA + NO Accept-Language -> exactly +20 pts."""
        from main import analyze_behavioral_ai
        req = self._req("headless-chrome")  # no lang
        assert analyze_behavioral_ai(req)[0] == 20.0

    def test_case_insensitive_ua_detection(self):
        """Headless detection is case-insensitive (.lower() applied)."""
        from main import analyze_behavioral_ai
        for ua in ["HEADLESS-CHROME", "Selenium/4.0", "PUPPETEER", "Playwright/1.40"]:
            req = self._req(ua, lang="en")
            score = analyze_behavioral_ai(req)[0]
            assert score >= 15.0, f"Did not detect uppercase UA: {ua}"

    def test_python_requests_ua_is_flagged(self):
        """python-requests UA is NOT in the headless keywords list.
        It gets +5 only for missing Accept-Language (no lang in default session).
        """
        from main import analyze_behavioral_ai
        # python-requests does NOT contain headless/selenium/puppeteer/playwright
        # so it only scores if Accept-Language is missing
        req = self._req("python-requests/2.31")  # no lang
        score = analyze_behavioral_ai(req)[0]
        # Gets +5 for missing Accept-Language only
        assert score == 5.0, f"python-requests scored {score}, expected 5.0"

    def test_legitimate_browser_with_all_headers_is_zero(self):
        """Full legitimate browser headers -> score 0."""
        from main import analyze_behavioral_ai
        req = MagicMock()
        req.headers = {
            "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
            "accept-language": "es-ES,es;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
        }
        assert analyze_behavioral_ai(req)[0] == 0.0

    def test_returns_tuple_with_two_elements(self):
        """analyze_behavioral_ai returns (score, events) tuple."""
        from main import analyze_behavioral_ai
        req = self._req("Mozilla/5.0", lang="en")
        result = analyze_behavioral_ai(req)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], list)


# ══════════════════════════════════════════════
# TestExtractFeatures
# ══════════════════════════════════════════════
class TestExtractFeatures:
    """
    extract_features() returns numpy array shape (1,6):
      [0] is_headless_ua   — 1 if headless keyword in UA
      [1] has_lang         — 1 if "accept-language" in headers (lowercase key)
      [2] rate             — request count from Redis zcard
      [3] risk_score       — float from Redis
      [4] failed_logins    — float from Redis
      [5] has_legit        — 1 if X-Legitimate-User == "true"
    """

    def _mock_redis(self, risk=0.0, fails=0.0, rate=0):
        r = MagicMock()
        r.get = MagicMock(side_effect=lambda k: (
            str(risk)  if k.startswith("risk:")  else
            str(fails) if k.startswith("fails:") else None
        ))
        r.zcard = MagicMock(return_value=rate)
        return r

    def _req(self, ua="Mozilla/5.0", lang="en", legit=None):
        req = MagicMock()
        h = {
            "User-Agent":      ua,       # mixed case for .get("User-Agent","")
            "accept-language": lang,     # lowercase for "accept-language" not in
        }
        if legit:
            h["X-Legitimate-User"] = legit
        req.headers = h
        req.client = MagicMock(); req.client.host = "1.2.3.4"
        req.url = MagicMock(); req.url.path = "/api/data"
        return req

    def test_shape_is_1x6(self):
        from main import extract_features, get_identifier
        req = self._req()
        r_mock = self._mock_redis()
        with patch("main.r", r_mock):
            f = extract_features(req, get_identifier(req))
        assert f.shape == (1, 6)

    def test_feature_0_normal_ua_is_0(self):
        from main import extract_features, get_identifier
        req = self._req(ua="Mozilla/5.0")
        with patch("main.r", self._mock_redis()):
            f = extract_features(req, get_identifier(req))
        assert f[0][0] == 0

    def test_feature_0_headless_ua_is_1(self):
        from main import extract_features, get_identifier
        req = self._req(ua="headless-chrome")
        with patch("main.r", self._mock_redis()):
            f = extract_features(req, get_identifier(req))
        assert f[0][0] == 1

    def test_feature_1_with_lang_header_is_1(self):
        """Feature 1 = has_lang. Key must be lowercase 'accept-language'."""
        from main import extract_features, get_identifier
        req = self._req(lang="en-US")   # _req sets "accept-language" lowercase
        with patch("main.r", self._mock_redis()):
            f = extract_features(req, get_identifier(req))
        assert f[0][1] == 1

    def test_feature_1_without_lang_is_0(self):
        from main import extract_features, get_identifier
        req = MagicMock()
        req.headers = {"User-Agent": "Mozilla/5.0"}  # no accept-language key
        req.client = MagicMock(); req.client.host = "1.2.3.4"
        req.url = MagicMock(); req.url.path = "/api/data"
        with patch("main.r", self._mock_redis()):
            f = extract_features(req, get_identifier(req))
        assert f[0][1] == 0

    def test_feature_2_rate_from_redis(self):
        from main import extract_features, get_identifier
        req = self._req()
        with patch("main.r", self._mock_redis(rate=42)):
            f = extract_features(req, get_identifier(req))
        assert f[0][2] == 42

    def test_feature_3_risk_score_from_redis(self):
        from main import extract_features, get_identifier
        req = self._req()
        with patch("main.r", self._mock_redis(risk=25.0)):
            f = extract_features(req, get_identifier(req))
        assert f[0][3] == 25.0

    def test_feature_4_failed_logins_from_redis(self):
        from main import extract_features, get_identifier
        req = self._req()
        with patch("main.r", self._mock_redis(fails=3.0)):
            f = extract_features(req, get_identifier(req))
        assert f[0][4] == 3.0

    def test_feature_5_legit_header_true_is_1(self):
        from main import extract_features, get_identifier
        req = self._req(legit="true")
        with patch("main.r", self._mock_redis()):
            f = extract_features(req, get_identifier(req))
        assert f[0][5] == 1

    def test_feature_5_no_legit_header_is_0(self):
        from main import extract_features, get_identifier
        req = self._req()
        with patch("main.r", self._mock_redis()):
            f = extract_features(req, get_identifier(req))
        assert f[0][5] == 0

    def test_redis_none_returns_zero_for_numeric_features(self):
        from main import extract_features, get_identifier
        req = self._req()
        r_mock = MagicMock()
        r_mock.get   = MagicMock(return_value=None)
        r_mock.zcard = MagicMock(return_value=0)
        with patch("main.r", r_mock):
            f = extract_features(req, get_identifier(req))
        assert f[0][2] == 0    # rate
        assert f[0][3] == 0.0  # risk_score
        assert f[0][4] == 0.0  # failed_logins


# ══════════════════════════════════════════════
# TestUpdateRiskScore
# ══════════════════════════════════════════════
class TestUpdateRiskScore:

    def _mock_redis(self, current=0.0):
        r = MagicMock()
        r.get = MagicMock(return_value=str(current))
        r.set = MagicMock()
        return r

    def test_adds_points_to_existing_score(self):
        from main import update_risk_score
        r_mock = self._mock_redis(current=20.0)
        with patch("main.r", r_mock), patch("main.RISK_SCORE_METRIC"):
            result = update_risk_score("fp:x", 15.0)
        assert result == 35.0

    def test_starts_from_zero_for_new_user(self):
        from main import update_risk_score
        r_mock = self._mock_redis(current=0.0)
        with patch("main.r", r_mock), patch("main.RISK_SCORE_METRIC"):
            result = update_risk_score("new:user", 15.0)
        assert result == 15.0

    def test_stores_in_redis_with_ttl_10000(self):
        from main import update_risk_score
        r_mock = self._mock_redis(current=0.0)
        with patch("main.r", r_mock), patch("main.RISK_SCORE_METRIC"):
            update_risk_score("fp:x", 25.0)
        r_mock.set.assert_called_once_with("risk:fp:x", 25.0, ex=10000)

    def test_updates_prometheus_gauge(self):
        from main import update_risk_score
        r_mock = self._mock_redis(current=0.0)
        mock_gauge = MagicMock()
        with patch("main.r", r_mock),              patch("main.RISK_SCORE_METRIC") as mock_metric:
            mock_metric.labels.return_value = mock_gauge
            update_risk_score("fp:x", 25.0)
        mock_gauge.set.assert_called_once_with(25.0)
