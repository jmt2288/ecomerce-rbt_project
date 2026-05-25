"""
tests/test_risk_codes.py
========================
Tests for the risk code system — risk_codes.py module.
These tests run WITHOUT API (pure unit tests with no Docker needed).

Run:
    pytest tests/test_risk_codes.py -v
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk_codes import (
    RiskCode, RiskEvent, RiskCodeDef, Severity, Category,
    CATALOG, get_by_code, get_by_severity, get_by_category,
    event_ml_blocked, event_score_exceeded, event_login_failure,
    event_headless_ua, event_missing_lang, event_false_positive,
    event_redis_error, catalog_summary,
)


# ══════════════════════════════════════════════
# CATALOG INTEGRITY
# ══════════════════════════════════════════════
class TestCatalogIntegrity:

    def test_catalog_not_empty(self):
        assert len(CATALOG) > 0

    def test_all_codes_follow_naming_convention(self):
        """Every code must match RBT-XXX-NNN format."""
        import re
        pattern = re.compile(r"^RBT-[A-Z]{3}-\d{3}$")
        for code in CATALOG:
            assert pattern.match(code), f"Invalid code format: {code}"

    def test_all_codes_unique(self):
        codes = list(CATALOG.keys())
        assert len(codes) == len(set(codes))

    def test_all_severities_valid(self):
        valid = {s.value for s in Severity}
        for c in CATALOG.values():
            assert c.severity.value in valid

    def test_all_categories_valid(self):
        valid = {cat.value for cat in Category}
        for c in CATALOG.values():
            assert c.category.value in valid

    def test_all_cvss_in_range(self):
        for c in CATALOG.values():
            assert 0.0 <= c.cvss <= 10.0, f"{c.code} CVSS={c.cvss} out of range"

    def test_all_points_non_negative(self):
        for c in CATALOG.values():
            assert c.points >= 0, f"{c.code} points={c.points} is negative"

    def test_all_codes_have_title_and_description(self):
        for c in CATALOG.values():
            assert len(c.title) > 0,       f"{c.code} has empty title"
            assert len(c.description) > 0, f"{c.code} has empty description"
            assert len(c.mitigation) > 0,  f"{c.code} has empty mitigation"

    def test_critical_codes_have_high_cvss(self):
        """CRITICAL severity codes should have CVSS >= 7.0."""
        for c in CATALOG.values():
            if c.severity == Severity.CRITICAL:
                assert c.cvss >= 7.0, \
                    f"{c.code} is CRITICAL but CVSS={c.cvss} < 7.0"

    def test_info_codes_have_zero_cvss(self):
        """INFO severity codes should have CVSS = 0."""
        for c in CATALOG.values():
            if c.severity == Severity.INFO:
                assert c.cvss == 0.0, \
                    f"{c.code} is INFO but CVSS={c.cvss} != 0"

    def test_each_category_has_at_least_one_code(self):
        for cat in Category:
            codes = get_by_category(cat)
            assert len(codes) > 0, f"Category {cat.value} has no codes"


# ══════════════════════════════════════════════
# SPECIFIC KNOWN CODES
# ══════════════════════════════════════════════
class TestKnownCodes:

    def test_mlx001_exists_and_is_high(self):
        c = get_by_code("RBT-MLX-001")
        assert c is not None
        assert c.severity == Severity.HIGH
        assert c.category == Category.MLX

    def test_aut003_brute_force_is_critical(self):
        c = get_by_code("RBT-AUT-003")
        assert c is not None
        assert c.severity == Severity.CRITICAL
        assert c.cvss >= 9.0

    def test_inj001_sql_injection_is_critical(self):
        c = get_by_code("RBT-INJ-001")
        assert c is not None
        assert c.severity == Severity.CRITICAL
        assert c.cvss >= 9.0

    def test_sys001_redis_error_is_critical(self):
        c = get_by_code("RBT-SYS-001")
        assert c is not None
        assert c.severity == Severity.CRITICAL

    def test_fpx001_false_positive_is_info(self):
        c = get_by_code("RBT-FPX-001")
        assert c is not None
        assert c.severity == Severity.INFO
        assert c.cvss == 0.0
        assert c.points == 0

    def test_bhv001_headless_ua_is_medium(self):
        c = get_by_code("RBT-BHV-001")
        assert c is not None
        assert c.severity == Severity.MEDIUM
        assert c.points == 15

    def test_bhv003_combined_is_higher_than_bhv001(self):
        c1 = get_by_code("RBT-BHV-001")
        c3 = get_by_code("RBT-BHV-003")
        assert c3.cvss >= c1.cvss, "Combined signals should have higher CVSS"
        assert c3.points >= c1.points, "Combined signals should add more points"

    def test_aut_escalation_order(self):
        """Brute force should be more severe than stuffing > single failure."""
        single = get_by_code("RBT-AUT-001")
        stuff  = get_by_code("RBT-AUT-002")
        brute  = get_by_code("RBT-AUT-003")
        sev_order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        assert sev_order.index(single.severity) < sev_order.index(stuff.severity)
        assert sev_order.index(stuff.severity)  < sev_order.index(brute.severity)


# ══════════════════════════════════════════════
# RISK EVENT CREATION
# ══════════════════════════════════════════════
class TestRiskEventCreation:

    def test_event_has_correct_code(self):
        ev = event_ml_blocked("fp:1.2.3.4", 0.87)
        assert ev.code == "RBT-MLX-001"

    def test_event_has_correct_identifier(self):
        ev = event_score_exceeded("abc:10.0.0.1", 55.0)
        assert ev.identifier == "abc:10.0.0.1"

    def test_event_details_preserved(self):
        ev = event_ml_blocked("fp:1.2.3.4", 0.92)
        assert ev.details["probability"] == 0.92

    def test_event_score_details(self):
        ev = event_score_exceeded("fp:x", 45.0)
        assert ev.details["score"] == 45.0
        assert ev.details["threshold"] == 30

    def test_login_single_failure_is_aut001(self):
        ev = event_login_failure("fp:x", 1)
        assert ev.code == "RBT-AUT-001"

    def test_login_3_failures_is_aut002(self):
        ev = event_login_failure("fp:x", 3)
        assert ev.code == "RBT-AUT-002"

    def test_login_5_failures_is_aut003(self):
        ev = event_login_failure("fp:x", 5)
        assert ev.code == "RBT-AUT-003"

    def test_login_10_failures_is_aut003(self):
        ev = event_login_failure("fp:x", 10)
        assert ev.code == "RBT-AUT-003"

    def test_headless_with_lang_is_bhv001(self):
        ev = event_headless_ua("fp:x", "headless-chrome", has_lang=True)
        assert ev.code == "RBT-BHV-001"

    def test_headless_without_lang_is_bhv003(self):
        ev = event_headless_ua("fp:x", "selenium/4.0", has_lang=False)
        assert ev.code == "RBT-BHV-003"

    def test_missing_lang_is_bhv002(self):
        ev = event_missing_lang("fp:x")
        assert ev.code == "RBT-BHV-002"

    def test_false_positive_is_fpx001(self):
        ev = event_false_positive("fp:x", 35.0)
        assert ev.code == "RBT-FPX-001"
        assert ev.details["score_at_bypass"] == 35.0

    def test_redis_error_is_sys001(self):
        ev = event_redis_error("fp:x", "connection refused")
        assert ev.code == "RBT-SYS-001"
        assert "connection refused" in ev.details["error"]


# ══════════════════════════════════════════════
# OUTPUT FORMATS
# ══════════════════════════════════════════════
class TestOutputFormats:

    def test_to_log_line_contains_code(self):
        ev = event_ml_blocked("fp:1.2.3.4", 0.75)
        line = ev.to_log_line()
        assert "RBT-MLX-001" in line
        assert "HIGH" in line
        assert "fp:1.2.3.4" in line

    def test_to_log_line_contains_probability(self):
        ev = event_ml_blocked("fp:x", 0.87)
        assert "0.87" in ev.to_log_line()

    def test_to_response_body_contains_code(self):
        ev = event_score_exceeded("fp:x", 45.0)
        body = ev.to_response_body()
        assert "RBT-RSK-001" in body
        assert "Access Denied" in body

    def test_to_dict_has_all_required_keys(self):
        ev = event_ml_blocked("fp:x", 0.65)
        d  = ev.to_dict()
        for key in ["risk_code", "severity", "title", "description",
                    "mitigation", "cvss", "identifier", "timestamp"]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_timestamp_is_iso_format(self):
        from datetime import datetime
        ev = event_ml_blocked("fp:x", 0.65)
        d  = ev.to_dict()
        # Should parse without error
        dt = datetime.fromisoformat(d["timestamp"])
        assert dt is not None

    def test_to_dict_cvss_is_float(self):
        ev = event_ml_blocked("fp:x", 0.65)
        assert isinstance(ev.to_dict()["cvss"], float)

    def test_catalog_summary_contains_all_categories(self):
        summary = catalog_summary()
        for cat in Category:
            assert cat.value in summary


# ══════════════════════════════════════════════
# QUERY FUNCTIONS
# ══════════════════════════════════════════════
class TestQueryFunctions:

    def test_get_by_code_returns_correct(self):
        c = get_by_code("RBT-MLX-001")
        assert c.code == "RBT-MLX-001"

    def test_get_by_code_returns_none_for_unknown(self):
        assert get_by_code("RBT-ZZZ-999") is None

    def test_get_by_severity_critical_not_empty(self):
        codes = get_by_severity(Severity.CRITICAL)
        assert len(codes) > 0

    def test_get_by_severity_only_returns_that_severity(self):
        for c in get_by_severity(Severity.HIGH):
            assert c.severity == Severity.HIGH

    def test_get_by_category_mlx_not_empty(self):
        codes = get_by_category(Category.MLX)
        assert len(codes) > 0

    def test_get_by_category_only_returns_that_category(self):
        for c in get_by_category(Category.AUT):
            assert c.category == Category.AUT

    def test_critical_codes_are_subset_of_all(self):
        all_codes    = set(c.code for c in CATALOG.values())
        crit_codes   = set(c.code for c in get_by_severity(Severity.CRITICAL))
        assert crit_codes.issubset(all_codes)
