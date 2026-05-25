"""
risk_codes.py
=============
Sistema de codigos de riesgo para el proyecto RBT Security.

Cada deteccion de amenaza recibe un codigo unico con:
  - ID estructurado  (RBT-XXX-NNN)
  - Severidad        (CRITICAL / HIGH / MEDIUM / LOW / INFO)
  - Categoria        (ML / RULES / AUTH / RATE / BEHAVIOR / SYSTEM)
  - Descripcion      (que se detecto)
  - Mitigacion       (que hizo el sistema)
  - CVSS aproximado  (para trazabilidad con estandares)

Estructura del codigo:
  RBT - [CATEGORIA 3 letras] - [NUMERO 3 digitos]
  ej:  RBT-MLX-001  (ML detection, codigo 001)
       RBT-AUT-003  (Authentication, codigo 003)
       RBT-BHV-002  (Behavioral, codigo 002)

Uso en main.py:
  from risk_codes import RiskCode, RiskEvent, build_response

  event = RiskEvent(RiskCode.MLX_001_BOT_DETECTED, identifier, probability=0.87)
  return event.to_response()
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import json


# ══════════════════════════════════════════════
# SEVERITY LEVELS
# ══════════════════════════════════════════════
class Severity(str, Enum):
    CRITICAL = "CRITICAL"   # Block immediately, alert
    HIGH     = "HIGH"       # Block, log, monitor
    MEDIUM   = "MEDIUM"     # Block or warn depending on context
    LOW      = "LOW"        # Warn, accumulate score
    INFO     = "INFO"       # Log only, no action


# ══════════════════════════════════════════════
# CATEGORIES
# ══════════════════════════════════════════════
class Category(str, Enum):
    MLX = "MLX"   # Machine Learning detection
    AUT = "AUT"   # Authentication failures
    BHV = "BHV"   # Behavioral signals (UA, headers)
    RTE = "RTE"   # Rate limiting
    INJ = "INJ"   # Injection attempts (SQL, XSS)
    RSK = "RSK"   # Risk Score threshold
    SYS = "SYS"   # System / infrastructure
    FPX = "FPX"   # False positive / bypass


# ══════════════════════════════════════════════
# RISK CODE CATALOG
# ══════════════════════════════════════════════
@dataclass(frozen=True)
class RiskCodeDef:
    code:        str
    category:    Category
    severity:    Severity
    title:       str
    description: str
    mitigation:  str
    cvss:        float       # approximate CVSS v3.1 base score
    points:      float = 0   # risk points added to the user's score


# Complete catalog — every detectable event in the project
CATALOG: dict[str, RiskCodeDef] = {}

def _reg(c: RiskCodeDef) -> RiskCodeDef:
    CATALOG[c.code] = c
    return c


class RiskCode:
    """All risk codes defined in the RBT project."""

    # ── MLX: Machine Learning Detection ──────────────────────
    MLX_001_BOT_DETECTED = _reg(RiskCodeDef(
        code        = "RBT-MLX-001",
        category    = Category.MLX,
        severity    = Severity.HIGH,
        title       = "Bot detected by ML model",
        description = "RandomForestClassifier predicted bot with probability >= 0.5. "
                      "Features: headless UA, high rate, accumulated risk score, "
                      "failed logins and legitimate-header signal.",
        mitigation  = "Request blocked (403). ml_blocked_total counter incremented. "
                      "bot_ml_probability gauge updated.",
        cvss        = 7.5,
        points      = 0,   # ML block is immediate — no score accumulation
    ))

    MLX_002_HIGH_BOT_PROBABILITY = _reg(RiskCodeDef(
        code        = "RBT-MLX-002",
        category    = Category.MLX,
        severity    = Severity.MEDIUM,
        title       = "Elevated bot probability",
        description = "ML model probability between 0.3 and 0.5 — not blocked but "
                      "flagged for monitoring. User shows partial bot signals.",
        mitigation  = "Request allowed. Score penalty applied. Logged in Prometheus.",
        cvss        = 4.0,
        points      = 10,
    ))

    MLX_003_MODEL_UNAVAILABLE = _reg(RiskCodeDef(
        code        = "RBT-MLX-003",
        category    = Category.MLX,
        severity    = Severity.LOW,
        title       = "ML model not loaded",
        description = "bot_detector.pkl not found at startup. Falling back to "
                      "rule-based detection only. Security coverage reduced.",
        mitigation  = "Rule-based layer remains active. Run: python ml/train_model.py",
        cvss        = 2.1,
        points      = 0,
    ))

    # ── AUT: Authentication ───────────────────────────────────
    AUT_001_INVALID_CREDENTIALS = _reg(RiskCodeDef(
        code        = "RBT-AUT-001",
        category    = Category.AUT,
        severity    = Severity.LOW,
        title       = "Invalid login credentials",
        description = "Username or password did not match any known account. "
                      "Single failure — may be a typo or forgotten password.",
        mitigation  = "401 returned. login_failures_total incremented. "
                      "+10 pts added to risk score.",
        cvss        = 3.1,
        points      = 10,
    ))

    AUT_002_CREDENTIAL_STUFFING = _reg(RiskCodeDef(
        code        = "RBT-AUT-002",
        category    = Category.AUT,
        severity    = Severity.HIGH,
        title       = "Credential stuffing detected",
        description = "3 or more failed login attempts from the same fingerprint "
                      "within the rate window. Pattern consistent with automated "
                      "credential stuffing attack using leaked credential lists.",
        mitigation  = "Risk score threshold may trigger 403 on next request. "
                      "login_failures_total counter incremented per attempt.",
        cvss        = 7.5,
        points      = 30,  # cumulative from multiple AUT-001
    ))

    AUT_003_BRUTE_FORCE = _reg(RiskCodeDef(
        code        = "RBT-AUT-003",
        category    = Category.AUT,
        severity    = Severity.CRITICAL,
        title       = "Brute force attack detected",
        description = "5 or more failed login attempts in rapid succession. "
                      "Automated tool confirmed by rate and timing pattern.",
        mitigation  = "Session blocked (403). Risk score exceeds threshold. "
                      "blocked_requests_total incremented with reason=auth_brute_force.",
        cvss        = 9.1,
        points      = 50,
    ))

    AUT_004_LOGIN_SUCCESS_AFTER_FAILURES = _reg(RiskCodeDef(
        code        = "RBT-AUT-004",
        category    = Category.AUT,
        severity    = Severity.MEDIUM,
        title       = "Successful login after multiple failures",
        description = "Login succeeded after 2+ previous failures from same fingerprint. "
                      "May indicate successful password guessing.",
        mitigation  = "Login allowed. Event logged. Risk score partially reduced. "
                      "login_success_total incremented.",
        cvss        = 5.4,
        points      = 0,
    ))

    # ── BHV: Behavioral Signals ───────────────────────────────
    BHV_001_HEADLESS_USER_AGENT = _reg(RiskCodeDef(
        code        = "RBT-BHV-001",
        category    = Category.BHV,
        severity    = Severity.MEDIUM,
        title       = "Headless browser User-Agent detected",
        description = "User-Agent contains headless browser signature: headless, "
                      "selenium, puppeteer, or playwright. Indicates automated "
                      "browser or scraping tool.",
        mitigation  = "+15 pts added to risk score. bot_ml_probability updated. "
                      "Not blocked immediately — score may trigger block.",
        cvss        = 5.3,
        points      = 15,
    ))

    BHV_002_MISSING_ACCEPT_LANGUAGE = _reg(RiskCodeDef(
        code        = "RBT-BHV-002",
        category    = Category.BHV,
        severity    = Severity.LOW,
        title       = "Missing Accept-Language header",
        description = "Request lacks Accept-Language header. Real browsers always "
                      "send this header. Absence suggests a scripted HTTP client "
                      "or bot that does not emulate browser headers fully.",
        mitigation  = "+5 pts added to risk score. Logged.",
        cvss        = 3.1,
        points      = 5,
    ))

    BHV_003_COMBINED_BOT_SIGNALS = _reg(RiskCodeDef(
        code        = "RBT-BHV-003",
        category    = Category.BHV,
        severity    = Severity.HIGH,
        title       = "Multiple bot behavioral signals",
        description = "Both headless UA and missing Accept-Language detected "
                      "in the same request. Strong indication of automated tool "
                      "with minimal browser emulation.",
        mitigation  = "+20 pts to risk score. Session likely to be blocked "
                      "within next few requests.",
        cvss        = 7.2,
        points      = 20,
    ))

    BHV_004_KNOWN_SCANNER_UA = _reg(RiskCodeDef(
        code        = "RBT-BHV-004",
        category    = Category.BHV,
        severity    = Severity.HIGH,
        title       = "Known security scanner User-Agent",
        description = "User-Agent matches known security scanner or vulnerability "
                      "assessment tool: nikto, nmap, sqlmap, masscan, zgrab, "
                      "nuclei, burpsuite, or similar.",
        mitigation  = "+25 pts to risk score. blocked_requests_total incremented "
                      "with reason=scanner_ua.",
        cvss        = 7.5,
        points      = 25,
    ))

    # ── RTE: Rate Limiting ────────────────────────────────────
    RTE_001_HIGH_REQUEST_RATE = _reg(RiskCodeDef(
        code        = "RBT-RTE-001",
        category    = Category.RTE,
        severity    = Severity.MEDIUM,
        title       = "High request rate detected",
        description = "Request rate from this fingerprint exceeds 60 requests "
                      "per 100-second window. Unusual for human browsing — "
                      "may be automated scraping or API abuse.",
        mitigation  = "+10 pts to risk score. Prometheus metric updated.",
        cvss        = 5.3,
        points      = 10,
    ))

    RTE_002_RATE_LIMIT_EXCEEDED = _reg(RiskCodeDef(
        code        = "RBT-RTE-002",
        category    = Category.RTE,
        severity    = Severity.HIGH,
        title       = "Rate limit exceeded",
        description = "Request rate exceeds the configured window limit. "
                      "Source fingerprint is sending requests faster than "
                      "any legitimate user would.",
        mitigation  = "Request blocked (429 or 403). blocked_requests_total "
                      "incremented with reason=rate_limit_exceeded.",
        cvss        = 7.5,
        points      = 0,   # block is immediate
    ))

    # ── INJ: Injection Attacks ────────────────────────────────
    INJ_001_SQL_INJECTION_ATTEMPT = _reg(RiskCodeDef(
        code        = "RBT-INJ-001",
        category    = Category.INJ,
        severity    = Severity.CRITICAL,
        title       = "SQL injection attempt",
        description = "Input contains SQL injection patterns: OR 1=1, UNION SELECT, "
                      "DROP TABLE, comment sequences (-- or /*), or sleep() calls. "
                      "Attacker attempting to manipulate database queries.",
        mitigation  = "Input sanitized or rejected (401). login_failures_total "
                      "incremented. +30 pts to risk score.",
        cvss        = 9.8,
        points      = 30,
    ))

    INJ_002_XSS_ATTEMPT = _reg(RiskCodeDef(
        code        = "RBT-INJ-002",
        category    = Category.INJ,
        severity    = Severity.HIGH,
        title       = "Cross-site scripting (XSS) attempt",
        description = "Input contains XSS patterns: <script>, onerror=, "
                      "javascript:, <svg/onload=, or similar HTML injection. "
                      "Attacker attempting to inject client-side code.",
        mitigation  = "Input not reflected unsanitized. +20 pts to risk score.",
        cvss        = 7.2,
        points      = 20,
    ))

    INJ_003_COMMAND_INJECTION = _reg(RiskCodeDef(
        code        = "RBT-INJ-003",
        category    = Category.INJ,
        severity    = Severity.CRITICAL,
        title       = "Command injection attempt",
        description = "Input contains shell command patterns: ;, &&, ||, pipe (|), "
                      "backticks, or system command keywords (cat, ls, wget, curl) "
                      "in unexpected fields.",
        mitigation  = "Input rejected. +40 pts to risk score. Immediate review advised.",
        cvss        = 9.8,
        points      = 40,
    ))

    # ── RSK: Risk Score Threshold ─────────────────────────────
    RSK_001_THRESHOLD_EXCEEDED = _reg(RiskCodeDef(
        code        = "RBT-RSK-001",
        category    = Category.RSK,
        severity    = Severity.HIGH,
        title       = "Risk score threshold exceeded",
        description = "Accumulated risk score for this fingerprint has exceeded "
                      "the configured threshold (default: 30 pts). Score is the "
                      "sum of all behavioral and authentication signals.",
        mitigation  = "Request blocked (403). blocked_requests_total incremented "
                      "with reason=risk_score_exceeded. Score persists in Redis "
                      "for 10000s.",
        cvss        = 7.5,
        points      = 0,
    ))

    RSK_002_SCORE_APPROACHING_THRESHOLD = _reg(RiskCodeDef(
        code        = "RBT-RSK-002",
        category    = Category.RSK,
        severity    = Severity.MEDIUM,
        title       = "Risk score approaching threshold",
        description = "Accumulated score is between 20 and 30 pts — within "
                      "one headless UA detection of the block threshold. "
                      "User is being actively monitored.",
        mitigation  = "Request allowed. Prometheus gauge updated. "
                      "Next suspicious action will trigger block.",
        cvss        = 4.0,
        points      = 0,
    ))

    # ── FPX: False Positive / Bypass ─────────────────────────
    FPX_001_LEGITIMATE_BYPASS = _reg(RiskCodeDef(
        code        = "RBT-FPX-001",
        category    = Category.FPX,
        severity    = Severity.INFO,
        title       = "False positive bypass via X-Legitimate-User",
        description = "Blocked user presented X-Legitimate-User: true header. "
                      "Access granted as configured. This may be a legitimate "
                      "user incorrectly flagged or an attacker who knows the bypass.",
        mitigation  = "Request allowed. false_positive_blocks_total incremented. "
                      "Review if bypass is used frequently by same fingerprint.",
        cvss        = 0.0,
        points      = 0,
    ))

    # ── SYS: System ───────────────────────────────────────────
    SYS_001_REDIS_UNAVAILABLE = _reg(RiskCodeDef(
        code        = "RBT-SYS-001",
        category    = Category.SYS,
        severity    = Severity.CRITICAL,
        title       = "Redis connection lost",
        description = "Cannot connect to Redis. Risk Score tracking, rate limiting, "
                      "and session state are unavailable. Security posture degraded — "
                      "all requests pass through without scoring.",
        mitigation  = "Request allowed (fail-open). CVSS 9.8 if exploited during "
                      "outage. Check Docker: docker compose ps redis.",
        cvss        = 9.8,
        points      = 0,
    ))

    SYS_002_ML_PREDICTION_ERROR = _reg(RiskCodeDef(
        code        = "RBT-SYS-002",
        category    = Category.SYS,
        severity    = Severity.MEDIUM,
        title       = "ML model prediction error",
        description = "Exception raised during model.predict() or predict_proba(). "
                      "ML detection layer skipped for this request. "
                      "Rule-based layer remains active.",
        mitigation  = "Request processed by rules only. Error logged. "
                      "Check model integrity: python ml/train_model.py",
        cvss        = 4.0,
        points      = 0,
    ))

    SYS_003_FEATURE_EXTRACTION_ERROR = _reg(RiskCodeDef(
        code        = "RBT-SYS-003",
        category    = Category.SYS,
        severity    = Severity.LOW,
        title       = "Feature extraction error",
        description = "extract_features() raised an exception, likely due to "
                      "malformed headers or unexpected Redis response. "
                      "ML detection fallback to rules.",
        mitigation  = "Default feature vector [0,0,0,0,0,0] used. "
                      "Request evaluated by rules only.",
        cvss        = 2.1,
        points      = 0,
    ))


# ══════════════════════════════════════════════
# RISK EVENT — one detected threat instance
# ══════════════════════════════════════════════
@dataclass
class RiskEvent:
    """A single detected risk event tied to a request."""
    risk_code:   RiskCodeDef
    identifier:  str
    timestamp:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details:     dict     = field(default_factory=dict)

    @property
    def code(self) -> str:
        return self.risk_code.code

    @property
    def severity(self) -> Severity:
        return self.risk_code.severity

    def to_dict(self) -> dict:
        return {
            "risk_code":   self.code,
            "severity":    self.severity.value,
            "title":       self.risk_code.title,
            "description": self.risk_code.description,
            "mitigation":  self.risk_code.mitigation,
            "cvss":        self.risk_code.cvss,
            "identifier":  self.identifier,
            "timestamp":   self.timestamp.isoformat(),
            **self.details,
        }

    def to_log_line(self) -> str:
        """Compact single-line format for structured logs."""
        d = self.details
        extra = " ".join(f"{k}={v}" for k, v in d.items())
        return (
            f"[{self.code}] severity={self.severity.value} "
            f"id={self.identifier} {extra}"
        ).strip()

    def to_response_body(self) -> str:
        """403/429 response body — code + brief reason."""
        return (
            f"Access Denied: {self.code} — {self.risk_code.title}"
        )


# ══════════════════════════════════════════════
# HELPERS FOR COMMON DETECTIONS
# ══════════════════════════════════════════════
def event_ml_blocked(identifier: str, probability: float) -> RiskEvent:
    return RiskEvent(
        RiskCode.MLX_001_BOT_DETECTED,
        identifier,
        details={"probability": round(probability, 3)},
    )

def event_score_exceeded(identifier: str, score: float) -> RiskEvent:
    return RiskEvent(
        RiskCode.RSK_001_THRESHOLD_EXCEEDED,
        identifier,
        details={"score": round(score, 1), "threshold": 30},
    )

def event_login_failure(identifier: str, attempt: int) -> RiskEvent:
    if attempt >= 5:
        code = RiskCode.AUT_003_BRUTE_FORCE
    elif attempt >= 3:
        code = RiskCode.AUT_002_CREDENTIAL_STUFFING
    else:
        code = RiskCode.AUT_001_INVALID_CREDENTIALS
    return RiskEvent(code, identifier, details={"attempt": attempt})

def event_headless_ua(identifier: str, ua: str, has_lang: bool) -> RiskEvent:
    if not has_lang:
        code = RiskCode.BHV_003_COMBINED_BOT_SIGNALS
    else:
        code = RiskCode.BHV_001_HEADLESS_USER_AGENT
    return RiskEvent(code, identifier, details={"user_agent": ua[:80]})

def event_missing_lang(identifier: str) -> RiskEvent:
    return RiskEvent(RiskCode.BHV_002_MISSING_ACCEPT_LANGUAGE, identifier)

def event_false_positive(identifier: str, score: float) -> RiskEvent:
    return RiskEvent(
        RiskCode.FPX_001_LEGITIMATE_BYPASS,
        identifier,
        details={"score_at_bypass": round(score, 1)},
    )

def event_redis_error(identifier: str, error: str) -> RiskEvent:
    return RiskEvent(
        RiskCode.SYS_001_REDIS_UNAVAILABLE,
        identifier,
        details={"error": str(error)[:120]},
    )


# ══════════════════════════════════════════════
# CATALOG UTILITIES
# ══════════════════════════════════════════════
def get_by_code(code: str) -> Optional[RiskCodeDef]:
    return CATALOG.get(code)

def get_by_severity(severity: Severity) -> list[RiskCodeDef]:
    return [c for c in CATALOG.values() if c.severity == severity]

def get_by_category(category: Category) -> list[RiskCodeDef]:
    return [c for c in CATALOG.values() if c.category == category]

def catalog_summary() -> str:
    lines = [f"RBT Risk Code Catalog — {len(CATALOG)} codes\n"]
    for cat in Category:
        codes = get_by_category(cat)
        if codes:
            lines.append(f"\n  [{cat.value}] {cat.name}")
            for c in sorted(codes, key=lambda x: x.code):
                lines.append(
                    f"    {c.code}  [{c.severity.value:8s}]  "
                    f"CVSS {c.cvss:4.1f}  {c.title}"
                )
    return "\n".join(lines)


if __name__ == "__main__":
    print(catalog_summary())
    print()
    # Example events
    ev = event_ml_blocked("abc123:10.0.0.1", 0.87)
    print("Log:", ev.to_log_line())
    print("Response body:", ev.to_response_body())
    print("JSON:", json.dumps(ev.to_dict(), indent=2))
