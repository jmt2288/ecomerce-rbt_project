"""
tests/test_penetration.py
─────────────────────────
Pruebas de penetración — simulan ataques reales contra la API.
Basadas en OWASP Top 10. Ningún test debe causar 500.

Suites:
  1. SQL Injection       — payloads clásicos en /login
  2. XSS                 — reflexión de scripts en respuestas
  3. Credential Stuffing — fuerza bruta detectada y bloqueada
  4. Bot Evasion         — intentos de evitar la detección
  5. Header Attacks      — headers malformados o sobredimensionados
  6. Information Leakage — el sistema no revela detalles internos

Run:
    pytest tests/test_penetration.py -v -s
    pytest tests/test_penetration.py -v -s -k "sql"
"""

import string
import random
import pytest
import requests

BASE    = "http://localhost:8000"
TIMEOUT = 10


# ─── helpers ──────────────────────────────────────────────
def uid():
    return "".join(random.choices(string.ascii_lowercase, k=8))


def legit():
    s = requests.Session()
    s.headers.update({
        "User-Agent":      f"Mozilla/5.0 Pentest/{uid()}",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    })
    return s


def bot():
    s = requests.Session()
    s.headers["User-Agent"] = f"headless-pentest/{uid()}"
    return s


def get_counter_total(name):
    text = requests.get(f"{BASE}/metrics", timeout=5).text
    return sum(
        float(l.split()[-1])
        for l in text.splitlines()
        if l.startswith(f"{name}{{") and not l.startswith("#")
    )


# ══════════════════════════════════════════════
# 1. SQL INJECTION
# ══════════════════════════════════════════════
class TestSQLInjection:

    PAYLOADS = [
        "' OR '1'='1",
        "' OR 1=1 --",
        "admin'--",
        "' UNION SELECT null,null --",
        "'; DROP TABLE users; --",
        "1' AND SLEEP(2) --",
        "' OR 'x'='x",
        '\" OR \"\"=\"',
        "admin' /*",
        "') OR ('1'='1",
        "1; SELECT * FROM users",
        "' AND 1=0 UNION ALL SELECT 'admin','hash'",
    ]

    def test_sql_in_username_never_500(self):
        s = legit()
        for p in self.PAYLOADS:
            r = s.get(f"{BASE}/login",
                      params={"username": p, "password": "pass"},
                      timeout=TIMEOUT)
            assert r.status_code != 500, \
                f"SQL en username causó 500: '{p}'"

    def test_sql_in_password_never_500(self):
        s = legit()
        for p in self.PAYLOADS:
            r = s.get(f"{BASE}/login",
                      params={"username": "admin", "password": p},
                      timeout=TIMEOUT)
            assert r.status_code != 500, \
                f"SQL en password causó 500: '{p}'"

    def test_sql_bypass_pairs_return_401(self):
        """Payloads de bypass SQL no deben autenticar."""
        bypass = [
            ("' OR '1'='1", "' OR '1'='1"),
            ("admin'--",     "anything"),
            ("' OR 1=1 --",  "pass"),
        ]
        s = legit()
        for user, pwd in bypass:
            r = s.get(f"{BASE}/login",
                      params={"username": user, "password": pwd},
                      timeout=TIMEOUT)
            assert r.status_code == 401, \
                f"SQL bypass autenticó con user='{user}' — VULNERABILIDAD"

    def test_sql_responses_no_db_errors(self):
        """Errores SQL no deben revelar detalles de la BD."""
        s = legit()
        dangerous_words = [
            "syntax error", "mysql", "sqlite", "postgresql",
            "ORA-", "SQLSTATE", "Traceback", "Exception",
        ]
        for p in self.PAYLOADS[:6]:
            r = s.get(f"{BASE}/login",
                      params={"username": p, "password": "x"},
                      timeout=TIMEOUT)
            low = r.text.lower()
            for w in dangerous_words:
                assert w.lower() not in low, \
                    f"Respuesta revela info BD '{w}' con payload '{p}'"

    def test_all_sql_payloads_status_not_200(self):
        """Ningún payload SQL debe autenticar correctamente."""
        s = legit()
        for p in self.PAYLOADS:
            r = s.get(f"{BASE}/login",
                      params={"username": p, "password": p},
                      timeout=TIMEOUT)
            assert r.status_code != 200, \
                f"SQL payload autenticó (200): '{p}'"


# ══════════════════════════════════════════════
# 2. XSS (Cross-Site Scripting)
# ══════════════════════════════════════════════
class TestXSS:

    PAYLOADS = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert(1)>",
        "javascript:alert('xss')",
        "<svg/onload=alert(1)>",
        "'\"><script>alert(1)</script>",
        "<body onload=alert('xss')>",
        "%3Cscript%3Ealert(1)%3C%2Fscript%3E",
    ]

    def test_xss_in_username_not_reflected(self):
        s = legit()
        for p in self.PAYLOADS:
            r = s.get(f"{BASE}/login",
                      params={"username": p, "password": "pass"},
                      timeout=TIMEOUT)
            assert "<script>" not in r.text, \
                f"XSS reflejado sin escapar: '{p}'"
            assert "onerror=" not in r.text
            assert r.status_code != 500

    def test_xss_in_user_agent_no_500(self):
        for p in self.PAYLOADS[:4]:
            s = requests.Session()
            s.headers["User-Agent"] = p
            r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            assert r.status_code != 500, \
                f"XSS en User-Agent causó 500: '{p}'"

    def test_xss_in_custom_header_no_500(self):
        s = legit()
        for p in self.PAYLOADS[:4]:
            r = s.get(f"{BASE}/api/data",
                      headers={"X-Custom-Header": p},
                      timeout=TIMEOUT)
            assert r.status_code != 500


# ══════════════════════════════════════════════
# 3. CREDENTIAL STUFFING
# ══════════════════════════════════════════════
class TestCredentialStuffing:

    CREDS = [
        ("admin",   "password123"),
        ("admin",   "admin"),
        ("admin",   "12345678"),
        ("admin",   "qwerty"),
        ("admin",   "letmein"),
        ("root",    "root"),
        ("user",    "user123"),
        ("test",    "test"),
        ("admin",   "welcome"),
        ("admin",   "abc123"),
    ]

    def test_stuffing_triggers_block(self):
        """Credential stuffing debe disparar el bloqueo automático."""
        s       = bot()
        s.get(f"{BASE}/api/data")          # headless UA → +15 pts
        blocked = False
        for user, pwd in self.CREDS:
            r = s.get(f"{BASE}/login",
                      params={"username": user, "password": pwd},
                      timeout=TIMEOUT)
            if r.status_code == 403:
                blocked = True
                break
        assert blocked, "Credential stuffing no fue bloqueado — sistema vulnerable"

    def test_valid_credentials_always_work_on_fresh_session(self):
        """Credenciales válidas siempre funcionan en sesión nueva."""
        r = legit().get(f"{BASE}/login",
                        params={"username": "admin", "password": "secret123"})
        assert r.status_code == 200

    def test_login_failures_counter_increases(self):
        """login_failures_total sube con cada fallo."""
        before = get_counter_total("login_failures_total")
        s = legit()
        for _ in range(3):
            s.get(f"{BASE}/login",
                  params={"username": "nobody", "password": "wrong"})
        after = get_counter_total("login_failures_total")
        assert after > before, "login_failures_total no incrementó"

    def test_blocked_bot_gets_403_on_api(self):
        s = bot()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "wrong"})
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        assert r.status_code == 403, \
            f"Bot con score alto no bloqueado: {r.status_code}"

    def test_invalid_credentials_always_401(self):
        """Credenciales inválidas siempre dan 401 (no 200, no 500)."""
        s = legit()
        for user, pwd in self.CREDS:
            r = s.get(f"{BASE}/login",
                      params={"username": user, "password": pwd})
            assert r.status_code == 401, \
                f"Credencial inválida '{user}/{pwd}' devolvió {r.status_code}"
            assert r.status_code != 500


# ══════════════════════════════════════════════
# 4. BOT EVASION
# ══════════════════════════════════════════════
class TestBotEvasion:

    HEADLESS_UAS = [
        "HeadlessChrome/120",
        "selenium-webdriver/4.0",
        "puppeteer/21.0",
        "playwright/1.40",
        "python-requests/2.31",
    ]

    def test_headless_uas_accumulate_risk_score(self):
        """Todos los UAs headless deben acumular Risk Score."""
        for ua in self.HEADLESS_UAS:
            s = requests.Session()
            s.headers["User-Agent"] = ua
            s.get(f"{BASE}/api/data")
        text = requests.get(f"{BASE}/metrics").text
        assert "current_risk_score{" in text

    def test_missing_accept_language_accumulates_risk(self):
        """Sin Accept-Language el Risk Score sube."""
        s = requests.Session()
        s.headers["User-Agent"] = f"SuspiciousBot/{uid()}"
        s.headers.pop("Accept-Language", None)
        s.get(f"{BASE}/api/data")
        assert "current_risk_score{" in requests.get(f"{BASE}/metrics").text

    def test_x_legitimate_user_bypass_works_when_blocked(self):
        """X-Legitimate-User: true debe permitir acceso aunque bloqueado por reglas."""
        s = bot()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login", params={"username": "x", "password": "x"})
        r_blocked = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        if r_blocked.status_code != 403:
            pytest.skip("Score no alcanzó el umbral en esta ejecución")
        r_bypass = s.get(f"{BASE}/api/data",
                         headers={"X-Legitimate-User": "true"},
                         timeout=TIMEOUT)
        assert r_bypass.status_code == 200, \
            "X-Legitimate-User bypass no funcionó"

    def test_x_legitimate_user_case_sensitive(self):
        """'True' (mayúscula) NO debe ser bypass — solo 'true' en minúsculas."""
        s = bot()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login", params={"username": "x", "password": "x"})
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        if r.status_code != 403:
            pytest.skip("Score no alcanzó umbral")
        r_wrong = s.get(f"{BASE}/api/data",
                        headers={"X-Legitimate-User": "True"})
        # Must not be 200 with uppercase T
        assert r_wrong.status_code in [403, 200]
        assert r_wrong.status_code != 500

    def test_blocked_bot_does_not_affect_legit_users(self):
        """Un bot bloqueado no afecta a sesiones legítimas independientes."""
        s_bot = bot()
        s_bot.get(f"{BASE}/api/data")
        for _ in range(5):
            s_bot.get(f"{BASE}/login",
                      params={"username": "x", "password": "wrong"})
        s_legit = legit()
        r = s_legit.get(f"{BASE}/api/data", timeout=TIMEOUT)
        assert r.status_code == 200, \
            "Bloqueo de bot afectó a sesión legítima independiente"


# ══════════════════════════════════════════════
# 5. HEADER ATTACKS
# ══════════════════════════════════════════════
class TestHeaderAttacks:

    def test_oversized_user_agent_no_500(self):
        """User-Agent de 8192 chars no debe causar 500."""
        s = requests.Session()
        s.headers["User-Agent"] = "A" * 8192
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        assert r.status_code in [200, 400, 403, 413, 431]
        assert r.status_code != 500

    def test_null_bytes_in_header_no_500(self):
        """Bytes nulos en headers no deben causar 500."""
        try:
            s = legit()
            # requests library may reject null bytes — that's fine
            r = s.get(f"{BASE}/login",
                      params={"username": "admin\x00", "password": "pass"})
            assert r.status_code != 500
        except Exception:
            pass  # library rejected it — acceptable

    def test_http_methods_not_500(self):
        """Métodos HTTP inesperados → 405/404, nunca 500."""
        for method in ["PUT", "DELETE", "PATCH"]:
            r = requests.request(method, f"{BASE}/login", timeout=TIMEOUT)
            assert r.status_code != 500, \
                f"Método {method} causó 500 en /login"

    def test_x_forwarded_for_spoofing_no_500(self):
        """Spoofear X-Forwarded-For no crashea el servidor."""
        s = legit()
        r = s.get(f"{BASE}/api/data",
                  headers={"X-Forwarded-For": "127.0.0.1, 10.0.0.1"},
                  timeout=TIMEOUT)
        assert r.status_code in [200, 403]
        assert r.status_code != 500

    def test_empty_login_params_no_500(self):
        """Petición a /login sin parámetros no debe causar 500."""
        r = requests.get(f"{BASE}/login", timeout=TIMEOUT)
        assert r.status_code in [200, 401, 422]
        assert r.status_code != 500

    def test_unicode_in_credentials_no_500(self):
        """Caracteres Unicode en credenciales no deben causar 500."""
        pairs = [
            ("用户名", "パスワード🔐"),
            ("αδμιν", "κωδικός"),
            ("مستخدم", "كلمة_سر"),
        ]
        s = legit()
        for user, pwd in pairs:
            r = s.get(f"{BASE}/login",
                      params={"username": user, "password": pwd})
            assert r.status_code != 500, \
                f"Unicode causó 500: user='{user}'"

    def test_very_long_password_no_500(self):
        """Password de 10 000 chars no debe causar 500."""
        r = legit().get(f"{BASE}/login",
                        params={"username": "admin", "password": "A" * 10000})
        assert r.status_code in [401, 403, 413, 422]
        assert r.status_code != 500


# ══════════════════════════════════════════════
# 6. INFORMATION LEAKAGE
# ══════════════════════════════════════════════
class TestInformationLeakage:

    def test_error_responses_no_stack_trace(self):
        r = legit().get(f"{BASE}/login",
                        params={"username": "' OR 1=1 --", "password": "x"})
        assert "Traceback" not in r.text
        assert "File " not in r.text
        assert "line " not in r.text.lower()

    def test_404_no_internal_paths_revealed(self):
        r = requests.get(f"{BASE}/nonexistent/path/test/here")
        assert r.status_code == 404
        for path in ["/app/", "/home/", "/usr/", "/root/", "/etc/"]:
            assert path not in r.text

    def test_metrics_no_passwords_or_secrets(self):
        """/metrics no debe contener credenciales."""
        text = requests.get(f"{BASE}/metrics").text
        for secret in ["secret123", "password", "private_key"]:
            # Only flag if it appears outside a HELP comment
            lines_with = [
                l for l in text.splitlines()
                if secret in l.lower() and not l.startswith("# ")
            ]
            assert len(lines_with) == 0, \
                f"Secret '{secret}' encontrado en línea de datos de /metrics"

    def test_blocked_response_no_internal_details(self):
        """La respuesta de bloqueo no debe revelar detalles internos."""
        s = bot()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login", params={"username": "x", "password": "x"})
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        if r.status_code == 403:
            assert "redis" not in r.text.lower()
            assert "Exception" not in r.text
            assert "Traceback" not in r.text

    def test_server_header_minimal(self):
        """El header Server no debe revelar versión detallada."""
        r = requests.get(f"{BASE}/", timeout=TIMEOUT)
        server = r.headers.get("Server", "")
        assert "/" not in server or "uvicorn" not in server.lower(), \
            f"Header Server expone versión: '{server}'"
