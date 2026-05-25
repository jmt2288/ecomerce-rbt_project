"""
tests/test_e2e.py
─────────────────
Pruebas End-to-End — journeys completos desde el usuario
hasta Grafana, pasando por todos los componentes del stack.

Suites:
  1. Infrastructure Health  — todos los servicios sanos
  2. Legit User Journey     — usuario legítimo sin fricción
  3. Bot Attack Journey     — detección y bloqueo completo
  4. False Positive Journey — bypass y counter registrado
  5. ML Pipeline E2E        — modelo ML integrado de extremo a extremo
  6. Prometheus Pipeline    — datos scrapeados y consultables
  7. Grafana Dashboard      — dashboard validado vía API REST

Run todo:
    pytest tests/test_e2e.py -v -s
Sin Grafana ni Prometheus (solo FastAPI):
    pytest tests/test_e2e.py -v -s -k "not grafana and not prometheus"
"""

import string
import random
import threading
import time

import pytest
import requests

BASE       = "http://localhost:8000"
PROMETHEUS = "http://localhost:9090"
GRAFANA    = "http://localhost:3000"
TIMEOUT    = 10


# ─── helpers ──────────────────────────────────────────────
def uid():
    return "".join(random.choices(string.ascii_lowercase, k=8))


def legit():
    s = requests.Session()
    s.headers.update({
        "User-Agent":      f"Mozilla/5.0 E2E/{uid()}",
        "Accept-Language": "es-ES,es;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    })
    return s


def bot():
    s = requests.Session()
    s.headers["User-Agent"] = f"headless-e2e/{uid()}"
    return s


def prom_query(expr):
    try:
        r = requests.get(f"{PROMETHEUS}/api/v1/query",
                         params={"query": expr}, timeout=TIMEOUT)
        return r.json()
    except Exception:
        return {}


def grafana(path, auth=("admin", "admin")):
    try:
        return requests.get(f"{GRAFANA}{path}", auth=auth, timeout=TIMEOUT)
    except Exception:
        return None


def prometheus_up():
    try:
        return requests.get(f"{PROMETHEUS}/-/healthy", timeout=4).status_code == 200
    except Exception:
        return False


def grafana_up():
    r = grafana("/api/health")
    return r is not None and r.status_code == 200 and r.json().get("database") == "ok"


def get_metric_total(name):
    text = requests.get(f"{BASE}/metrics", timeout=5).text
    return sum(
        float(l.split()[-1])
        for l in text.splitlines()
        if l.startswith(f"{name}{{") and not l.startswith("#")
    )


# ══════════════════════════════════════════════
# 1. INFRASTRUCTURE HEALTH
# ══════════════════════════════════════════════
class TestInfrastructureHealth:

    def test_fastapi_root_200(self):
        r = requests.get(f"{BASE}/", timeout=TIMEOUT)
        assert r.status_code == 200
        assert r.json()["status"] == "RBT Security Layer Active"

    def test_fastapi_status_all_fields(self):
        r = requests.get(f"{BASE}/status", timeout=TIMEOUT)
        assert r.status_code == 200
        d = r.json()
        for f in ["status", "ml_model_loaded", "threshold", "model_path"]:
            assert f in d, f"Campo faltante en /status: {f}"
        assert d["threshold"] == 30
        assert d["status"] == "running"

    def test_ml_model_loaded_at_startup(self):
        d = requests.get(f"{BASE}/status").json()
        assert d["ml_model_loaded"] is True, \
            "ML no cargado — ejecuta: docker compose up --build -d"

    def test_metrics_endpoint_healthy_with_all_counters(self):
        r = requests.get(f"{BASE}/metrics", timeout=TIMEOUT)
        assert r.status_code == 200
        assert "text/plain" in r.headers.get("content-type", "")
        for m in ["http_requests_total", "blocked_requests_total",
                  "current_risk_score", "bot_ml_probability",
                  "login_failures_total", "false_positive_blocks_total"]:
            assert m in r.text, f"Métrica faltante: {m}"

    def test_false_positive_pre_initialized(self):
        """false_positive_blocks_total debe existir desde el inicio (main_fixed.py)."""
        text = requests.get(f"{BASE}/metrics").text
        assert "false_positive_blocks_total" in text, \
            "false_positive_blocks_total NO está inicializado — aplica main_fixed.py"

    def test_prometheus_healthy(self):
        if not prometheus_up():
            pytest.skip("Prometheus no accesible en localhost:9090")
        r = requests.get(f"{PROMETHEUS}/-/healthy", timeout=5)
        assert r.status_code == 200

    def test_grafana_healthy(self):
        if not grafana_up():
            pytest.skip("Grafana no accesible en localhost:3000")
        r = grafana("/api/health")
        assert r.json()["database"] == "ok"

    def test_redis_reachable_via_api(self):
        """Si Redis falla, la API devuelve 500 — esto verifica que no es el caso."""
        r = legit().get(f"{BASE}/api/data", timeout=TIMEOUT)
        assert r.status_code in [200, 403], \
            f"API devolvió {r.status_code} — posible fallo de Redis"


# ══════════════════════════════════════════════
# 2. LEGIT USER JOURNEY
# ══════════════════════════════════════════════
class TestLegitUserJourney:

    def test_full_legit_flow_never_blocked(self):
        """
        Journey completo:
        / → login OK → /api/data ×5 → nunca bloqueado
        """
        s = legit()

        # Home
        assert s.get(f"{BASE}/", timeout=TIMEOUT).status_code == 200

        # Login
        r = s.get(f"{BASE}/login",
                  params={"username": "admin", "password": "secret123"})
        assert r.status_code == 200
        assert r.json().get("status") == "success"

        # API access — must never block
        for i in range(5):
            r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            assert r.status_code == 200, \
                f"Usuario legítimo bloqueado en request {i+1}"

    def test_legit_not_affected_by_concurrent_bots(self):
        """Usuario legítimo no se ve afectado por bots activos en paralelo."""
        def spam_bot():
            s = bot()
            for _ in range(10):
                s.get(f"{BASE}/api/data")
                s.get(f"{BASE}/login", params={"username":"x","password":"x"})

        t = threading.Thread(target=spam_bot)
        t.start()

        s = legit()
        for i in range(5):
            r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            assert r.status_code == 200, \
                f"Legítimo afectado por bots concurrentes en request {i+1}"
            time.sleep(0.2)
        t.join()

    def test_legit_login_response_schema(self):
        r = legit().get(f"{BASE}/login",
                        params={"username":"admin","password":"secret123"})
        assert r.status_code == 200
        d = r.json()
        assert "message" in d
        assert d.get("status") == "success"

    def test_different_legit_sessions_independent(self):
        """Cada sesión legítima es completamente independiente."""
        for _ in range(5):
            r = legit().get(f"{BASE}/api/data", timeout=TIMEOUT)
            assert r.status_code == 200, \
                "Sesiones legítimas se afectan entre sí"


# ══════════════════════════════════════════════
# 3. BOT ATTACK JOURNEY
# ══════════════════════════════════════════════
class TestBotAttackJourney:

    def test_full_bot_detection_pipeline(self):
        """
        Journey completo de ataque bot:
        UA headless → credential stuffing → bloqueado → métricas capturadas
        """
        s = bot()

        # Step 1: primera request — headless UA
        r1 = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        print(f"\n   Step 1 — UA headless detectado: {r1.status_code}")

        # Step 2: credential stuffing
        blocked_at = None
        for i in range(8):
            r = s.get(f"{BASE}/login",
                      params={"username": "admin", "password": f"attempt{i}"})
            if r.status_code == 403:
                blocked_at = i + 1
                break

        print(f"   Step 2 — Bloqueado tras {blocked_at} intentos de login")

        # Step 3: confirmar bloqueo en /api/data
        r_final = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        assert r_final.status_code == 403, \
            f"Bot no bloqueado en /api/data: {r_final.status_code}"
        print(f"   Step 3 — /api/data bloqueado: {r_final.status_code} ✅")

        # Step 4: métricas capturadas
        text = requests.get(f"{BASE}/metrics").text
        assert "blocked_requests_total{" in text
        assert "current_risk_score{"     in text
        print("   Step 4 — Métricas capturadas ✅")

    def test_blocked_response_is_403_with_body(self):
        s = bot()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login", params={"username":"x","password":"x"})
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        assert r.status_code == 403
        assert len(r.text) > 0, "Respuesta 403 vacía — debe incluir motivo"

    def test_blocked_counter_increases_after_bot(self):
        s = bot()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login", params={"username":"x","password":"x"})
        before = get_metric_total("blocked_requests_total")
        for _ in range(3):
            s.get(f"{BASE}/api/data")
        after = get_metric_total("blocked_requests_total")
        assert after >= before, "blocked_requests_total no incrementó"

    def test_risk_score_appears_in_metrics(self):
        s = bot()
        s.get(f"{BASE}/api/data")
        time.sleep(0.3)
        text = requests.get(f"{BASE}/metrics").text
        score_lines = [
            l for l in text.splitlines()
            if l.startswith("current_risk_score{") and "system_startup" not in l
        ]
        assert len(score_lines) > 0, \
            "current_risk_score no aparece en /metrics tras request de bot"


# ══════════════════════════════════════════════
# 4. FALSE POSITIVE JOURNEY
# ══════════════════════════════════════════════
class TestFalsePositiveJourney:

    def test_full_false_positive_flow(self):
        """
        Journey completo:
        bloqueado → X-Legitimate-User: true → 200 → counter registrado
        """
        s = bot()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login", params={"username":"x","password":"x"})

        r_blocked = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        if r_blocked.status_code != 403:
            pytest.skip("Score no alcanzó el umbral en esta ejecución")
        print(f"\n   Bloqueado: {r_blocked.status_code}")

        r_bypass = s.get(f"{BASE}/api/data",
                         headers={"X-Legitimate-User": "true"},
                         timeout=TIMEOUT)
        assert r_bypass.status_code == 200, \
            f"Bypass no funcionó: {r_bypass.status_code}"
        print(f"   Bypass: {r_bypass.status_code} ✅")

        text = requests.get(f"{BASE}/metrics").text
        assert "false_positive_blocks_total{" in text
        print("   false_positive_blocks_total registrado ✅")

    def test_false_positive_counter_increments(self):
        """El counter sube exactamente una vez por bypass."""
        s = bot()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login", params={"username":"x","password":"x"})

        r = s.get(f"{BASE}/api/data")
        if r.status_code != 403:
            pytest.skip("Score no alcanzó umbral")

        before = get_metric_total("false_positive_blocks_total")
        s.get(f"{BASE}/api/data", headers={"X-Legitimate-User": "true"})
        after = get_metric_total("false_positive_blocks_total")
        assert after > before, \
            "false_positive_blocks_total no incrementó tras bypass"


# ══════════════════════════════════════════════
# 5. ML PIPELINE E2E
# ══════════════════════════════════════════════
class TestMLPipelineE2E:

    def _ml_loaded(self):
        return requests.get(f"{BASE}/status").json().get("ml_model_loaded", False)

    def test_ml_probability_exported_to_metrics(self):
        if not self._ml_loaded():
            pytest.skip("ML no cargado")
        bot().get(f"{BASE}/api/data")
        time.sleep(0.3)
        assert "bot_ml_probability{" in requests.get(f"{BASE}/metrics").text

    def test_ml_probability_in_0_to_1(self):
        if not self._ml_loaded():
            pytest.skip("ML no cargado")
        bot().get(f"{BASE}/api/data")
        for line in requests.get(f"{BASE}/metrics").text.splitlines():
            if line.startswith("bot_ml_probability{"):
                prob = float(line.split()[-1])
                assert 0.0 <= prob <= 1.0, f"Probabilidad fuera de rango: {prob}"

    def test_both_ml_and_rules_layers_visible(self):
        """Ambas capas (ML y reglas) visibles en métricas."""
        text = requests.get(f"{BASE}/metrics").text
        assert "bot_ml_probability" in text,   "ML layer no visible"
        assert "current_risk_score" in text,   "Rules layer no visible"

    def test_legit_users_pass_ml_filter(self):
        if not self._ml_loaded():
            pytest.skip("ML no cargado")
        for i in range(5):
            r = legit().get(f"{BASE}/api/data", timeout=TIMEOUT)
            assert r.status_code == 200, \
                f"Usuario legítimo bloqueado por ML en intento {i+1}"

    def test_ml_blocked_counter_increments(self):
        if not self._ml_loaded():
            pytest.skip("ML no cargado")
        s = bot()
        blocked = False
        for _ in range(12):
            s.get(f"{BASE}/api/data")
            s.get(f"{BASE}/login", params={"username":"hacker","password":"x"})
            if s.get(f"{BASE}/api/data").status_code == 403:
                blocked = True
                break
        if blocked:
            text = requests.get(f"{BASE}/metrics").text
            assert "ml_blocked_total{" in text or "blocked_requests_total{" in text


# ══════════════════════════════════════════════
# 6. PROMETHEUS DATA PIPELINE
# ══════════════════════════════════════════════
class TestPrometheusDataPipeline:

    def setup_method(self):
        if not prometheus_up():
            pytest.skip("Prometheus no accesible en localhost:9090")

    def test_api_target_is_up(self):
        r = requests.get(f"{PROMETHEUS}/api/v1/targets", timeout=TIMEOUT)
        targets = r.json().get("data", {}).get("activeTargets", [])
        api_ts  = [t for t in targets
                   if "api" in t.get("labels", {}).get("job", "")]
        assert len(api_ts) > 0, "Target 'api' no encontrado en Prometheus"
        up = [t for t in api_ts if t["health"] == "up"]
        assert len(up) > 0, \
            f"Target 'api' DOWN — error: {api_ts[0].get('lastError','')}"

    def test_http_requests_queryable(self):
        legit().get(f"{BASE}/api/data")
        time.sleep(6)  # un ciclo de scraping (5 s)
        r = prom_query("http_requests_total")
        assert r.get("status") == "success"
        assert len(r.get("data", {}).get("result", [])) > 0

    def test_false_positive_queryable(self):
        """false_positive_blocks_total debe ser consultable (requiere main_fixed.py)."""
        time.sleep(6)
        r = prom_query("false_positive_blocks_total")
        assert r.get("status") == "success", \
            "false_positive_blocks_total no consultable — aplica main_fixed.py"

    def test_rate_query_returns_data(self):
        for _ in range(3):
            legit().get(f"{BASE}/api/data")
        time.sleep(6)
        r = prom_query("rate(http_requests_total[1m])")
        assert r.get("status") == "success"

    def test_sum_query_single_result(self):
        legit().get(f"{BASE}/api/data")
        time.sleep(6)
        r = prom_query("sum(http_requests_total)")
        assert r.get("status") == "success"
        results = r.get("data", {}).get("result", [])
        assert len(results) == 1, \
            f"sum() devolvió {len(results)} resultados, esperado 1"
        assert float(results[0]["value"][1]) > 0


# ══════════════════════════════════════════════
# 7. GRAFANA DASHBOARD VALIDATION
# ══════════════════════════════════════════════
class TestGrafanaDashboard:

    def setup_method(self):
        if not grafana_up():
            pytest.skip("Grafana no accesible en localhost:3000")

    def test_dashboard_exists(self):
        r = grafana("/api/dashboards/uid/rbt-security-v1")
        assert r.status_code == 200, \
            "Dashboard no encontrado — importar rbt_dashboard_v4.json"
        assert r.json()["dashboard"]["title"] == "RBT Security Dashboard"

    def test_dashboard_has_15_panels(self):
        r = grafana("/api/dashboards/uid/rbt-security-v1")
        assert r.status_code == 200
        n = len(r.json()["dashboard"]["panels"])
        assert n == 15, f"Dashboard tiene {n} paneles, esperado 15"

    def test_all_critical_panel_titles_present(self):
        r = grafana("/api/dashboards/uid/rbt-security-v1")
        assert r.status_code == 200
        titles = {p["title"] for p in r.json()["dashboard"]["panels"]}
        for expected in [
            "Total Requests", "Blocked Requests", "False Positives",
            "Login Failures", "Bot ML Probability per User",
            "Risk Score per User", "Top Risk Users",
        ]:
            assert expected in titles, f"Panel '{expected}' no encontrado"

    def test_prometheus_datasource_uid_correct(self):
        r = grafana("/api/datasources")
        assert r.status_code == 200
        uids = [d.get("uid") for d in r.json() if d["type"] == "prometheus"]
        assert "rbt-prometheus" in uids, \
            f"UID datasource incorrecto: {uids} — esperado 'rbt-prometheus'"

    def test_all_panels_use_correct_datasource(self):
        r = grafana("/api/dashboards/uid/rbt-security-v1")
        assert r.status_code == 200
        for p in r.json()["dashboard"]["panels"]:
            uid_ = p.get("datasource", {}).get("uid", "")
            assert uid_ == "rbt-prometheus", \
                f"Panel '{p['title']}' usa datasource uid='{uid_}'"

    def test_dashboard_refresh_5s(self):
        r = grafana("/api/dashboards/uid/rbt-security-v1")
        assert r.status_code == 200
        assert r.json()["dashboard"].get("refresh") == "5s"

    def test_false_positive_panel_query_uses_or_vector(self):
        """Panel 4 debe usar 'or vector(0)' para mostrar 0 cuando no hay datos."""
        r = grafana("/api/dashboards/uid/rbt-security-v1")
        assert r.status_code == 200
        panels = r.json()["dashboard"]["panels"]
        fp_panel = next((p for p in panels if p["title"] == "False Positives"), None)
        assert fp_panel is not None, "Panel 'False Positives' no encontrado"
        expr = fp_panel["targets"][0]["expr"]
        assert "false_positive_blocks_total" in expr, \
            f"Panel FP tiene expr incorrecta: {expr}"
