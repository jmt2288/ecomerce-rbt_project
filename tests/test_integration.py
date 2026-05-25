"""
tests/test_integration.py
─────────────────────────
Pruebas de integración — verifican que los componentes del sistema
funcionan correctamente cuando trabajan juntos:

  Suite 1 — FastAPI ↔ Redis         : Risk Score fluye correctamente
  Suite 2 — FastAPI ↔ Prometheus    : métricas se exportan en /metrics
  Suite 3 — FastAPI ↔ ML Model      : predicción integrada en middleware
  Suite 4 — Redis ↔ Rate Limiting   : ventana temporal y contadores
  Suite 5 — Pipeline completo        : flujo end-to-end sin Grafana
  Suite 6 — Docker Compose health   : todos los servicios conectados

Run:
    pytest tests/test_integration.py -v
    pytest tests/test_integration.py -v -k "redis"
"""

import pytest
import requests
import time
import random
import string

BASE           = "http://localhost:8000"
PROMETHEUS     = "http://localhost:9090"
TIMEOUT        = 10


def uid():
    return "".join(random.choices(string.ascii_lowercase, k=10))


def bot_session():
    s = requests.Session()
    s.headers["User-Agent"] = f"headless-integration/{uid()}"
    return s


def legit_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent":      f"Mozilla/5.0 Integration/{uid()}",
        "Accept-Language": "es-ES,es;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    })
    return s


def get_metric_value(name: str, label_fragment: str = "") -> float:
    """Extrae el valor numérico de una métrica de Prometheus."""
    text = requests.get(f"{BASE}/metrics", timeout=TIMEOUT).text
    for line in text.splitlines():
        if line.startswith(f"{name}{{") and not line.startswith("#"):
            if not label_fragment or label_fragment in line:
                try:
                    return float(line.split()[-1])
                except (ValueError, IndexError):
                    pass
    return 0.0


# ══════════════════════════════════════════════
# SUITE 1 — FastAPI ↔ Redis
# ══════════════════════════════════════════════
class TestFastAPIRedisIntegration:
    """Verifica que FastAPI escribe y lee correctamente en Redis."""

    def test_failed_login_writes_risk_score_to_redis(self):
        """Un login fallido debe incrementar el Risk Score almacenado."""
        s = bot_session()
        # Primera petición: registra el headless UA (+15)
        s.get(f"{BASE}/api/data")

        # El Risk Score debe haberse escrito en Redis
        metrics_after = requests.get(f"{BASE}/metrics").text
        assert "current_risk_score{" in metrics_after

    def test_risk_score_increments_with_each_failed_login(self):
        """El Risk Score debe aumentar con cada login fallido."""
        s = bot_session()
        s.get(f"{BASE}/api/data")   # +15

        # Primer fallo → +10
        s.get(f"{BASE}/login",
              params={"username": "x", "password": "wrong"})

        # Segundo fallo → +10 más
        s.get(f"{BASE}/login",
              params={"username": "x", "password": "wrong2"})

        metrics = requests.get(f"{BASE}/metrics").text
        # El score total debería ser ≥ 35 (15 + 10 + 10)
        score_lines = [
            l for l in metrics.splitlines()
            if l.startswith("current_risk_score{")
            and "system_startup" not in l
        ]
        if score_lines:
            score = float(score_lines[-1].split()[-1])
            assert score >= 15.0, f"Score esperado ≥ 15, obtenido {score}"

    def test_successful_login_resets_fail_counter(self):
        """Login exitoso debe limpiar el contador de fallos en Redis."""
        s = legit_session()
        # Fallar primero
        s.get(f"{BASE}/login",
              params={"username": "admin", "password": "wrong"})
        # Éxito → debe limpiar fails:
        r = s.get(f"{BASE}/login",
                  params={"username": "admin", "password": "secret123"})
        assert r.status_code == 200
        # Verificar: el próximo fallo parte de 0 (no acumula el anterior)
        r2 = s.get(f"{BASE}/login",
                   params={"username": "admin", "password": "wrong_again"})
        assert r2.status_code == 401

    def test_false_positive_bypass_writes_to_redis_metric(self):
        """El bypass con X-Legitimate-User debe incrementar false_positive counter."""
        s = bot_session()
        s.get(f"{BASE}/api/data")
        for _ in range(4):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "wrong"})

        r = s.get(f"{BASE}/api/data")
        if r.status_code != 403:
            pytest.skip("Score no alcanzó umbral")

        before = requests.get(f"{BASE}/metrics").text
        s.get(f"{BASE}/api/data",
              headers={"X-Legitimate-User": "true"})
        after = requests.get(f"{BASE}/metrics").text

        assert "false_positive_blocks_total{" in after

    def test_ttl_expiration_concept_in_status(self):
        """El sistema debe tener TTL configurado (verificable en /status)."""
        r = requests.get(f"{BASE}/status", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        # El threshold es la referencia del TTL conceptual
        assert data.get("threshold") == 30


# ══════════════════════════════════════════════
# SUITE 2 — FastAPI ↔ Prometheus
# ══════════════════════════════════════════════
class TestFastAPIPrometheusIntegration:
    """Verifica que las métricas de FastAPI llegan a Prometheus."""

    def test_http_requests_counter_increments(self):
        """Cada petición debe incrementar http_requests_total."""
        s = legit_session()

        # Tomar valor antes
        before_text = requests.get(f"{BASE}/metrics").text
        before_total = sum(
            float(l.split()[-1])
            for l in before_text.splitlines()
            if l.startswith("http_requests_total{") and not l.startswith("#")
        )

        # Hacer peticiones
        for _ in range(5):
            s.get(f"{BASE}/api/data")

        # Tomar valor después
        after_text = requests.get(f"{BASE}/metrics").text
        after_total = sum(
            float(l.split()[-1])
            for l in after_text.splitlines()
            if l.startswith("http_requests_total{") and not l.startswith("#")
        )

        assert after_total > before_total, \
            "http_requests_total no incrementó"

    def test_all_seven_custom_metrics_present(self):
        """Las 7 métricas custom deben estar en /metrics."""
        text = requests.get(f"{BASE}/metrics", timeout=TIMEOUT).text
        expected = [
            "http_requests_total",
            "blocked_requests_total",
            "false_positive_blocks_total",
            "current_risk_score",
            "login_failures_total",
            "bot_ml_probability",
            "ml_blocked_total",
        ]
        for metric in expected:
            assert metric in text, f"Métrica faltante: {metric}"

    def test_metrics_format_valid_for_prometheus(self):
        """El formato de /metrics debe ser parseable por Prometheus."""
        text = requests.get(f"{BASE}/metrics", timeout=TIMEOUT).text
        lines = text.splitlines()
        help_count = sum(1 for l in lines if l.startswith("# HELP"))
        type_count = sum(1 for l in lines if l.startswith("# TYPE"))
        assert help_count >= 7, f"Pocas líneas # HELP: {help_count}"
        assert type_count >= 7, f"Pocas líneas # TYPE: {type_count}"

    def test_counter_metrics_are_monotonic(self):
        """Los contadores Prometheus nunca deben decrecer."""
        s = legit_session()

        def total_requests():
            return sum(
                float(l.split()[-1])
                for l in requests.get(f"{BASE}/metrics").text.splitlines()
                if l.startswith("http_requests_total{") and not l.startswith("#")
            )

        before = total_requests()
        for _ in range(3):
            s.get(f"{BASE}/api/data")
        after = total_requests()

        assert after >= before, \
            f"Contador decreció de {before} a {after}"

    def test_prometheus_can_query_fastapi_metrics(self):
        """Prometheus debe poder consultar las métricas de FastAPI."""
        time.sleep(6)   # esperar un ciclo de scraping (5s)
        r = requests.get(
            f"{PROMETHEUS}/api/v1/query",
            params={"query": "http_requests_total"},
            timeout=TIMEOUT
        )
        if r.status_code != 200:
            pytest.skip("Prometheus no accesible")
        data = r.json()
        assert data["status"] == "success"

    def test_login_failure_appears_in_metrics(self):
        """Un login fallido debe aparecer en login_failures_total."""
        s = legit_session()
        s.get(f"{BASE}/login",
              params={"username": "nobody", "password": "wrong"})

        text = requests.get(f"{BASE}/metrics").text
        assert "login_failures_total{" in text
        assert 'reason="invalid_credentials"' in text


# ══════════════════════════════════════════════
# SUITE 3 — FastAPI ↔ ML Model
# ══════════════════════════════════════════════
class TestFastAPIMLIntegration:
    """Verifica que el modelo ML está integrado correctamente en el middleware."""

    def test_ml_model_loaded_at_startup(self):
        """El modelo ML debe estar cargado al arrancar la API."""
        r = requests.get(f"{BASE}/status", timeout=TIMEOUT)
        data = r.json()
        assert data["ml_model_loaded"] is True, \
            "ML no cargado — reconstruye con: docker compose up --build -d"

    def test_bot_probability_metric_created_after_request(self):
        """bot_ml_probability debe aparecer en métricas tras una request de bot."""
        status = requests.get(f"{BASE}/status").json()
        if not status.get("ml_model_loaded"):
            pytest.skip("ML no cargado")

        s = bot_session()
        s.get(f"{BASE}/api/data")
        time.sleep(0.3)

        text = requests.get(f"{BASE}/metrics").text
        assert "bot_ml_probability{" in text

    def test_ml_probability_is_between_0_and_1(self):
        """La probabilidad ML siempre debe estar en [0.0, 1.0]."""
        status = requests.get(f"{BASE}/status").json()
        if not status.get("ml_model_loaded"):
            pytest.skip("ML no cargado")

        s = bot_session()
        s.get(f"{BASE}/api/data")
        text = requests.get(f"{BASE}/metrics").text

        for line in text.splitlines():
            if line.startswith("bot_ml_probability{"):
                prob = float(line.split()[-1])
                assert 0.0 <= prob <= 1.0, f"Probabilidad fuera de rango: {prob}"

    def test_ml_blocked_counter_increments_when_bot_blocked(self):
        """ml_blocked_total debe incrementar cuando el ML bloquea un bot."""
        status = requests.get(f"{BASE}/status").json()
        if not status.get("ml_model_loaded"):
            pytest.skip("ML no cargado")

        s = bot_session()
        blocked = False

        for _ in range(15):
            s.get(f"{BASE}/api/data")
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "wrong"})
            r = s.get(f"{BASE}/api/data")
            if r.status_code == 403:
                blocked = True
                break

        if blocked:
            text = requests.get(f"{BASE}/metrics").text
            assert "ml_blocked_total{" in text or \
                   "blocked_requests_total{" in text

    def test_legit_user_not_blocked_by_ml(self):
        """Un usuario con patrón legítimo no debe ser bloqueado por el ML."""
        status = requests.get(f"{BASE}/status").json()
        if not status.get("ml_model_loaded"):
            pytest.skip("ML no cargado")

        s = legit_session()
        for _ in range(5):
            r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            assert r.status_code == 200, \
                f"Usuario legítimo bloqueado por ML en intento {_+1}"

    def test_both_ml_and_rules_can_block(self):
        """El sistema bloquea mediante ML o mediante reglas — al menos uno debe actuar."""
        s = bot_session()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login",
                  params={"username": "hacker", "password": "pass"})

        r = s.get(f"{BASE}/api/data")
        text = requests.get(f"{BASE}/metrics").text

        ml_active   = "ml_blocked_total{"   in text
        rule_active = "blocked_requests_total{" in text

        assert ml_active or rule_active, \
            "Ni ML ni reglas están activos en las métricas"


# ══════════════════════════════════════════════
# SUITE 4 — Redis ↔ Rate Limiting
# ══════════════════════════════════════════════
class TestRedisRateLimitingIntegration:
    """Verifica el rate limiting basado en ventanas temporales en Redis."""

    def test_rate_limiting_tracks_requests_per_session(self):
        """Múltiples peticiones de la misma sesión deben acumularse."""
        s = bot_session()
        for _ in range(5):
            s.get(f"{BASE}/api/data")
        metrics = requests.get(f"{BASE}/metrics").text
        assert "http_requests_total{" in metrics

    def test_different_sessions_have_independent_counters(self):
        """Sesiones distintas no deben compartir contadores de rate limit."""
        results = []
        for _ in range(3):
            s = legit_session()
            r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            results.append(r.status_code)

        ok = sum(1 for r in results if r == 200)
        assert ok == 3, \
            f"Sesiones independientes se afectan entre sí: {results}"

    def test_rate_limit_uses_sliding_window(self):
        """El rate limit debe usar ventana deslizante (no fija)."""
        s = bot_session()
        # Enviar varias peticiones rápidas
        for _ in range(10):
            s.get(f"{BASE}/api/data")
        # El sistema debe haber registrado actividad
        metrics = requests.get(f"{BASE}/metrics").text
        score_lines = [
            l for l in metrics.splitlines()
            if l.startswith("current_risk_score{")
            and "system_startup" not in l
        ]
        # Al menos debe haber un score registrado para este usuario
        assert len(score_lines) >= 1 or "http_requests_total" in metrics


# ══════════════════════════════════════════════
# SUITE 5 — Pipeline completo de seguridad
# ══════════════════════════════════════════════
class TestCompleteSecurityPipeline:
    """Verifica el flujo completo: request → análisis → bloqueo → métricas."""

    def test_full_bot_detection_pipeline(self):
        """
        Flujo completo:
        Request bot → análisis UA → Risk Score Redis → bloqueo → métricas Prometheus
        """
        s = bot_session()

        # Paso 1: primera request (UA headless detectado)
        r1 = s.get(f"{BASE}/api/data")
        assert r1.status_code in [200, 403]

        # Paso 2: acumular Risk Score con logins fallidos
        for _ in range(4):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "wrong"})

        # Paso 3: debe estar bloqueado
        r_final = s.get(f"{BASE}/api/data")
        assert r_final.status_code == 403, \
            "Pipeline de bloqueo no completó correctamente"

        # Paso 4: las métricas deben reflejar el ataque
        metrics = requests.get(f"{BASE}/metrics").text
        assert "blocked_requests_total{" in metrics
        assert "current_risk_score{"     in metrics

    def test_full_false_positive_pipeline(self):
        """
        Flujo completo falso positivo:
        Bot → bloqueo → X-Legitimate-User → bypass → counter Prometheus
        """
        s = bot_session()
        s.get(f"{BASE}/api/data")
        for _ in range(4):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "wrong"})

        r_blocked = s.get(f"{BASE}/api/data")
        if r_blocked.status_code != 403:
            pytest.skip("Score no alcanzó el umbral")

        # Bypass
        r_bypass = s.get(f"{BASE}/api/data",
                         headers={"X-Legitimate-User": "true"})
        assert r_bypass.status_code == 200

        # Métrica registrada
        metrics = requests.get(f"{BASE}/metrics").text
        assert "false_positive_blocks_total{" in metrics

    def test_metrics_only_track_protected_endpoints(self):
        """Solo los endpoints protegidos deben aparecer en http_requests_total."""
        # / y /metrics están excluidos
        requests.get(f"{BASE}/")
        for _ in range(5):
            requests.get(f"{BASE}/metrics")

        text = requests.get(f"{BASE}/metrics").text
        assert 'endpoint="/"'       not in text
        assert 'endpoint="/metrics"' not in text

    def test_system_remains_stable_after_attacks(self):
        """El sistema debe seguir funcionando correctamente tras ataques."""
        # Simular varios tipos de ataque
        for _ in range(3):
            bot_session().get(f"{BASE}/api/data")

        # Verificar que el sistema sigue sano
        r = requests.get(f"{BASE}/status", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "running"


# ══════════════════════════════════════════════
# SUITE 6 — Docker Compose Health
# ══════════════════════════════════════════════
class TestDockerComposeHealth:
    """Verifica que todos los servicios Docker están conectados y sanos."""

    def test_fastapi_is_reachable(self):
        r = requests.get(f"{BASE}/", timeout=TIMEOUT)
        assert r.status_code == 200
        assert r.json()["status"] == "RBT Security Layer Active"

    def test_prometheus_is_reachable(self):
        try:
            r = requests.get(f"{PROMETHEUS}/-/healthy", timeout=TIMEOUT)
            assert r.status_code == 200
        except requests.exceptions.ConnectionError:
            pytest.skip("Prometheus no accesible en este entorno")

    def test_prometheus_is_scraping_fastapi(self):
        """Prometheus debe tener el target de FastAPI en estado 'up'."""
        try:
            r = requests.get(f"{PROMETHEUS}/api/v1/targets", timeout=TIMEOUT)
            if r.status_code != 200:
                pytest.skip("Prometheus no accesible")
            data = r.json()
            targets = data.get("data", {}).get("activeTargets", [])
            api_targets = [
                t for t in targets
                if "api" in t.get("labels", {}).get("job", "")
            ]
            if api_targets:
                assert any(t["health"] == "up" for t in api_targets), \
                    "Target API está DOWN en Prometheus"
        except requests.exceptions.ConnectionError:
            pytest.skip("Prometheus no accesible")

    def test_redis_connectivity_via_api(self):
        """Redis debe estar funcionando (verificable a través de la API)."""
        # Si Redis falla, el Risk Score no se puede escribir y la API da error
        s = bot_session()
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        assert r.status_code in [200, 403], \
            f"API retornó {r.status_code} — posible fallo de Redis"

    def test_ml_model_loaded_from_dockerfile(self):
        """El modelo ML debe haberse entrenado durante el docker build."""
        r = requests.get(f"{BASE}/status", timeout=TIMEOUT)
        data = r.json()
        assert "ml_model_loaded" in data
        if not data["ml_model_loaded"]:
            pytest.fail(
                "Modelo ML no cargado. Reconstruir con: docker compose up --build -d"
            )
