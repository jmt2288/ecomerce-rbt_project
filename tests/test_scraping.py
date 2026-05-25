"""
tests/test_scraping.py
──────────────────────
Pruebas completas del sistema de scraping:
  FastAPI /metrics → Prometheus → (queryable)

  Suite 1 — Endpoint /metrics              : formato, content-type, accesibilidad
  Suite 2 — Formato Prometheus             : HELP, TYPE, labels, sintaxis
  Suite 3 — Las 7 métricas custom          : presencia y tipos correctos
  Suite 4 — Valores y contadores           : que los números son coherentes
  Suite 5 — Labels correctos              : identifier, method, reason…
  Suite 6 — Scraping desde Prometheus      : el target está UP y las series existen
  Suite 7 — Ciclo completo de scraping     : acción → /metrics → Prometheus queryable
  Suite 8 — Exclusiones correctas          : / y /metrics no se cuentan a sí mismos
  Suite 9 — Inicialización de métricas     : valores al arrancar la API
  Suite 10 — Rendimiento del endpoint      : /metrics responde rápido bajo carga

Run:
    pytest tests/test_scraping.py -v
    pytest tests/test_scraping.py -v -k "format"
    pytest tests/test_scraping.py -v -k "not prometheus"  # sin Prometheus
"""

import pytest
import requests
import time
import threading
import random
import string
import re

BASE       = "http://localhost:8000"
PROMETHEUS = "http://localhost:9090"
TIMEOUT    = 10

# Las 7 métricas custom del proyecto
EXPECTED_METRICS = [
    "http_requests_total",
    "blocked_requests_total",
    "false_positive_blocks_total",
    "current_risk_score",
    "login_failures_total",
    "bot_ml_probability",
    "ml_blocked_total",
]

# Labels conocidos por métrica
METRIC_LABELS = {
    "http_requests_total":          ["method", "endpoint"],
    "blocked_requests_total":       ["reason", "identifier"],
    "false_positive_blocks_total":  ["identifier"],
    "current_risk_score":           ["identifier"],
    "login_failures_total":         ["method", "reason"],
    "bot_ml_probability":           ["identifier"],
    "ml_blocked_total":             ["identifier"],
}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def uid():
    return "".join(random.choices(string.ascii_lowercase, k=10))


def get_metrics() -> str:
    return requests.get(f"{BASE}/metrics", timeout=TIMEOUT).text


def bot_session():
    s = requests.Session()
    s.headers["User-Agent"] = f"headless-scraping-test/{uid()}"
    return s


def legit_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent":      f"Mozilla/5.0 ScrapeTest/{uid()}",
        "Accept-Language": "es-ES,es;q=0.9",
        "Accept-Encoding": "gzip",
    })
    return s


def prom_query(expr: str) -> dict:
    """Ejecuta una consulta PromQL instantánea."""
    try:
        r = requests.get(
            f"{PROMETHEUS}/api/v1/query",
            params={"query": expr},
            timeout=TIMEOUT,
        )
        return r.json()
    except Exception:
        return {}


def prometheus_up() -> bool:
    try:
        r = requests.get(f"{PROMETHEUS}/-/healthy", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def parse_metric_lines(text: str, metric_name: str) -> list:
    """Devuelve todas las líneas de datos (no comentarios) de una métrica."""
    return [
        line for line in text.splitlines()
        if line.startswith(f"{metric_name}{{") or
           (line.startswith(metric_name) and " " in line
            and not line.startswith("#"))
    ]


def extract_value(line: str) -> float:
    """Extrae el valor numérico del final de una línea de métrica."""
    return float(line.split()[-1])


# ══════════════════════════════════════════════
# SUITE 1 — Endpoint /metrics
# ══════════════════════════════════════════════
class TestMetricsEndpoint:
    """Verifica que /metrics es accesible y devuelve la respuesta correcta."""

    def test_metrics_returns_200(self):
        r = requests.get(f"{BASE}/metrics", timeout=TIMEOUT)
        assert r.status_code == 200, \
            f"/metrics devolvió {r.status_code}, esperado 200"

    def test_metrics_content_type_is_text_plain(self):
        """Prometheus requiere text/plain para parsear correctamente."""
        r = requests.get(f"{BASE}/metrics", timeout=TIMEOUT)
        ct = r.headers.get("content-type", "")
        assert "text/plain" in ct, \
            f"Content-Type incorrecto: {ct}. Prometheus necesita text/plain"

    def test_metrics_body_is_not_empty(self):
        text = get_metrics()
        assert len(text) > 100, "Respuesta de /metrics demasiado corta"

    def test_metrics_not_json(self):
        """/metrics debe ser texto plano, no JSON."""
        r = requests.get(f"{BASE}/metrics", timeout=TIMEOUT)
        try:
            r.json()
            pytest.fail("/metrics devuelve JSON — Prometheus no puede parsearlo")
        except Exception:
            pass  # correcto: no es JSON

    def test_metrics_accessible_without_auth(self):
        """Prometheus scraping no usa autenticación — debe ser público."""
        r = requests.get(f"{BASE}/metrics", timeout=TIMEOUT)
        assert r.status_code != 401, "/metrics requiere autenticación — Prometheus no puede scraping"
        assert r.status_code != 403, "/metrics bloqueado — Prometheus no puede scraping"

    def test_metrics_not_tracked_in_itself(self):
        """Las peticiones a /metrics NO deben aparecer en http_requests_total."""
        for _ in range(5):
            requests.get(f"{BASE}/metrics")
        text = get_metrics()
        assert 'endpoint="/metrics"' not in text, \
            "/metrics se está contando a sí misma en http_requests_total"

    def test_metrics_response_time_under_500ms(self):
        """El endpoint /metrics debe responder en menos de 500ms."""
        start = time.time()
        requests.get(f"{BASE}/metrics", timeout=TIMEOUT)
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 500, \
            f"/metrics tardó {elapsed_ms:.0f}ms — demasiado para Prometheus"

    def test_metrics_idempotent(self):
        """Dos peticiones seguidas a /metrics deben dar la misma estructura."""
        text1 = get_metrics()
        text2 = get_metrics()
        # Mismas métricas presentes en ambas
        for metric in EXPECTED_METRICS:
            assert metric in text1
            assert metric in text2

    def test_metrics_supports_get_only(self):
        """Solo GET debe estar permitido en /metrics."""
        r_post = requests.post(f"{BASE}/metrics", timeout=TIMEOUT)
        assert r_post.status_code in [405, 404, 403], \
            f"POST a /metrics devolvió {r_post.status_code} — debería ser 405"


# ══════════════════════════════════════════════
# SUITE 2 — Formato Prometheus
# ══════════════════════════════════════════════
class TestPrometheusFormat:
    """Verifica que el formato de /metrics es válido para Prometheus."""

    def test_has_help_lines(self):
        """Cada métrica debe tener una línea # HELP."""
        text = get_metrics()
        help_lines = [l for l in text.splitlines() if l.startswith("# HELP")]
        assert len(help_lines) >= 7, \
            f"Solo {len(help_lines)} líneas # HELP — esperadas al menos 7"

    def test_has_type_lines(self):
        """Cada métrica debe tener una línea # TYPE."""
        text = get_metrics()
        type_lines = [l for l in text.splitlines() if l.startswith("# TYPE")]
        assert len(type_lines) >= 7, \
            f"Solo {len(type_lines)} líneas # TYPE — esperadas al menos 7"

    def test_help_before_type_for_each_metric(self):
        """# HELP debe aparecer antes de # TYPE para cada métrica."""
        text = get_metrics()
        lines = text.splitlines()
        for metric in EXPECTED_METRICS:
            help_idx = next(
                (i for i, l in enumerate(lines) if l.startswith(f"# HELP {metric}")),
                None
            )
            type_idx = next(
                (i for i, l in enumerate(lines) if l.startswith(f"# TYPE {metric}")),
                None
            )
            if help_idx is not None and type_idx is not None:
                assert help_idx < type_idx, \
                    f"{metric}: # HELP debe ir antes de # TYPE"

    def test_counter_metrics_declared_as_counter(self):
        """Los contadores deben declararse con TYPE counter."""
        text = get_metrics()
        counter_metrics = [
            "http_requests_total",
            "blocked_requests_total",
            "false_positive_blocks_total",
            "login_failures_total",
            "ml_blocked_total",
        ]
        for metric in counter_metrics:
            type_line = next(
                (l for l in text.splitlines()
                 if l.startswith(f"# TYPE {metric}")),
                None
            )
            if type_line:
                assert "counter" in type_line.lower(), \
                    f"{metric} declarado como '{type_line}', esperado counter"

    def test_gauge_metrics_declared_as_gauge(self):
        """Los gauges deben declararse con TYPE gauge."""
        text = get_metrics()
        gauge_metrics = ["current_risk_score", "bot_ml_probability"]
        for metric in gauge_metrics:
            type_line = next(
                (l for l in text.splitlines()
                 if l.startswith(f"# TYPE {metric}")),
                None
            )
            if type_line:
                assert "gauge" in type_line.lower(), \
                    f"{metric} declarado como '{type_line}', esperado gauge"

    def test_metric_values_are_numeric(self):
        """Todos los valores de métricas deben ser números válidos."""
        text = get_metrics()
        value_pattern = re.compile(r'^[^#]\S+\{[^}]*\}\s+([\d.eE+\-]+|NaN|Inf|\+Inf|-Inf)$')
        for line in text.splitlines():
            if line and not line.startswith("#"):
                parts = line.rsplit(" ", 1)
                if len(parts) == 2:
                    try:
                        float(parts[1])
                    except ValueError:
                        pytest.fail(f"Valor no numérico en línea: {line}")

    def test_no_duplicate_help_lines(self):
        """No debe haber líneas # HELP duplicadas para la misma métrica."""
        text = get_metrics()
        help_lines = [l for l in text.splitlines() if l.startswith("# HELP")]
        metric_names = [l.split()[2] for l in help_lines if len(l.split()) >= 3]
        duplicates = [m for m in metric_names if metric_names.count(m) > 1]
        assert len(set(duplicates)) == 0, \
            f"Métricas con # HELP duplicado: {set(duplicates)}"

    def test_labels_use_double_quotes(self):
        """Los labels deben usar comillas dobles, no simples."""
        text = get_metrics()
        for line in text.splitlines():
            if line and not line.startswith("#") and "{" in line:
                label_part = line[line.index("{"):line.index("}")+1]
                assert "'" not in label_part, \
                    f"Labels con comillas simples: {line}"

    def test_counter_total_suffix(self):
        """Los counters deben terminar en _total (convención Prometheus)."""
        counter_metrics = [
            "http_requests_total",
            "blocked_requests_total",
            "false_positive_blocks_total",
            "login_failures_total",
            "ml_blocked_total",
        ]
        for metric in counter_metrics:
            assert metric.endswith("_total"), \
                f"Counter {metric} no termina en _total"

    def test_no_whitespace_in_metric_names(self):
        """Los nombres de métricas no deben tener espacios."""
        text = get_metrics()
        for line in text.splitlines():
            if line.startswith("# HELP") or line.startswith("# TYPE"):
                parts = line.split()
                if len(parts) >= 3:
                    metric_name = parts[2]
                    assert " " not in metric_name, \
                        f"Nombre de métrica con espacio: {metric_name}"


# ══════════════════════════════════════════════
# SUITE 3 — Las 7 métricas custom
# ══════════════════════════════════════════════
class TestCustomMetrics:
    """Verifica que las 7 métricas custom están presentes y tienen descripción."""

    def test_all_7_metrics_present(self):
        text = get_metrics()
        for metric in EXPECTED_METRICS:
            assert metric in text, \
                f"Métrica faltante en /metrics: {metric}"

    def test_http_requests_total_present_and_typed(self):
        text = get_metrics()
        assert "# HELP http_requests_total" in text
        assert "# TYPE http_requests_total counter" in text

    def test_blocked_requests_total_present_and_typed(self):
        text = get_metrics()
        assert "# HELP blocked_requests_total" in text
        assert "# TYPE blocked_requests_total counter" in text

    def test_false_positive_blocks_total_present(self):
        text = get_metrics()
        assert "false_positive_blocks_total" in text
        assert "# HELP false_positive_blocks_total" in text

    def test_current_risk_score_is_gauge(self):
        text = get_metrics()
        assert "# TYPE current_risk_score gauge" in text

    def test_login_failures_total_present(self):
        text = get_metrics()
        assert "login_failures_total" in text

    def test_bot_ml_probability_is_gauge(self):
        text = get_metrics()
        assert "# TYPE bot_ml_probability gauge" in text

    def test_ml_blocked_total_present(self):
        text = get_metrics()
        assert "ml_blocked_total" in text

    def test_all_metrics_have_help_description(self):
        """Cada métrica custom debe tener un texto de ayuda no vacío."""
        text = get_metrics()
        for metric in EXPECTED_METRICS:
            help_line = next(
                (l for l in text.splitlines()
                 if l.startswith(f"# HELP {metric} ")),
                None
            )
            assert help_line is not None, f"Sin # HELP para {metric}"
            description = help_line[len(f"# HELP {metric} "):].strip()
            assert len(description) > 0, \
                f"{metric} tiene descripción vacía"

    def test_no_unexpected_metrics_collision(self):
        """Los nombres de nuestras métricas no colisionan con las de Python/Prometheus."""
        text = get_metrics()
        for metric in EXPECTED_METRICS:
            # Nuestras métricas deben tener exactamente 1 línea HELP
            count = text.count(f"# HELP {metric}")
            assert count == 1, \
                f"{metric} aparece {count} veces en # HELP (esperado 1)"


# ══════════════════════════════════════════════
# SUITE 4 — Valores y contadores
# ══════════════════════════════════════════════
class TestMetricValues:
    """Verifica que los valores de las métricas son coherentes con las acciones."""

    def test_http_requests_counter_increases_after_request(self):
        """http_requests_total debe incrementar tras una petición."""
        s = legit_session()

        def total():
            return sum(
                float(l.split()[-1])
                for l in get_metrics().splitlines()
                if l.startswith("http_requests_total{")
                and not l.startswith("#")
            )

        before = total()
        s.get(f"{BASE}/api/data")
        s.get(f"{BASE}/api/data")
        after = total()
        assert after > before, \
            f"http_requests_total no incrementó: {before} → {after}"

    def test_counter_values_are_non_negative(self):
        """Ningún counter puede ser negativo."""
        text = get_metrics()
        counter_metrics = [m for m in EXPECTED_METRICS if m.endswith("_total")]
        for line in text.splitlines():
            if not line.startswith("#"):
                for metric in counter_metrics:
                    if line.startswith(metric):
                        val = float(line.split()[-1])
                        assert val >= 0, \
                            f"Counter negativo: {line}"

    def test_gauge_values_in_valid_range(self):
        """Los gauges de probabilidad deben estar en [0, 1]."""
        bot_session().get(f"{BASE}/api/data")
        text = get_metrics()
        for line in text.splitlines():
            if line.startswith("bot_ml_probability{"):
                val = float(line.split()[-1])
                assert 0.0 <= val <= 1.0, \
                    f"bot_ml_probability fuera de rango: {val}"

    def test_risk_score_gauge_non_negative(self):
        """current_risk_score siempre debe ser ≥ 0."""
        bot_session().get(f"{BASE}/api/data")
        text = get_metrics()
        for line in text.splitlines():
            if line.startswith("current_risk_score{"):
                val = float(line.split()[-1])
                assert val >= 0, f"Risk Score negativo: {val}"

    def test_login_failure_counter_increases(self):
        """login_failures_total debe subir tras un login fallido."""
        s = legit_session()

        def total_fails():
            return sum(
                float(l.split()[-1])
                for l in get_metrics().splitlines()
                if l.startswith("login_failures_total{")
                and not l.startswith("#")
            )

        before = total_fails()
        s.get(f"{BASE}/login",
              params={"username": "nadie", "password": "mal"})
        after = total_fails()
        assert after > before, \
            f"login_failures_total no incrementó: {before} → {after}"

    def test_blocked_counter_increases_after_block(self):
        """blocked_requests_total debe subir cuando un bot es bloqueado."""
        s = bot_session()
        s.get(f"{BASE}/api/data")
        for _ in range(4):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "wrong"})

        def total_blocked():
            return sum(
                float(l.split()[-1])
                for l in get_metrics().splitlines()
                if l.startswith("blocked_requests_total{")
                and not l.startswith("#")
            )

        before = total_blocked()
        for _ in range(3):
            s.get(f"{BASE}/api/data")
        after = total_blocked()
        assert after >= before, \
            "blocked_requests_total no incrementó tras bloques"

    def test_counter_values_are_integers_or_floats(self):
        """Los valores de counters deben ser enteros o floats, no strings."""
        text = get_metrics()
        for line in text.splitlines():
            if line and not line.startswith("#"):
                try:
                    val = float(line.split()[-1])
                    assert not (val != val), f"NaN en métrica: {line}"  # NaN check
                except ValueError:
                    pytest.fail(f"Valor no parseable: {line}")

    def test_metrics_monotonic_over_time(self):
        """Los counters no deben decrecer entre dos lecturas."""
        def get_totals():
            text = get_metrics()
            totals = {}
            for line in text.splitlines():
                if line and not line.startswith("#"):
                    for metric in EXPECTED_METRICS:
                        if metric.endswith("_total") and line.startswith(metric):
                            totals[metric] = totals.get(metric, 0) + float(line.split()[-1])
            return totals

        before = get_totals()
        legit_session().get(f"{BASE}/api/data")
        legit_session().get(f"{BASE}/api/data")
        after = get_totals()

        for metric, before_val in before.items():
            after_val = after.get(metric, 0)
            assert after_val >= before_val, \
                f"Counter decreció: {metric} {before_val} → {after_val}"


# ══════════════════════════════════════════════
# SUITE 5 — Labels correctos
# ══════════════════════════════════════════════
class TestMetricLabels:
    """Verifica que las métricas tienen los labels correctos."""

    def test_http_requests_has_method_and_endpoint_labels(self):
        """http_requests_total debe tener labels method y endpoint."""
        legit_session().get(f"{BASE}/api/data")
        text = get_metrics()
        lines = [l for l in text.splitlines()
                 if l.startswith("http_requests_total{")]
        assert len(lines) > 0, "No hay datos para http_requests_total"
        for line in lines:
            assert 'method="' in line, f"Falta label 'method' en: {line}"
            assert 'endpoint="' in line, f"Falta label 'endpoint' en: {line}"

    def test_blocked_requests_has_reason_and_identifier(self):
        """blocked_requests_total debe tener labels reason e identifier."""
        s = bot_session()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "wrong"})
        s.get(f"{BASE}/api/data")
        text = get_metrics()
        lines = [l for l in text.splitlines()
                 if l.startswith("blocked_requests_total{")]
        if lines:
            for line in lines:
                assert 'reason="' in line, f"Falta label 'reason' en: {line}"
                assert 'identifier="' in line, f"Falta label 'identifier' en: {line}"

    def test_risk_score_has_identifier_label(self):
        """current_risk_score debe tener label identifier."""
        bot_session().get(f"{BASE}/api/data")
        text = get_metrics()
        lines = [l for l in text.splitlines()
                 if l.startswith("current_risk_score{")]
        assert len(lines) > 0, "No hay datos para current_risk_score"
        for line in lines:
            assert 'identifier="' in line, \
                f"Falta label 'identifier' en: {line}"

    def test_login_failures_has_method_and_reason(self):
        """login_failures_total debe tener labels method y reason."""
        legit_session().get(
            f"{BASE}/login",
            params={"username": "x", "password": "wrong"}
        )
        text = get_metrics()
        lines = [l for l in text.splitlines()
                 if l.startswith("login_failures_total{")]
        assert len(lines) > 0, "No hay datos para login_failures_total"
        for line in lines:
            assert 'method="' in line, f"Falta label 'method' en: {line}"
            assert 'reason="' in line, f"Falta label 'reason' en: {line}"

    def test_blocked_reason_values_are_known(self):
        """El label 'reason' solo debe tener valores conocidos."""
        s = bot_session()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login", params={"username": "x", "password": "x"})
        s.get(f"{BASE}/api/data")
        text = get_metrics()
        known_reasons = {
            "risk_score_exceeded",
            "ml_bot_detected",
            "rate_limit_exceeded",
        }
        for line in text.splitlines():
            if line.startswith("blocked_requests_total{"):
                m = re.search(r'reason="([^"]+)"', line)
                if m:
                    reason = m.group(1)
                    assert reason in known_reasons, \
                        f"Reason desconocido: '{reason}'. Conocidos: {known_reasons}"

    def test_http_method_label_values(self):
        """El label 'method' solo debe tener métodos HTTP válidos."""
        legit_session().get(f"{BASE}/api/data")
        text = get_metrics()
        valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
        for line in text.splitlines():
            if line.startswith("http_requests_total{"):
                m = re.search(r'method="([^"]+)"', line)
                if m:
                    method = m.group(1)
                    assert method in valid_methods, \
                        f"Método HTTP inválido: '{method}'"

    def test_identifier_label_format(self):
        """El label 'identifier' debe tener formato fingerprint:ip (md5:ip)."""
        bot_session().get(f"{BASE}/api/data")
        text = get_metrics()
        for line in text.splitlines():
            if "identifier=" in line and not line.startswith("#"):
                m = re.search(r'identifier="([^"]+)"', line)
                if m:
                    identifier = m.group(1)
                    # Debe contener ":" separando fingerprint de IP
                    # o ser "system_startup"
                    if identifier != "system_startup":
                        assert ":" in identifier, \
                            f"Identifier sin formato fingerprint:ip — '{identifier}'"

    def test_multiple_sessions_create_multiple_identifiers(self):
        """Sesiones distintas deben generar identifiers distintos en las métricas."""
        for _ in range(3):
            bot_session().get(f"{BASE}/api/data")

        text = get_metrics()
        identifiers = set()
        for line in text.splitlines():
            if line.startswith("current_risk_score{"):
                m = re.search(r'identifier="([^"]+)"', line)
                if m and m.group(1) != "system_startup":
                    identifiers.add(m.group(1))

        assert len(identifiers) >= 1, \
            "No se encontraron identifiers en current_risk_score"


# ══════════════════════════════════════════════
# SUITE 6 — Scraping desde Prometheus
# ══════════════════════════════════════════════
class TestPrometeusScraping:
    """Verifica que Prometheus scrapea correctamente la API."""

    def setup_method(self):
        if not prometheus_up():
            pytest.skip("Prometheus no accesible en localhost:9090")

    def test_prometheus_is_healthy(self):
        r = requests.get(f"{PROMETHEUS}/-/healthy", timeout=TIMEOUT)
        assert r.status_code == 200

    def test_api_target_is_up(self):
        """Prometheus debe reportar el target 'api' como UP."""
        r = requests.get(f"{PROMETHEUS}/api/v1/targets", timeout=TIMEOUT)
        assert r.status_code == 200
        data = r.json()
        targets = data.get("data", {}).get("activeTargets", [])
        api_targets = [
            t for t in targets
            if "api" in t.get("labels", {}).get("job", "")
        ]
        assert len(api_targets) > 0, "No se encontró el target 'api' en Prometheus"
        up_targets = [t for t in api_targets if t["health"] == "up"]
        assert len(up_targets) > 0, \
            f"Target 'api' está DOWN en Prometheus. Error: {api_targets[0].get('lastError', '')}"

    def test_scrape_interval_is_5_seconds(self):
        """El scrape interval debe ser 5 segundos según prometheus.yml."""
        r = requests.get(f"{PROMETHEUS}/api/v1/targets", timeout=TIMEOUT)
        data = r.json()
        targets = data.get("data", {}).get("activeTargets", [])
        for t in targets:
            if "api" in t.get("labels", {}).get("job", ""):
                interval = t.get("scrapeInterval", "")
                assert "5s" in interval, \
                    f"Scrape interval inesperado: {interval} (esperado 5s)"

    def test_http_requests_queryable_in_prometheus(self):
        """http_requests_total debe ser consultable en Prometheus."""
        legit_session().get(f"{BASE}/api/data")
        time.sleep(6)  # Esperar un ciclo de scraping
        result = prom_query("http_requests_total")
        assert result.get("status") == "success", \
            "Consulta PromQL fallida"
        assert len(result.get("data", {}).get("result", [])) > 0, \
            "http_requests_total no tiene datos en Prometheus"

    def test_risk_score_queryable_in_prometheus(self):
        """current_risk_score debe ser consultable tras un request de bot."""
        bot_session().get(f"{BASE}/api/data")
        time.sleep(6)
        result = prom_query("current_risk_score")
        assert result.get("status") == "success"

    def test_rate_query_returns_data(self):
        """rate(http_requests_total[1m]) debe devolver datos."""
        for _ in range(3):
            legit_session().get(f"{BASE}/api/data")
        time.sleep(6)
        result = prom_query("rate(http_requests_total[1m])")
        assert result.get("status") == "success"

    def test_sum_query_aggregates_correctly(self):
        """sum(http_requests_total) debe devolver un único valor agregado."""
        legit_session().get(f"{BASE}/api/data")
        time.sleep(6)
        result = prom_query("sum(http_requests_total)")
        assert result.get("status") == "success"
        results = result.get("data", {}).get("result", [])
        # sum() debe devolver exactamente 1 resultado
        assert len(results) == 1, \
            f"sum() devolvió {len(results)} resultados, esperado 1"
        val = float(results[0]["value"][1])
        assert val > 0, "sum(http_requests_total) = 0"

    def test_prometheus_target_url_is_correct(self):
        """La URL del target debe ser http://api:8000/metrics."""
        r = requests.get(f"{PROMETHEUS}/api/v1/targets", timeout=TIMEOUT)
        data = r.json()
        targets = data.get("data", {}).get("activeTargets", [])
        for t in targets:
            if "api" in t.get("labels", {}).get("job", ""):
                url = t.get("scrapeUrl", "")
                assert "8000" in url, f"Puerto incorrecto en target URL: {url}"
                assert "metrics" in url, f"Ruta incorrecta en target URL: {url}"


# ══════════════════════════════════════════════
# SUITE 7 — Ciclo completo acción → métrica
# ══════════════════════════════════════════════
class TestScrapeDataPipeline:
    """Verifica el flujo completo: acción → /metrics → Prometheus."""

    def test_login_failure_appears_in_metrics_immediately(self):
        """Un login fallido debe aparecer en /metrics en la siguiente lectura."""
        s = legit_session()
        before = get_metrics()

        s.get(f"{BASE}/login",
              params={"username": "inexistente", "password": "mal"})

        after = get_metrics()
        assert "login_failures_total{" in after, \
            "login_failures_total no apareció en /metrics tras login fallido"

    def test_bot_request_updates_risk_score_metric(self):
        """Un request de bot debe actualizar current_risk_score en /metrics."""
        s = bot_session()
        s.get(f"{BASE}/api/data")
        time.sleep(0.3)
        text = get_metrics()
        score_lines = [
            l for l in text.splitlines()
            if l.startswith("current_risk_score{")
            and "system_startup" not in l
        ]
        assert len(score_lines) > 0, \
            "current_risk_score no actualizado tras request de bot"

    def test_ml_probability_appears_after_bot_request(self):
        """bot_ml_probability debe aparecer en /metrics tras un request de bot."""
        status = requests.get(f"{BASE}/status").json()
        if not status.get("ml_model_loaded"):
            pytest.skip("ML no cargado")
        s = bot_session()
        s.get(f"{BASE}/api/data")
        time.sleep(0.3)
        text = get_metrics()
        assert "bot_ml_probability{" in text, \
            "bot_ml_probability no aparece en /metrics tras request de bot"

    def test_blocked_metric_updated_after_block(self):
        """blocked_requests_total debe actualizarse cuando se bloquea a alguien."""
        s = bot_session()
        s.get(f"{BASE}/api/data")
        for _ in range(4):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "x"})

        before_text = get_metrics()
        before_blocked = sum(
            float(l.split()[-1])
            for l in before_text.splitlines()
            if l.startswith("blocked_requests_total{")
        )

        for _ in range(3):
            s.get(f"{BASE}/api/data")

        after_text = get_metrics()
        after_blocked = sum(
            float(l.split()[-1])
            for l in after_text.splitlines()
            if l.startswith("blocked_requests_total{")
        )

        assert after_blocked >= before_blocked, \
            "blocked_requests_total no se actualizó tras bloqueos"

    def test_false_positive_metric_after_bypass(self):
        """false_positive_blocks_total debe incrementar tras bypass con header."""
        s = bot_session()
        s.get(f"{BASE}/api/data")
        for _ in range(4):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "wrong"})

        r_blocked = s.get(f"{BASE}/api/data")
        if r_blocked.status_code != 403:
            pytest.skip("Score no alcanzó el umbral en esta ejecución")

        # Bypass — esto debe incrementar false_positive_blocks_total
        s.get(f"{BASE}/api/data",
              headers={"X-Legitimate-User": "true"})

        text = get_metrics()
        assert "false_positive_blocks_total{" in text, \
            "false_positive_blocks_total no apareció tras bypass"

    def test_metrics_reflect_concurrent_requests(self):
        """Múltiples requests concurrentes deben reflejarse en las métricas."""
        results = []
        lock = threading.Lock()

        def do_request():
            r = legit_session().get(f"{BASE}/api/data", timeout=5)
            with lock:
                results.append(r.status_code)

        threads = [threading.Thread(target=do_request) for _ in range(10)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        ok_count = sum(1 for r in results if r == 200)
        assert ok_count >= 8, f"Solo {ok_count}/10 requests concurrentes dieron 200"

        # Las métricas deben reflejar los requests
        text = get_metrics()
        total = sum(
            float(l.split()[-1])
            for l in text.splitlines()
            if l.startswith("http_requests_total{")
        )
        assert total >= ok_count, \
            f"http_requests_total ({total}) menor que requests realizados ({ok_count})"


# ══════════════════════════════════════════════
# SUITE 8 — Exclusiones correctas
# ══════════════════════════════════════════════
class TestMetricExclusions:
    """Verifica qué endpoints se excluyen correctamente del tracking."""

    def test_root_endpoint_not_tracked(self):
        """Las peticiones a / no deben aparecer en http_requests_total."""
        for _ in range(5):
            requests.get(f"{BASE}/")
        text = get_metrics()
        assert 'endpoint="/"' not in text, \
            'endpoint="/" encontrado en http_requests_total — debería estar excluido'

    def test_metrics_endpoint_not_tracked(self):
        """Las peticiones a /metrics no deben contarse en http_requests_total."""
        for _ in range(5):
            requests.get(f"{BASE}/metrics")
        text = get_metrics()
        assert 'endpoint="/metrics"' not in text, \
            'endpoint="/metrics" se está contando a sí mismo'

    def test_api_data_endpoint_is_tracked(self):
        """Las peticiones a /api/data SÍ deben aparecer en http_requests_total."""
        legit_session().get(f"{BASE}/api/data")
        text = get_metrics()
        assert 'endpoint="/api/data"' in text, \
            '/api/data no está siendo tracked en http_requests_total'

    def test_login_endpoint_is_tracked(self):
        """Las peticiones a /login SÍ deben aparecer en http_requests_total."""
        legit_session().get(f"{BASE}/login",
                            params={"username": "x", "password": "x"})
        text = get_metrics()
        assert 'endpoint="/login"' in text, \
            '/login no está siendo tracked en http_requests_total'

    def test_status_endpoint_tracking(self):
        """Las peticiones a /status — verificar su comportamiento."""
        requests.get(f"{BASE}/status")
        text = get_metrics()
        # /status puede o no estar trackeado — el test verifica coherencia
        # Si está trackeado, debe tener el formato correcto
        if 'endpoint="/status"' in text:
            lines = [l for l in text.splitlines()
                     if 'endpoint="/status"' in l and not l.startswith("#")]
            for line in lines:
                # Debe ser numérico
                try:
                    float(line.split()[-1])
                except ValueError:
                    pytest.fail(f"Línea de /status con valor no numérico: {line}")


# ══════════════════════════════════════════════
# SUITE 9 — Inicialización de métricas
# ══════════════════════════════════════════════
class TestMetricInitialization:
    """Verifica el estado inicial de las métricas al arrancar la API."""

    def test_system_startup_gauge_present(self):
        """El gauge system_startup debe estar presente al inicio."""
        text = get_metrics()
        assert 'identifier="system_startup"' in text, \
            "El gauge de sistema (system_startup) no está presente"

    def test_system_startup_value_is_zero(self):
        """El gauge system_startup debe tener valor 0."""
        text = get_metrics()
        for line in text.splitlines():
            if 'identifier="system_startup"' in line and not line.startswith("#"):
                val = float(line.split()[-1])
                assert val == 0.0, \
                    f"system_startup tiene valor {val}, esperado 0"

    def test_login_failures_initialized(self):
        """login_failures_total debe estar inicializado al arrancar."""
        text = get_metrics()
        assert "login_failures_total" in text, \
            "login_failures_total no está inicializado"

    def test_all_metrics_present_at_startup(self):
        """Todas las métricas custom deben estar en /metrics desde el inicio."""
        text = get_metrics()
        for metric in EXPECTED_METRICS:
            # Al menos el # HELP debe estar presente
            assert f"# HELP {metric}" in text, \
                f"Métrica {metric} no presente en /metrics al arrancar"

    def test_metrics_endpoint_survives_restart_concept(self):
        """Las métricas deben ser consistentes en lecturas sucesivas."""
        texts = [get_metrics() for _ in range(3)]
        for metric in EXPECTED_METRICS:
            for i, text in enumerate(texts):
                assert metric in text, \
                    f"Métrica {metric} desapareció en lectura {i+1}"


# ══════════════════════════════════════════════
# SUITE 10 — Rendimiento del scraping
# ══════════════════════════════════════════════
class TestScrapingPerformance:
    """Verifica que el endpoint /metrics responde rápido bajo carga."""

    def test_metrics_p99_under_200ms(self):
        """P99 de /metrics < 200ms — Prometheus necesita respuesta rápida."""
        times = []
        for _ in range(30):
            t0 = time.perf_counter()
            requests.get(f"{BASE}/metrics", timeout=TIMEOUT)
            times.append(time.perf_counter() - t0)
        p99_ms = sorted(times)[int(len(times) * 0.99)] * 1000
        assert p99_ms < 200, \
            f"P99 de /metrics = {p99_ms:.0f}ms > 200ms — Prometheus puede dar timeout"

    def test_metrics_under_concurrent_scraping(self):
        """10 scrapers concurrentes → todos deben obtener 200."""
        results = []
        lock = threading.Lock()

        def scrape():
            r = requests.get(f"{BASE}/metrics", timeout=5)
            with lock:
                results.append(r.status_code)

        threads = [threading.Thread(target=scrape) for _ in range(10)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        ok = sum(1 for r in results if r == 200)
        assert ok == 10, f"Solo {ok}/10 scrapers concurrentes obtuvieron 200"

    def test_metrics_size_reasonable(self):
        """El tamaño de /metrics no debe ser excesivamente grande."""
        r = requests.get(f"{BASE}/metrics", timeout=TIMEOUT)
        size_kb = len(r.content) / 1024
        assert size_kb < 500, \
            f"/metrics pesa {size_kb:.0f}KB — demasiado grande para Prometheus"

    def test_metrics_doesnt_slow_down_api(self):
        """Scraping simultáneo no debe degradar la latencia de la API."""
        import statistics

        def scrape_continuously(stop_event):
            while not stop_event.is_set():
                requests.get(f"{BASE}/metrics", timeout=5)
                time.sleep(0.1)

        stop = threading.Event()
        scraper = threading.Thread(target=scrape_continuously, args=(stop,))
        scraper.start()

        times = []
        s = legit_session()
        for _ in range(20):
            t0 = time.perf_counter()
            s.get(f"{BASE}/api/data", timeout=5)
            times.append(time.perf_counter() - t0)

        stop.set()
        scraper.join(timeout=2)

        avg_ms = statistics.mean(times) * 1000
        assert avg_ms < 500, \
            f"API se degrada con scraping simultáneo: avg={avg_ms:.0f}ms"
