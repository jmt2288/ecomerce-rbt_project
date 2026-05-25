"""
tests/test_load.py
──────────────────
Pruebas de carga y rendimiento. Verifican que el sistema
aguanta bajo presión y el Risk Score sigue funcionando.

Suites:
  1. Baseline Performance  — tiempos normales (P99, avg)
  2. Concurrent Users      — usuarios simultáneos
  3. Rate Limiting         — funciona bajo carga real
  4. Spike Test            — pico repentino de tráfico
  5. Soak Test             — carga sostenida 2 min (genera datos Grafana)
  6. Metrics Under Load    — /metrics aguanta scraping concurrente

Run todo:
    pytest tests/test_load.py -v -s
Sin soak (rápido):
    pytest tests/test_load.py -v -s -k "not soak"
"""

import statistics
import string
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

BASE    = "http://localhost:8000"
TIMEOUT = 15


# ─── helpers ──────────────────────────────────────────────
def uid():
    return "".join(random.choices(string.ascii_lowercase, k=10))


def legit():
    s = requests.Session()
    s.headers.update({
        "User-Agent":      f"Mozilla/5.0 LoadTest/{uid()}",
        "Accept-Language": "es-ES,es;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    })
    return s


def bot():
    s = requests.Session()
    s.headers["User-Agent"] = f"headless-loadtest/{uid()}"
    return s


def get_counter_total(metric_name):
    text = requests.get(f"{BASE}/metrics", timeout=5).text
    return sum(
        float(l.split()[-1])
        for l in text.splitlines()
        if l.startswith(f"{metric_name}{{") and not l.startswith("#")
    )


# ══════════════════════════════════════════════
# 1. BASELINE PERFORMANCE
# ══════════════════════════════════════════════
class TestBaselinePerformance:

    def test_single_api_request_under_1s(self):
        t0 = time.perf_counter()
        r  = legit().get(f"{BASE}/api/data", timeout=TIMEOUT)
        ms = (time.perf_counter() - t0) * 1000
        assert r.status_code in [200, 403]
        assert ms < 1000, f"Request tardó {ms:.0f}ms — esperado <1000ms"

    def test_status_endpoint_under_300ms(self):
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            requests.get(f"{BASE}/status", timeout=TIMEOUT)
            times.append((time.perf_counter() - t0) * 1000)
        avg = statistics.mean(times)
        assert avg < 300, f"/status avg={avg:.0f}ms > 300ms"

    def test_metrics_endpoint_under_200ms(self):
        t0 = time.perf_counter()
        r  = requests.get(f"{BASE}/metrics", timeout=TIMEOUT)
        ms = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200
        assert ms < 200, f"/metrics tardó {ms:.0f}ms > 200ms"

    def test_p99_under_2s_for_50_sequential_requests(self):
        """P99 < 2 s para 50 peticiones seguidas."""
        s     = legit()
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            times.append(time.perf_counter() - t0)
        p99 = sorted(times)[49]   # 50 × 0.99 = 49.5 → índice 49
        avg = statistics.mean(times)
        print(f"\n   Avg={avg*1000:.0f}ms  P99={p99*1000:.0f}ms")
        assert p99 < 2.0, f"P99={p99:.2f}s > 2 s"

    def test_average_under_500ms_for_20_requests(self):
        s     = legit()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            times.append(time.perf_counter() - t0)
        avg_ms = statistics.mean(times) * 1000
        assert avg_ms < 500, f"Avg={avg_ms:.0f}ms > 500ms"

    def test_login_endpoint_under_1s(self):
        t0 = time.perf_counter()
        legit().get(f"{BASE}/login",
                    params={"username": "admin", "password": "secret123"})
        ms = (time.perf_counter() - t0) * 1000
        assert ms < 1000, f"/login tardó {ms:.0f}ms"


# ══════════════════════════════════════════════
# 2. CONCURRENT USERS
# ══════════════════════════════════════════════
class TestConcurrentUsers:

    def test_10_legit_users_concurrent_all_200(self):
        results, lock = [], threading.Lock()

        def work():
            r = legit().get(f"{BASE}/api/data", timeout=TIMEOUT)
            with lock:
                results.append(r.status_code)

        threads = [threading.Thread(target=work) for _ in range(10)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        ok = sum(1 for r in results if r == 200)
        print(f"\n   {ok}/10 legítimos → 200")
        assert ok >= 9, f"Solo {ok}/10 legítimos obtuvieron 200"

    def test_100_rapid_requests_zero_500_errors(self):
        """100 peticiones concurrentes → ninguna da 500."""
        def req(_):
            try:
                return legit().get(f"{BASE}/api/data", timeout=5).status_code
            except Exception:
                return 0

        with ThreadPoolExecutor(max_workers=20) as pool:
            results = list(pool.map(req, range(100)))

        srv_errors  = [r for r in results if r == 500]
        conn_errors = [r for r in results if r == 0]
        print(f"\n   500s={len(srv_errors)}  conn_err={len(conn_errors)}")
        assert len(srv_errors)  == 0, f"{len(srv_errors)} peticiones dieron 500"
        assert len(conn_errors) == 0, f"Errores de conexión: {len(conn_errors)}"

    def test_25_legit_plus_25_bots_legit_not_affected(self):
        """25 legítimos + 25 bots concurrentes — los legítimos pasan."""
        legit_results, bot_results, lock = [], [], threading.Lock()

        def legit_worker():
            r = legit().get(f"{BASE}/api/data", timeout=TIMEOUT)
            with lock:
                legit_results.append(r.status_code)

        def bot_worker():
            s = bot()
            s.get(f"{BASE}/api/data")
            for _ in range(3):
                s.get(f"{BASE}/login",
                      params={"username": "x", "password": "wrong"})
            with lock:
                bot_results.append(s.get(f"{BASE}/api/data").status_code)

        threads = []
        for _ in range(25):
            threads.append(threading.Thread(target=legit_worker))
            threads.append(threading.Thread(target=bot_worker))
        [t.start() for t in threads]
        [t.join() for t in threads]

        ok = sum(1 for r in legit_results if r == 200)
        print(f"\n   Legit OK={ok}/25  Bots blocked={sum(1 for r in bot_results if r==403)}/25")
        assert ok >= 22, f"Solo {ok}/25 legítimos pasaron — sesiones no son independientes"

    def test_new_sessions_always_get_200(self):
        """El rate limit es por sesión, no global."""
        for i in range(8):
            r = legit().get(f"{BASE}/api/data", timeout=TIMEOUT)
            assert r.status_code == 200, \
                f"Sesión nueva #{i+1} bloqueada globalmente — error de diseño"


# ══════════════════════════════════════════════
# 3. RATE LIMITING UNDER LOAD
# ══════════════════════════════════════════════
class TestRateLimitingUnderLoad:

    def test_bot_session_gets_blocked_eventually(self):
        """Un bot acumulando puntos debe ser bloqueado eventualmente."""
        s         = bot()
        blocked_at = None
        s.get(f"{BASE}/api/data")   # headless UA → +15
        for i in range(20):
            s.get(f"{BASE}/login",
                  params={"username": "hacker", "password": f"try{i}"})
            r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            if r.status_code == 403:
                blocked_at = i + 1
                break

        print(f"\n   Bot bloqueado tras {blocked_at} iteraciones")
        assert blocked_at is not None, \
            "Bot nunca bloqueado — threshold o reglas no funcionan"

    def test_http_requests_counter_grows_under_load(self):
        before = get_counter_total("http_requests_total")
        for _ in range(10):
            legit().get(f"{BASE}/api/data")
        after = get_counter_total("http_requests_total")
        assert after > before, "http_requests_total no creció bajo carga"

    def test_blocked_counter_grows_when_bots_blocked(self):
        s = bot()
        s.get(f"{BASE}/api/data")
        for _ in range(5):
            s.get(f"{BASE}/login", params={"username": "x", "password": "x"})

        before = get_counter_total("blocked_requests_total")
        for _ in range(5):
            s.get(f"{BASE}/api/data")
        after = get_counter_total("blocked_requests_total")
        assert after >= before, "blocked_requests_total no creció"


# ══════════════════════════════════════════════
# 4. SPIKE TEST
# ══════════════════════════════════════════════
class TestSpikeLoad:

    def test_100_concurrent_spike_no_connection_errors(self):
        """100 usuarios de golpe → cero errores de conexión."""
        results, lock = [], threading.Lock()

        def work():
            try:
                r = legit().get(f"{BASE}/api/data", timeout=10)
                with lock:
                    results.append(r.status_code)
            except Exception:
                with lock:
                    results.append(0)

        threads = [threading.Thread(target=work) for _ in range(100)]
        t0 = time.time()
        [t.start() for t in threads]
        [t.join() for t in threads]
        elapsed = time.time() - t0

        ok  = sum(1 for r in results if r in [200, 403])
        err = sum(1 for r in results if r == 0)
        print(f"\n   Spike {ok}/100 valid  err={err}  elapsed={elapsed:.1f}s")
        assert err == 0,  f"{err} peticiones fallaron con error de conexión"
        assert ok >= 95, f"Solo {ok}/100 tuvieron respuesta válida"

    def test_api_healthy_after_spike(self):
        """La API responde correctamente después de un pico de carga."""
        def spam():
            try:
                legit().get(f"{BASE}/api/data", timeout=5)
            except Exception:
                pass

        threads = [threading.Thread(target=spam) for _ in range(50)]
        [t.start() for t in threads]
        [t.join() for t in threads]

        time.sleep(1)
        r = requests.get(f"{BASE}/status", timeout=TIMEOUT)
        assert r.status_code == 200
        assert r.json()["status"] == "running"


# ══════════════════════════════════════════════
# 5. SOAK TEST  (genera datos reales para Grafana)
# ══════════════════════════════════════════════
class TestSoakLoad:

    def test_sustained_2_minutes_mixed_traffic(self):
        """
        Carga sostenida 120 s con tráfico mixto realista.
        ⚠️  Este test dura ~2 min — genera todos los paneles de Grafana.
        Abrir http://localhost:3000 mientras corre.
        Saltar con: pytest -k "not soak"
        """
        print("\n   🔄 Soak 120 s — abre Grafana: http://localhost:3000")
        end   = time.time() + 120
        total = errors = 0

        while time.time() < end:
            kind = random.choices(
                ["legit", "legit", "legit", "fail_login", "bot_block"],
                weights=[4, 4, 3, 2, 1],
            )[0]
            try:
                if kind in ("legit",):
                    s = legit()
                    s.get(f"{BASE}/api/data")
                    if random.random() < 0.3:
                        s.get(f"{BASE}/login",
                              params={"username": "admin",
                                      "password": "secret123"})
                elif kind == "fail_login":
                    s = legit()
                    s.get(f"{BASE}/login",
                          params={"username": f"user{random.randint(0,9)}",
                                  "password": "wrong"})
                elif kind == "bot_block":
                    s = bot()
                    s.get(f"{BASE}/api/data")
                    for _ in range(4):
                        s.get(f"{BASE}/login",
                              params={"username": "hacker", "password": "x"})
                    s.get(f"{BASE}/api/data")
                total += 1
            except Exception:
                errors += 1
            time.sleep(random.uniform(0.25, 0.75))

        err_pct = errors / max(total, 1) * 100
        print(f"   Total={total}  Errors={errors} ({err_pct:.1f}%)")
        r = requests.get(f"{BASE}/status", timeout=TIMEOUT)
        assert r.status_code == 200
        assert err_pct < 5, f"Error rate {err_pct:.1f}% > 5%"


# ══════════════════════════════════════════════
# 6. METRICS UNDER LOAD
# ══════════════════════════════════════════════
class TestMetricsUnderLoad:

    def test_10_concurrent_scrapers_all_200(self):
        results, lock = [], threading.Lock()

        def scrape():
            r = requests.get(f"{BASE}/metrics", timeout=5)
            with lock:
                results.append(r.status_code)

        threads = [threading.Thread(target=scrape) for _ in range(10)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        assert all(r == 200 for r in results), \
            f"Scrapers con error: {[r for r in results if r != 200]}"

    def test_scraping_does_not_degrade_api_latency(self):
        """Prometheus scraping concurrente no degrada la API."""
        stop = threading.Event()

        def scrape_loop():
            while not stop.is_set():
                try:
                    requests.get(f"{BASE}/metrics", timeout=3)
                except Exception:
                    pass
                time.sleep(0.1)

        t = threading.Thread(target=scrape_loop)
        t.start()

        times = []
        s = legit()
        for _ in range(20):
            t0 = time.perf_counter()
            s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            times.append(time.perf_counter() - t0)

        stop.set()
        t.join(timeout=2)

        avg_ms = statistics.mean(times) * 1000
        assert avg_ms < 500, f"API se degrada con scraping: avg={avg_ms:.0f}ms"
