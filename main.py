"""
main.py — RBT Security Layer  (with Risk Codes)
================================================
Every security event now emits a structured RBT-XXX-NNN risk code
visible in logs, API responses, and Prometheus labels.
"""

from fastapi import FastAPI, Request
from fastapi.responses import Response
from prometheus_client import Counter, Gauge, Histogram, Summary, generate_latest
from contextlib import asynccontextmanager
import redis, time, os, hashlib, joblib, numpy as np
from pathlib import Path

# ── Risk code system ──────────────────────────────────────────────────────────
from risk_codes import (
    RiskCode, RiskEvent, Severity,
    event_ml_blocked, event_score_exceeded, event_login_failure,
    event_headless_ua, event_missing_lang, event_false_positive,
    event_redis_error,
)

# ── Redis connection with retry ───────────────────────────────────────────────
def _connect_redis(retries: int = 10, delay: float = 2.0):
    host = os.getenv("REDIS_HOST", "localhost")
    for attempt in range(1, retries + 1):
        client = redis.Redis(host=host, port=6379, decode_responses=True)
        try:
            client.ping()
            print(f"[Redis] Connected on attempt {attempt}")
            return client
        except redis.exceptions.ConnectionError as e:
            print(f"[Redis] Attempt {attempt}/{retries} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
    raise RuntimeError(f"[Redis] Could not connect after {retries} attempts")

r = _connect_redis()

# ── ML Model ──────────────────────────────────────────────────────────────────
MODEL_PATH = Path("ml/bot_detector.pkl")
bot_model  = None

def load_model():
    global bot_model
    if MODEL_PATH.exists():
        bot_model = joblib.load(MODEL_PATH)
        print("ML model loaded")
    else:
        ev = RiskEvent(RiskCode.MLX_003_MODEL_UNAVAILABLE, "system")
        print(ev.to_log_line())

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    RISK_SCORE_METRIC.labels(identifier="system_startup").set(0)
    FALSE_POSITIVES.labels(identifier="system_startup").inc(0)
    # Observe 0 so ml_prediction_seconds_sum/count series exist from startup
    ML_PREDICTION_TIME.observe(0)
    ACTIVE_USERS.set(0)
    THREAT_LEVEL.set(0)
    # Pre-initialize so Grafana panels show 0 not "No data" before first request
    BYPASS_ATTEMPTS.labels(result="allowed").inc(0)
    BYPASS_ATTEMPTS.labels(result="denied").inc(0)
    REQUESTS_BY_STATUS.labels(status_code="200", endpoint="/api/data").inc(0)
    REQUESTS_BY_STATUS.labels(status_code="403", endpoint="/api/data").inc(0)
    REQUESTS_BY_STATUS.labels(status_code="401", endpoint="/login").inc(0)
    LOGIN_FAILURES.labels(method="password", reason="none").inc(0)
    # Pre-initialize risk_code counter with known codes so series exist at startup
    for code in ["RBT-MLX-001", "RBT-AUT-001", "RBT-BHV-001",
                 "RBT-RSK-001", "RBT-FPX-001", "RBT-SYS-001"]:
        RISK_CODE_EVENTS.labels(risk_code=code, severity="INFO").inc(0)
    yield
    r.close()

app = FastAPI(lifespan=lifespan)

# ── Prometheus metrics ────────────────────────────────────────────────────────
REQUESTS          = Counter("http_requests_total",          "Total requests",              ["method", "endpoint"])
BLOCKED           = Counter("blocked_requests_total",       "Blocked requests",            ["reason", "identifier"])
FALSE_POSITIVES   = Counter("false_positive_blocks_total",  "False positive detections",   ["identifier"])
RISK_SCORE_METRIC = Gauge  ("current_risk_score",           "Risk score per user",         ["identifier"])
LOGIN_FAILURES    = Counter("login_failures_total",         "Failed login attempts",       ["method", "reason"])
BOT_PROBABILITY   = Gauge  ("bot_ml_probability",           "ML bot probability 0-1",      ["identifier"])
ML_BLOCKED        = Counter("ml_blocked_total",             "Requests blocked by ML",      ["identifier"])

ML_PREDICTION_TIME = Summary("ml_prediction_seconds",
                             "Time spent running model.predict() per request")

ACTIVE_USERS  = Gauge("active_users_total",
                      "Distinct user fingerprints seen in Redis right now")
THREAT_LEVEL  = Gauge("system_threat_level",
                      "Overall threat level: 0=low 1=medium 2=high")

REQUEST_LATENCY    = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
REQUESTS_BY_STATUS = Counter(
    "http_responses_total",
    "Responses grouped by HTTP status code",
    ["status_code", "endpoint"]
)
BYPASS_ATTEMPTS    = Counter(
    "bypass_attempts_total",
    "X-Legitimate-User header bypass attempts",
    ["result"]   # allowed / denied
)

# NEW: counter that tracks every risk code emission
RISK_CODE_EVENTS  = Counter("rbt_risk_events_total",
                             "Security events by risk code",
                             ["risk_code", "severity"])

# ── Security config ───────────────────────────────────────────────────────────
THRESHOLD = 30
WINDOW    = 100

# ── Core functions ────────────────────────────────────────────────────────────
def get_fingerprint(request: Request) -> str:
    ua   = request.headers.get("User-Agent", "")
    lang = request.headers.get("Accept-Language", "")
    enc  = request.headers.get("Accept-Encoding", "")
    return hashlib.md5(f"{ua}|{lang}|{enc}".encode()).hexdigest()

def get_identifier(request: Request) -> str:
    return f"{get_fingerprint(request)}:{request.client.host}"

def emit(event: RiskEvent) -> None:
    """Emit a risk event to logs and Prometheus."""
    print(event.to_log_line())
    RISK_CODE_EVENTS.labels(
        risk_code=event.code,
        severity=event.severity.value,
    ).inc()

def block_response(event: RiskEvent, status: int = 403) -> Response:
    """Build a blocked response with the risk code in the body and headers."""
    emit(event)
    return Response(
        content=event.to_response_body(),
        status_code=status,
        headers={
            "X-RBT-Risk-Code":     event.code,
            "X-RBT-Severity":      event.severity.value,
            "X-RBT-Block-Reason":  event.risk_code.title,
        },
    )

def update_risk_score(identifier: str, points: float) -> float:
    key     = f"risk:{identifier}"
    current = float(r.get(key) or 0)
    new     = current + points
    r.set(key, new, ex=10000)
    RISK_SCORE_METRIC.labels(identifier=identifier).set(new)
    return new

def extract_features(request: Request, identifier: str) -> np.ndarray:
    ua        = request.headers.get("User-Agent", "").lower()
    has_lang  = int("accept-language" in request.headers)
    has_legit = int(request.headers.get("X-Legitimate-User") == "true")
    rate_key  = f"rate:{identifier}"
    rate      = int(r.zcard(rate_key))
    score     = float(r.get(f"risk:{identifier}") or 0)
    fails     = float(r.get(f"fails:{identifier}") or 0)
    is_bot_ua = int(any(b in ua for b in ["headless","selenium","puppeteer","playwright"]))
    return np.array([[is_bot_ua, has_lang, rate, score, fails, has_legit]])

def analyze_behavioral_ai(request: Request) -> tuple[float, list[RiskEvent]]:
    """
    Returns (score_points, list_of_risk_events).
    Caller decides whether to emit events.
    """
    ua       = request.headers.get("User-Agent", "").lower()
    has_lang = "accept-language" in request.headers
    events   = []
    points   = 0.0

    is_headless = any(b in ua for b in ["headless","selenium","puppeteer","playwright"])

    if is_headless and not has_lang:
        points  = 20.0
        # BHV-003 covers both signals together
    elif is_headless:
        points  = 15.0
    elif not has_lang:
        points  = 5.0

    return points, events  # caller builds specific events with identifier

# ── Middleware ────────────────────────────────────────────────────────────────
def _update_active_users_and_threat() -> None:
    """Refresh active-user count and threat level from Redis keys.
    Called at the end of every request that passes through the middleware.
    """
    try:
        # Count distinct risk: keys — each represents one active user session
        keys   = r.keys("risk:*")
        active = len([k for k in keys if k != "risk:system_startup"])
        ACTIVE_USERS.set(active)

        # Threat level: based on how many sessions have failed-login keys
        blocked_keys = len(r.keys("fails:*"))
        if blocked_keys > 10:
            THREAT_LEVEL.set(2)   # HIGH
        elif blocked_keys > 3:
            THREAT_LEVEL.set(1)   # MEDIUM
        else:
            THREAT_LEVEL.set(0)   # LOW
    except redis.exceptions.ConnectionError:
        pass  # Redis unavailable — keep last known values


@app.middleware("http")
async def security_middleware(request: Request, call_next):

    # Excluded from inspection: health, metrics, docs, status
    if request.url.path in ["/", "/metrics", "/status", "/openapi.json", "/docs", "/redoc"]:
        return await call_next(request)

    REQUESTS.labels(method=request.method, endpoint=request.url.path).inc()
    _t0 = time.perf_counter()

    identifier = get_identifier(request)

    # ── Rate window tracking ───────────────────────────────────
    rate_key = f"rate:{identifier}"
    now      = int(time.time())
    try:
        r.zadd(rate_key, {now: now})
        r.zremrangebyscore(rate_key, 0, now - WINDOW)
        r.expire(rate_key, WINDOW + 10)
    except redis.exceptions.ConnectionError as e:
        ev = event_redis_error(identifier, str(e))
        emit(ev)
        return await call_next(request)  # fail-open

    # ── ML detection ───────────────────────────────────────────
    if bot_model is not None:
        try:
            features = extract_features(request, identifier)
            with ML_PREDICTION_TIME.time():
                prediction  = int(bot_model.predict(features)[0])
                probability = float(bot_model.predict_proba(features)[0][1])
            BOT_PROBABILITY.labels(identifier=identifier).set(probability)

            if prediction == 1:
                # Check bypass before blocking
                if request.headers.get("X-Legitimate-User") == "true":
                    final_score = float(r.get(f"risk:{identifier}") or 0)
                    ev = event_false_positive(identifier, final_score)
                    emit(ev)
                    FALSE_POSITIVES.labels(identifier=identifier).inc()
                    return await call_next(request)

                ev = event_ml_blocked(identifier, probability)
                ML_BLOCKED.labels(identifier=identifier).inc()
                BLOCKED.labels(
                    reason=f"{ev.code}:ml_bot_detected",
                    identifier=identifier,
                ).inc()
                elapsed = time.perf_counter() - _t0
                REQUEST_LATENCY.labels(endpoint=request.url.path).observe(elapsed)
                REQUESTS_BY_STATUS.labels(status_code="403",
                                          endpoint=request.url.path).inc()
                return block_response(ev)

        except Exception as e:
            ev = RiskEvent(RiskCode.SYS_002_ML_PREDICTION_ERROR, identifier,
                           details={"error": str(e)[:80]})
            emit(ev)

    # ── Rule-based scoring ─────────────────────────────────────
    ua       = request.headers.get("User-Agent", "").lower()
    has_lang = "accept-language" in request.headers
    is_headless = any(b in ua for b in ["headless","selenium","puppeteer","playwright"])

    if is_headless or not has_lang:
        ai_points = 20.0 if (is_headless and not has_lang) else (15.0 if is_headless else 5.0)
        bhv_event = event_headless_ua(identifier, ua, has_lang) if is_headless \
                    else event_missing_lang(identifier)
        emit(bhv_event)
        update_risk_score(identifier, ai_points)

    final_score = float(r.get(f"risk:{identifier}") or 0)

    if final_score > THRESHOLD:
        if request.headers.get("X-Legitimate-User") == "true":
            ev = event_false_positive(identifier, final_score)
            emit(ev)
            FALSE_POSITIVES.labels(identifier=identifier).inc()
            BYPASS_ATTEMPTS.labels(result="allowed").inc()
            response = await call_next(request)
            elapsed = time.perf_counter() - _t0
            REQUEST_LATENCY.labels(endpoint=request.url.path).observe(elapsed)
            REQUESTS_BY_STATUS.labels(
                status_code=str(response.status_code),
                endpoint=request.url.path).inc()
            return response

        ev = event_score_exceeded(identifier, final_score)
        BLOCKED.labels(
            reason=f"{ev.code}:risk_score_exceeded",
            identifier=identifier,
        ).inc()
        BYPASS_ATTEMPTS.labels(result="denied").inc()
        elapsed = time.perf_counter() - _t0
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(elapsed)
        REQUESTS_BY_STATUS.labels(status_code="403",
                                  endpoint=request.url.path).inc()
        return block_response(ev)

    _update_active_users_and_threat()
    response = await call_next(request)
    elapsed = time.perf_counter() - _t0
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(elapsed)
    REQUESTS_BY_STATUS.labels(
        status_code=str(response.status_code),
        endpoint=request.url.path).inc()
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"status": "RBT Security Layer Active", "ml_model_loaded": bot_model is not None}

@app.get("/status")
def status():
    return {
        "status":          "running",
        "ml_model_loaded": bot_model is not None,
        "model_path":      str(MODEL_PATH),
        "threshold":       THRESHOLD,
        "risk_codes":      len(__import__("risk_codes").CATALOG),
    }

@app.get("/api/data")
def protected_data():
    return {"data": "secure_content"}

@app.get("/login")
def login(username: str = "", password: str = ""):
    identifier = "login_check"
    if username == "admin" and password == "secret123":
        return {"message": "Welcome", "status": "success"}

    fails_key = f"fails:{identifier}"
    fails     = int(r.incr(fails_key))
    r.expire(fails_key, 600)
    LOGIN_FAILURES.labels(method="password", reason="invalid_credentials").inc()

    ev = event_login_failure(identifier, fails)
    emit(ev)
    if ev.risk_code.points > 0:
        update_risk_score(identifier, ev.risk_code.points)

    return Response(content="Invalid credentials", status_code=401)

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/risk-codes")
def risk_codes_catalog():
    """List all defined risk codes — useful for documentation and dashboards."""
    from risk_codes import CATALOG
    return {
        "total": len(CATALOG),
        "codes": [
            {
                "code":        c.code,
                "category":    c.category.value,
                "severity":    c.severity.value,
                "title":       c.title,
                "cvss":        c.cvss,
                "points":      c.points,
            }
            for c in sorted(CATALOG.values(), key=lambda x: x.code)
        ],
    }
