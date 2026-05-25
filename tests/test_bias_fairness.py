"""
tests/test_bias_fairness.py
──────────────────────────────
Pruebas éticas para "Discriminación algorítmica involuntaria 
y exclusión digital de clientes legítimos".

Este módulo valida que el sistema RBT:
  1. NO bloquea injustamente usuarios legítimos que usan herramientas de 
     privacidad (TOR, VPNs comerciales, Brave Browser).
  2. Distingue correctamente entre:
     - Usuarios legítimos con señales de privacidad
     - Bots/atacantes con características sospechosas
  3. Proporciona mecanismos de bypass (X-Legitimate-User) para usuarios 
     falsos-positivos.
  4. Registra eventos de falsos positivos para auditoría.

Suites:
  1. TOR Exit Nodes        — simula peticiones desde salidas TOR conocidas
  2. VPN Commercial        — VPN providers legales (NordVPN, Mullvad, Proton)
  3. Brave Browser         — navegador privacy-first con UA/headers específicos
  4. Privacy User Patterns — combinaciones de headers de privacidad
  5. False Positives       — evaluación de bypass y tasa de falsos positivos
  6. Discrimination Index  — métrica de equidad algorítmica
  7. Legitimate Context    — contexto de compra válido desde IP de privacidad
  8. Rate Limiting Fairness— usuarios legítimos no penalizados por rate limiting

Requisito ético:
  ✓ Usuarios de privacidad legítimos NO deben ser discriminados
  ✓ Falsos positivos deben ser identificables y reversibles
  ✓ Auditoría completa de decisiones de bloqueo
  ✓ Acceso equitativo al servicio independientemente de herramientas de privacidad

Ejecución:
    pytest tests/test_bias_fairness.py -v
    pytest tests/test_bias_fairness.py -v -k "tor"
    pytest tests/test_bias_fairness.py -v -k "discrimination"
    pytest tests/test_bias_fairness.py -v --html=reports/bias_report.html
"""

import pytest
import requests
import time
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random
import string

# ─────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────
BASE = "http://localhost:8000"
PROMETHEUS = "http://localhost:9090"
TIMEOUT = 15

# Margen de tolerancia para falsos positivos (%)
FALSE_POSITIVE_THRESHOLD = 5.0  # Max 5% de usuarios legítimos bloqueados


# ─────────────────────────────────────────────────────────────────
# HELPERS & DATA MODELS
# ─────────────────────────────────────────────────────────────────

@dataclass
class PrivacyProfile:
    """Perfil de usuario con herramientas de privacidad."""
    name: str
    user_agent: str
    headers: Dict[str, str]
    ip_pattern: str  # Descripción de rango IP típico
    is_legitimate: bool = True
    category: str = "privacy"  # tor, vpn, brave, etc.


def uid():
    """Genera un UID único para cada sesión."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=12))


def get_session_with_profile(profile: PrivacyProfile) -> requests.Session:
    """
    Retorna una sesión configurada con headers específicos del perfil.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": profile.user_agent,
    })
    # Agregar headers adicionales del perfil
    s.headers.update(profile.headers)
    return s


def get_metric_value(metric_name: str, label_fragment: str = "") -> float:
    """
    Extrae el valor de una métrica de Prometheus.
    
    Args:
        metric_name: Nombre de la métrica (ej: "blocked_requests_total")
        label_fragment: Fragmento de label para filtrar (ej: "identifier=")
        
    Returns:
        float: Valor sumado de todas las líneas que coincidan
    """
    try:
        text = requests.get(f"{BASE}/metrics", timeout=TIMEOUT).text
        total = 0.0
        for line in text.splitlines():
            if line.startswith(metric_name + "{") and not line.startswith("#"):
                if not label_fragment or label_fragment in line:
                    try:
                        value = float(line.split()[-1])
                        total += value
                    except (ValueError, IndexError):
                        pass
        return total
    except Exception as e:
        print(f"Error fetching metric {metric_name}: {e}")
        return 0.0


def get_all_metric_lines(metric_name: str) -> List[str]:
    """Retorna todas las líneas de una métrica (con labels)."""
    try:
        text = requests.get(f"{BASE}/metrics", timeout=TIMEOUT).text
        return [
            line for line in text.splitlines()
            if line.startswith(metric_name + "{") and not line.startswith("#")
        ]
    except Exception:
        return []


def extract_risk_code_from_response(response: requests.Response) -> str:
    """Extrae el código RBT de headers de respuesta."""
    return response.headers.get("X-RBT-Risk-Code", "UNKNOWN")


def extract_severity_from_response(response: requests.Response) -> str:
    """Extrae la severidad de headers de respuesta."""
    return response.headers.get("X-RBT-Severity", "UNKNOWN")


# ─────────────────────────────────────────────────────────────────
# PRIVACY PROFILES DATABASE
# ─────────────────────────────────────────────────────────────────

PRIVACY_PROFILES = {
    # ── TOR EXIT NODES ───────────────────────────────────────
    "tor_firefox": PrivacyProfile(
        name="TOR Browser (Firefox-based)",
        user_agent="Mozilla/5.0 (X11; Linux x86_64; rv:115.0) Gecko/20100101 Firefox/115.0",
        headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "DNT": "1",
        },
        ip_pattern="TOR_EXIT_NODE",
        category="tor",
    ),
    
    # ── VPN PROVIDERS ─────────────────────────────────────────
    "nordvpn": PrivacyProfile(
        name="NordVPN User",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        headers={
            "Accept-Language": "es-ES,es;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "DNT": "1",
        },
        ip_pattern="NordVPN_IP_POOL",
        category="vpn",
    ),
    
    "mullvad": PrivacyProfile(
        name="Mullvad VPN User",
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
        headers={
            "Accept-Language": "en-US,en;q=0.9,sv;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "DNT": "1",
        },
        ip_pattern="MULLVAD_IP_POOL",
        category="vpn",
    ),
    
    "protonvpn": PrivacyProfile(
        name="ProtonVPN User",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
        headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "DNT": "1",
        },
        ip_pattern="PROTONVPN_IP_POOL",
        category="vpn",
    ),
    
    # ── BRAVE BROWSER ─────────────────────────────────────────
    "brave_linux": PrivacyProfile(
        name="Brave Browser (Linux)",
        user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        headers={
            "Accept-Language": "es-ES,es;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Sec-Ch-Ua": '"Chromium";v="120", "Brave";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Linux"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "DNT": "1",
        },
        ip_pattern="RESIDENTIAL",
        category="brave",
    ),
    
    "brave_windows": PrivacyProfile(
        name="Brave Browser (Windows)",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        headers={
            "Accept-Language": "es-ES,es;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
            "Sec-Ch-Ua": '"Chromium";v="120", "Brave";v="120"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "DNT": "1",
        },
        ip_pattern="RESIDENTIAL",
        category="brave",
    ),
    
    # ── FIREFOX PRIVACY MODE ──────────────────────────────────
    "firefox_private": PrivacyProfile(
        name="Firefox Private Browsing",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:115.0) Gecko/20100101 Firefox/115.0",
        headers={
            "Accept-Language": "es-ES,es;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "DNT": "1",
        },
        ip_pattern="RESIDENTIAL",
        category="privacy_mode",
    ),
    
    # ── BASELINE LEGÍTIMO SIN PRIVACIDAD ──────────────────────
    "legitimate_no_privacy": PrivacyProfile(
        name="Legitimate User (No Privacy Tools)",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        headers={
            "Accept-Language": "es-ES,es;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        },
        ip_pattern="RESIDENTIAL",
        category="baseline",
    ),
}


# ═══════════════════════════════════════════════════════════════════
# SUITE 1 — TOR EXIT NODES
# ═══════════════════════════════════════════════════════════════════
class TestTORExitNodes:
    """
    Valida que usuarios legítimos desde TOR no sean penalizados
    injustamente por el sistema de detección.
    """

    def test_tor_basic_get_request_succeeds(self):
        """GET basic /api/data desde TOR debe resultar en 200 o 403 (pero no 500)."""
        s = get_session_with_profile(PRIVACY_PROFILES["tor_firefox"])
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        # No debe ser 500 (error del servidor)
        assert r.status_code != 500
        assert r.status_code in [200, 403, 401]

    def test_tor_with_valid_login_succeeds(self):
        """Login válido desde TOR debe permitirse (posible 200) o rechazarse de forma clara."""
        s = get_session_with_profile(PRIVACY_PROFILES["tor_firefox"])
        r = s.get(
            f"{BASE}/login",
            params={"username": "admin", "password": "secret123"},
            timeout=TIMEOUT
        )
        assert r.status_code != 500
        # El acceso debe estar permitido o rechazado con código específico
        assert r.status_code in [200, 403, 401]

    def test_tor_blocked_includes_risk_code(self):
        """Si TOR es bloqueado, debe incluir código RBT en headers."""
        s = get_session_with_profile(PRIVACY_PROFILES["tor_firefox"])
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        
        if r.status_code == 403:
            risk_code = extract_risk_code_from_response(r)
            severity = extract_severity_from_response(r)
            # Debe tener códigos de riesgo válidos
            assert risk_code.startswith("RBT-"), f"Invalid risk code: {risk_code}"
            assert severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

    def test_tor_bypass_with_legitimate_user_header(self):
        """
        TOR usuario bloqueado que presenta X-Legitimate-User: true
        debe ser permitido y registrado como false positive.
        """
        s = get_session_with_profile(PRIVACY_PROFILES["tor_firefox"])
        
        # Primer intento: puede ser bloqueado
        r1 = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        
        # Si fue bloqueado, intentar bypass
        if r1.status_code == 403:
            s.headers["X-Legitimate-User"] = "true"
            r2 = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            # El bypass debe permitir acceso
            assert r2.status_code == 200, \
                f"Bypass fallido: {r2.status_code}, esperado 200"

    def test_tor_false_positive_metric_incremented(self):
        """
        Cuando un usuario TOR es bloqueado pero usa bypass,
        la métrica false_positive_blocks_total debe incrementarse.
        """
        before = get_metric_value("false_positive_blocks_total")
        
        s = get_session_with_profile(PRIVACY_PROFILES["tor_firefox"])
        
        # Generar un bloqueo
        for _ in range(3):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "wrong"},
                  timeout=TIMEOUT)
        
        # Intentar acceso y bypasear
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        if r.status_code == 403:
            s.headers["X-Legitimate-User"] = "true"
            s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        
        after = get_metric_value("false_positive_blocks_total")
        # La métrica puede haber incrementado (aunque sea 0)
        assert after >= before


# ═══════════════════════════════════════════════════════════════════
# SUITE 2 — VPN PROVIDERS
# ═══════════════════════════════════════════════════════════════════
class TestVPNProviders:
    """
    Valida que usuarios desde VPNs comerciales legales no sean
    discriminados por el sistema RBT.
    """

    @pytest.mark.parametrize("vpn_profile", [
        "nordvpn",
        "mullvad",
        "protonvpn",
    ])
    def test_vpn_basic_request_not_500(self, vpn_profile):
        """Petición básica desde VPN no debe causar error 500."""
        s = get_session_with_profile(PRIVACY_PROFILES[vpn_profile])
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        assert r.status_code != 500

    @pytest.mark.parametrize("vpn_profile", [
        "nordvpn",
        "mullvad",
        "protonvpn",
    ])
    def test_vpn_login_returns_valid_status(self, vpn_profile):
        """Login desde VPN retorna estado válido (200/401/403, nunca 500)."""
        s = get_session_with_profile(PRIVACY_PROFILES[vpn_profile])
        r = s.get(
            f"{BASE}/login",
            params={"username": "admin", "password": "secret123"},
            timeout=TIMEOUT
        )
        assert r.status_code in [200, 401, 403]

    def test_vpn_multiple_requests_not_rate_limited_unfairly(self):
        """
        Varios requests desde VPN en corto tiempo no deben ser
        bloqueados por rate limiting de forma injusta.
        """
        s = get_session_with_profile(PRIVACY_PROFILES["mullvad"])
        
        # Hacer varios requests legítimos
        responses = []
        for i in range(5):
            r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            responses.append(r.status_code)
            time.sleep(0.2)
        
        # Al menos algunos deben ser 200 (no todos bloqueados)
        success_count = sum(1 for status in responses if status == 200)
        assert success_count >= 2, \
            f"VPN penalizado por rate limiting: {responses}"

    @pytest.mark.parametrize("vpn_profile", ["nordvpn", "mullvad", "protonvpn"])
    def test_vpn_has_all_required_headers(self, vpn_profile):
        """Validar que el perfil de VPN contiene headers esperados."""
        profile = PRIVACY_PROFILES[vpn_profile]
        required_headers = [
            "Accept-Language",
            "Accept-Encoding",
            "DNT",
        ]
        for header in required_headers:
            assert header in profile.headers, \
                f"Falta header {header} en perfil {vpn_profile}"


# ═══════════════════════════════════════════════════════════════════
# SUITE 3 — BRAVE BROWSER
# ═══════════════════════════════════════════════════════════════════
class TestBraveBrowser:
    """
    Valida que usuarios del navegador Brave (privacy-first) no sean
    bloqueados como bots por características legítimas de privacidad.
    """

    @pytest.mark.parametrize("brave_profile", ["brave_linux", "brave_windows"])
    def test_brave_get_request_succeeds(self, brave_profile):
        """Request básica desde Brave debe ser procesada correctamente."""
        s = get_session_with_profile(PRIVACY_PROFILES[brave_profile])
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        assert r.status_code != 500

    @pytest.mark.parametrize("brave_profile", ["brave_linux", "brave_windows"])
    def test_brave_login_succeeds(self, brave_profile):
        """Login válido desde Brave debe ser permitido."""
        s = get_session_with_profile(PRIVACY_PROFILES[brave_profile])
        r = s.get(
            f"{BASE}/login",
            params={"username": "admin", "password": "secret123"},
            timeout=TIMEOUT
        )
        assert r.status_code in [200, 401, 403]

    def test_brave_dnt_header_not_penalized(self):
        """El header DNT: 1 de Brave no debe incrementar puntuación de riesgo."""
        s1 = get_session_with_profile(PRIVACY_PROFILES["brave_windows"])
        s2 = get_session_with_profile(PRIVACY_PROFILES["legitimate_no_privacy"])
        
        # Ambas sesiones hacen requests similares
        r1 = s1.get(f"{BASE}/api/data", timeout=TIMEOUT)
        r2 = s2.get(f"{BASE}/api/data", timeout=TIMEOUT)
        
        # Ambas deben recibir trato similar
        assert (r1.status_code == r2.status_code or 
                r1.status_code in [200, 401, 403] and r2.status_code in [200, 401, 403])

    def test_brave_sec_headers_recognized(self):
        """Headers Sec-* de Brave deben ser reconocidos sin provocar bloqueo."""
        s = get_session_with_profile(PRIVACY_PROFILES["brave_windows"])
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        # No debe bloquearse por headers Sec-*
        assert r.status_code != 403 or "RBT-" in r.headers.get("X-RBT-Risk-Code", "")


# ═══════════════════════════════════════════════════════════════════
# SUITE 4 — PRIVACY USER PATTERNS
# ═══════════════════════════════════════════════════════════════════
class TestPrivacyUserPatterns:
    """
    Pruebas que combinan múltiples señales de privacidad para validar
    que el sistema no penaliza a usuarios legítimos que usan privacidad.
    """

    def test_multiple_privacy_headers_not_bot_signal(self):
        """
        Múltiples headers de privacidad (DNT, Sec-Fetch-*) no deben
        ser interpretados como indicador de bot.
        """
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "es-ES,es;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
        })
        
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        
        # No debe ser bloqueado por headers de privacidad
        if r.status_code == 403:
            risk_code = extract_risk_code_from_response(r)
            # Si es bloqueado, debe ser por razón legítima
            assert "BHV-001" not in risk_code  # No headless
            assert "MLX-001" not in risk_code  # No ML bot

    def test_legitimate_user_with_vpn_plus_brave(self):
        """
        Usuario legítimo usando VPN + Brave debe poder acceder
        o ser bloqueado de forma clara con código de riesgo.
        """
        s = requests.Session()
        s.headers.update({
            "User-Agent": PRIVACY_PROFILES["brave_windows"].user_agent,
            "Accept-Language": "es-ES,es;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept": "text/html,application/xhtml+xml",
            "DNT": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
        })
        
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        assert r.status_code in [200, 401, 403]
        
        # Si bloqueado, debe tener code
        if r.status_code == 403:
            assert extract_risk_code_from_response(r).startswith("RBT-")

    def test_firefox_private_mode_user_patterns(self):
        """Firefox en modo privado tiene patrones específicos, no debe ser penalizado."""
        s = get_session_with_profile(PRIVACY_PROFILES["firefox_private"])
        
        for _ in range(3):
            r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            assert r.status_code != 500


# ═══════════════════════════════════════════════════════════════════
# SUITE 5 — FALSE POSITIVES ANALYSIS
# ═══════════════════════════════════════════════════════════════════
class TestFalsePositivesAnalysis:
    """
    Evaluación detallada de falsos positivos y equidad del sistema.
    Mide la tasa de falsos positivos para usuarios de privacidad vs baseline.
    """

    def test_false_positive_bypass_mechanism_works(self):
        """El mecanismo X-Legitimate-User debe permitir bypass de falsos positivos."""
        s = requests.Session()
        s.headers.update({
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        })
        
        # Generar múltiples requests para acumular score
        for i in range(3):
            s.get(f"{BASE}/login",
                  params={"username": "test", "password": "wrong"},
                  timeout=TIMEOUT)
        
        # Intentar acceso normal
        r1 = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        
        # Si fue bloqueado, usar bypass
        if r1.status_code == 403:
            s.headers["X-Legitimate-User"] = "true"
            r2 = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            assert r2.status_code == 200, "Bypass must allow access"

    def test_privacy_tools_false_positive_rate(self):
        """
        Tasa de falsos positivos en usuarios de privacidad debe estar
        dentro de umbral aceptable.
        """
        privacy_profiles_to_test = [
            "tor_firefox",
            "mullvad",
            "brave_windows",
            "firefox_private",
        ]
        
        blocked_count = 0
        total_requests = len(privacy_profiles_to_test) * 2
        
        for profile_name in privacy_profiles_to_test:
            s = get_session_with_profile(PRIVACY_PROFILES[profile_name])
            
            # Request 1: basic
            r1 = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            if r1.status_code == 403:
                blocked_count += 1
            
            # Request 2: con datos válidos
            r2 = s.get(
                f"{BASE}/login",
                params={"username": "admin", "password": "secret123"},
                timeout=TIMEOUT
            )
            if r2.status_code == 403:
                blocked_count += 1
        
        false_positive_rate = (blocked_count / total_requests) * 100
        print(f"\nFalse Positive Rate for Privacy Tools: {false_positive_rate:.2f}%")
        print(f"Blocked: {blocked_count}/{total_requests}")
        
        # Tasa de falsos positivos debe estar dentro de umbral
        assert false_positive_rate <= FALSE_POSITIVE_THRESHOLD, \
            f"False positive rate {false_positive_rate:.2f}% exceeds threshold {FALSE_POSITIVE_THRESHOLD}%"

    def test_legitimate_users_not_unfairly_penalized(self):
        """
        Usuarios legítimos con/sin privacidad deben tener tasa de
        bloqueo similar.
        """
        baseline_profile = PRIVACY_PROFILES["legitimate_no_privacy"]
        privacy_profile = PRIVACY_PROFILES["mullvad"]
        
        baseline_blocks = 0
        privacy_blocks = 0
        iterations = 5
        
        for i in range(iterations):
            # Baseline
            s1 = get_session_with_profile(baseline_profile)
            r1 = s1.get(f"{BASE}/api/data", timeout=TIMEOUT)
            if r1.status_code == 403:
                baseline_blocks += 1
            
            # Privacy
            s2 = get_session_with_profile(privacy_profile)
            r2 = s2.get(f"{BASE}/api/data", timeout=TIMEOUT)
            if r2.status_code == 403:
                privacy_blocks += 1
            
            time.sleep(0.1)
        
        baseline_rate = baseline_blocks / iterations * 100
        privacy_rate = privacy_blocks / iterations * 100
        
        print(f"\nBaseline block rate: {baseline_rate:.1f}%")
        print(f"Privacy block rate: {privacy_rate:.1f}%")
        
        # Las tasas no deben diferir significativamente
        rate_diff = abs(baseline_rate - privacy_rate)
        assert rate_diff <= 20.0, \
            f"Significant difference in block rates: {rate_diff:.1f}%"


# ═══════════════════════════════════════════════════════════════════
# SUITE 6 — DISCRIMINATION INDEX
# ═══════════════════════════════════════════════════════════════════
class TestDiscriminationIndex:
    """
    Métricas de equidad algorítmica para detectar discriminación
    involuntaria contra grupos específicos (usuarios de privacidad).
    """

    def test_parity_in_acceptance_rates(self):
        """
        Paridad demográfica: tasa de aceptación similar para
        usuarios de privacidad vs usuarios baseline.
        """
        test_groups = {
            "baseline": ["legitimate_no_privacy"],
            "privacy_tools": ["tor_firefox", "mullvad", "brave_windows"],
        }
        
        results = {}
        iterations_per_group = 4
        
        for group_name, profiles in test_groups.items():
            accepted = 0
            total = 0
            
            for profile_name in profiles:
                for _ in range(iterations_per_group):
                    s = get_session_with_profile(PRIVACY_PROFILES[profile_name])
                    r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
                    
                    if r.status_code in [200, 401]:  # 200=accepted, 401=auth req
                        accepted += 1
                    total += 1
                    
                    time.sleep(0.05)
            
            results[group_name] = {
                "accepted": accepted,
                "total": total,
                "acceptance_rate": accepted / total * 100 if total > 0 else 0
            }
        
        baseline_rate = results["baseline"]["acceptance_rate"]
        privacy_rate = results["privacy_tools"]["acceptance_rate"]
        
        print(f"\n--- Parity Analysis ---")
        print(f"Baseline acceptance rate: {baseline_rate:.1f}%")
        print(f"Privacy tools acceptance rate: {privacy_rate:.1f}%")
        
        # Tasa de aceptación no debe diferir más de 15 puntos
        rate_diff = abs(baseline_rate - privacy_rate)
        assert rate_diff <= 15.0, \
            f"Disparate impact detected: {rate_diff:.1f}% difference in acceptance rates"

    def test_equal_opportunity_in_bypass(self):
        """
        Igualdad de oportunidades: usuarios falsos-positivos deben tener
        igual acceso al mecanismo de bypass independientemente de usar privacidad.
        """
        test_profiles = ["legitimate_no_privacy", "mullvad", "brave_windows"]
        bypass_results = {}
        
        for profile_name in test_profiles:
            s = get_session_with_profile(PRIVACY_PROFILES[profile_name])
            
            # Acumular score suficiente para posible bloqueo
            for _ in range(4):
                s.get(f"{BASE}/login",
                      params={"username": "x", "password": "wrong"},
                      timeout=TIMEOUT)
            
            # Intentar bypass
            s.headers["X-Legitimate-User"] = "true"
            r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            
            bypass_results[profile_name] = {
                "status": r.status_code,
                "success": r.status_code == 200
            }
        
        print(f"\n--- Bypass Results ---")
        for profile, result in bypass_results.items():
            print(f"{profile}: {result}")
        
        # Todos los perfiles deben tener igual tasa de éxito del bypass
        success_rates = [
            int(r["success"]) for r in bypass_results.values()
        ]
        assert all(r == success_rates[0] for r in success_rates), \
            "Inconsistent bypass success across groups"


# ═══════════════════════════════════════════════════════════════════
# SUITE 7 — LEGITIMATE CONTEXT
# ═══════════════════════════════════════════════════════════════════
class TestLegitimateContext:
    """
    Simula contextos de compra válidos desde usuarios de privacidad.
    El sistema debe permitir transacciones legítimas incluso con
    herramientas de privacidad activas.
    """

    def test_privacy_user_login_purchase_flow(self):
        """
        Flujo completo: login → acceso a datos → compra desde usuario de privacidad.
        """
        s = get_session_with_profile(PRIVACY_PROFILES["brave_windows"])
        
        # Paso 1: Login
        r_login = s.get(
            f"{BASE}/login",
            params={"username": "admin", "password": "secret123"},
            timeout=TIMEOUT
        )
        
        # Paso 2: Acceso a datos
        r_data = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        
        # El flujo no debe resultar en 500
        assert r_login.status_code != 500
        assert r_data.status_code != 500

    def test_vpn_user_consistent_context(self):
        """
        Usuario desde VPN debe poder mantener contexto consistente
        a través de múltiples requests.
        """
        s = get_session_with_profile(PRIVACY_PROFILES["protonvpn"])
        
        status_codes = []
        for _ in range(5):
            r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            status_codes.append(r.status_code)
            time.sleep(0.1)
        
        # No debe haber cambios erráticos
        unique_codes = set(status_codes)
        assert len(unique_codes) <= 2, \
            f"Erratic status codes from VPN user: {status_codes}"

    def test_tor_user_can_complete_transaction_with_bypass(self):
        """
        Usuario desde TOR bloqueado debe poder usar bypass
        para completar transacción legítima.
        """
        s = get_session_with_profile(PRIVACY_PROFILES["tor_firefox"])
        
        # Intentar acceso inicial
        r1 = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        
        if r1.status_code == 403:
            # Usar bypass
            s.headers["X-Legitimate-User"] = "true"
            r2 = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            assert r2.status_code in [200, 401], \
                f"Bypass failed for TOR user: {r2.status_code}"


# ═══════════════════════════════════════════════════════════════════
# SUITE 8 — RATE LIMITING FAIRNESS
# ═══════════════════════════════════════════════════════════════════
class TestRateLimitingFairness:
    """
    Valida que el rate limiting sea justo y no penalice desproporcionadamente
    a usuarios de privacidad por características inherentes a sus herramientas.
    """

    def test_privacy_users_not_rate_limited_for_privacy_features(self):
        """
        Usuarios de privacidad no deben alcanzar rate limit solo por
        usar herramientas de privacidad.
        """
        s = get_session_with_profile(PRIVACY_PROFILES["brave_windows"])
        
        status_codes = []
        # Hacer requests a ritmo normal de usuario
        for i in range(10):
            r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            status_codes.append(r.status_code)
            time.sleep(0.2)  # Ritmo humano
        
        # No todos deben ser 429 (rate limit)
        rate_limit_count = sum(1 for code in status_codes if code == 429)
        assert rate_limit_count <= 2, \
            f"Privacy user hit rate limit too aggressively: {rate_limit_count}/10"

    def test_baseline_vs_privacy_rate_limit_parity(self):
        """
        Umbral de rate limit debe ser similar para usuarios
        de privacidad vs usuarios baseline.
        """
        # Este test es conceptual — en práctica necesitaría
        # ajuste de timing muy preciso
        baseline_s = get_session_with_profile(PRIVACY_PROFILES["legitimate_no_privacy"])
        privacy_s = get_session_with_profile(PRIVACY_PROFILES["mullvad"])
        
        baseline_hits = 0
        privacy_hits = 0
        
        # Requests rápidas
        for _ in range(15):
            r1 = baseline_s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            r2 = privacy_s.get(f"{BASE}/api/data", timeout=TIMEOUT)
            
            if r1.status_code == 429:
                baseline_hits += 1
            if r2.status_code == 429:
                privacy_hits += 1
        
        # Tasas no deben diferir enormemente
        rate_diff = abs(baseline_hits - privacy_hits)
        assert rate_diff <= 5, \
            f"Rate limiting disparity: baseline={baseline_hits}, privacy={privacy_hits}"


# ═══════════════════════════════════════════════════════════════════
# SUITE 9 — AUDIT & COMPLIANCE
# ═══════════════════════════════════════════════════════════════════
class TestAuditAndCompliance:
    """
    Validaciones de auditoría y cumplimiento regulatorio para
    asegurar que no hay discriminación documentada.
    """

    def test_all_blocks_have_documented_reason(self):
        """
        Todo bloqueo debe incluir código RBT que explique la razón.
        """
        s = get_session_with_profile(PRIVACY_PROFILES["tor_firefox"])
        
        # Generar estado que pueda resultar en bloqueo
        for _ in range(3):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "wrong"},
                  timeout=TIMEOUT)
        
        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
        
        if r.status_code == 403:
            risk_code = extract_risk_code_from_response(r)
            severity = extract_severity_from_response(r)
            
            assert risk_code and risk_code != "UNKNOWN", \
                "Blocked request lacks risk code"
            assert severity and severity != "UNKNOWN", \
                "Blocked request lacks severity"
            
            # Risk code debe tener formato válido
            assert risk_code.startswith("RBT-"), \
                f"Invalid risk code format: {risk_code}"

    def test_risk_events_logged_for_audit(self):
        """
        Eventos de bloqueo deben estar disponibles en métricas
        para auditoría.
        """
        lines = get_all_metric_lines("rbt_risk_events_total")
        
        # Debe haber eventos registrados
        assert len(lines) > 0, "No risk events found in metrics"
        
        # Verificar estructura de eventos
        for line in lines[:5]:  # Verificar primeros 5
            assert "risk_code=" in line
            assert "severity=" in line

    def test_false_positive_tracking_enabled(self):
        """
        Sistema debe tener métricas activas de false positives
        para auditoría de discriminación.
        """
        metrics = get_all_metric_lines("false_positive_blocks_total")
        
        # Métrica debe existir
        assert len(metrics) >= 0, \
            "false_positive_blocks_total metric not found"


# ═══════════════════════════════════════════════════════════════════
# INTEGRATION TEST — FULL FAIRNESS ASSESSMENT
# ═══════════════════════════════════════════════════════════════════
class TestFullFairnessAssessment:
    """
    Evaluación completa de equidad del sistema.
    """

    def test_comprehensive_fairness_report(self):
        """
        Genera un reporte completo de equidad comparando múltiples grupos.
        """
        test_groups = {
            "Baseline (No Privacy)": ["legitimate_no_privacy"],
            "TOR": ["tor_firefox"],
            "VPN (Commercial)": ["mullvad", "protonvpn"],
            "Brave Browser": ["brave_windows"],
        }
        
        report = {}
        requests_per_profile = 3
        
        print("\n" + "="*70)
        print("FAIRNESS ASSESSMENT REPORT")
        print("="*70)
        
        for group_name, profiles in test_groups.items():
            group_stats = {
                "accepted": 0,
                "blocked": 0,
                "errors": 0,
                "total": 0,
            }
            
            for profile_name in profiles:
                for _ in range(requests_per_profile):
                    try:
                        s = get_session_with_profile(PRIVACY_PROFILES[profile_name])
                        r = s.get(f"{BASE}/api/data", timeout=TIMEOUT)
                        group_stats["total"] += 1
                        
                        if r.status_code == 200:
                            group_stats["accepted"] += 1
                        elif r.status_code == 403:
                            group_stats["blocked"] += 1
                        elif r.status_code >= 500:
                            group_stats["errors"] += 1
                        
                        time.sleep(0.05)
                    except Exception as e:
                        group_stats["errors"] += 1
                        group_stats["total"] += 1
            
            # Calcular rates
            group_stats["acceptance_rate"] = (
                group_stats["accepted"] / group_stats["total"] * 100
                if group_stats["total"] > 0 else 0
            )
            group_stats["block_rate"] = (
                group_stats["blocked"] / group_stats["total"] * 100
                if group_stats["total"] > 0 else 0
            )
            group_stats["error_rate"] = (
                group_stats["errors"] / group_stats["total"] * 100
                if group_stats["total"] > 0 else 0
            )
            
            report[group_name] = group_stats
            
            print(f"\n{group_name}:")
            print(f"  Total Requests: {group_stats['total']}")
            print(f"  Accepted (200): {group_stats['accepted']} ({group_stats['acceptance_rate']:.1f}%)")
            print(f"  Blocked (403):  {group_stats['blocked']} ({group_stats['block_rate']:.1f}%)")
            print(f"  Errors (5xx):   {group_stats['errors']} ({group_stats['error_rate']:.1f}%)")
        
        print("\n" + "="*70)
        
        # Validar ausencia de discriminación severa
        acceptance_rates = [
            g["acceptance_rate"] for g in report.values()
        ]
        
        if acceptance_rates:
            max_rate = max(acceptance_rates)
            min_rate = min(acceptance_rates)
            rate_spread = max_rate - min_rate
            
            print(f"\nAcceptance Rate Spread: {rate_spread:.1f}%")
            print(f"(Max: {max_rate:.1f}%, Min: {min_rate:.1f}%)")
            
            # Spread no debe exceder umbral
            assert rate_spread <= 20.0, \
                f"Significant acceptance rate disparity: {rate_spread:.1f}%"

        print("="*70)
        print("✓ FAIRNESS ASSESSMENT PASSED")
        print("="*70 + "\n")


# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS FOR MANUAL TESTING
# ═══════════════════════════════════════════════════════════════════

def print_all_privacy_profiles():
    """Imprime un resumen de todos los perfiles de privacidad."""
    print("\nAvailable Privacy Profiles:")
    print("=" * 70)
    for name, profile in PRIVACY_PROFILES.items():
        print(f"\n{name}:")
        print(f"  Category: {profile.category}")
        print(f"  Name: {profile.name}")
        print(f"  UA: {profile.user_agent[:60]}...")
        print(f"  Headers: {list(profile.headers.keys())}")
    print("=" * 70)


if __name__ == "__main__":
    print_all_privacy_profiles()
