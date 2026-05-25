#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_tests.py  --  Master Test Runner for RBT Security
======================================================

Run with PYTHON, not pytest:
    python run_tests.py              # default: all RBT cases (CP-01 to CP-09)
    python run_tests.py --cp 01      # specific case CP-01
    python run_tests.py --cp 01 02   # multiple cases CP-01, CP-02
    python run_tests.py --quick      # CP-01, CP-05, CP-06
    python run_tests.py --all        # all cases
    python run_tests.py --report     # all + HTML reports in reports/
    python run_tests.py --grafana    # generate Grafana traffic

WRONG: pytest run_tests.py   (has no test_ functions -- 0 items collected)
RIGHT: python run_tests.py
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

BASE       = "http://localhost:8000"
PROMETHEUS = "http://localhost:9090"
GRAFANA    = "http://localhost:3000"

# Force UTF-8 output on Windows so print() works everywhere
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
    # Enable ANSI color support in Windows 10+ console
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(
            ctypes.windll.kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass

G  = "\033[92m"   # green
R  = "\033[91m"   # red
Y  = "\033[93m"   # yellow
C  = "\033[96m"   # cyan
B  = "\033[1m"    # bold
RS = "\033[0m"    # reset


# =====================================================================
# RBT TEST CASES DEFINITION
# =====================================================================
RBT_CASES = {
    "01": {
        "id": "CP-01",
        "requisitos": "RF1, RNF3",
        "escenario": "Ataque masivo de Credential Stuffing",
        "tecnica": "Threat-Based Testing",
        "tests": ["tests/test_ai.py", "tests/test_security.py"],
    },
    "02": {
        "id": "CP-02",
        "requisitos": "RF2, RNF3",
        "escenario": "Scraping Altamente Distribuido",
        "tecnica": "Adversarial ML Testing",
        "tests": ["tests/test_scraping.py"],
    },
    "03": {
        "id": "CP-03",
        "requisitos": "RF3",
        "escenario": "Integridad de Telemetría y Logs",
        "tecnica": "Data Integrity / Boundary Value Analysis",
        "tests": ["tests/test_risk_codes.py", "tests/test_integration.py", "tests/test_e2e.py"],
    },
    "04": {
        "id": "CP-04",
        "requisitos": "RF4",
        "escenario": "Intento de Evasión de API",
        "tecnica": "Exploit Simulation / Penetration Testing",
        "tests": ["tests/test_penetration.py"],
    },
    "05": {
        "id": "CP-05",
        "requisitos": "RNF1",
        "escenario": "Carga Concurrente y Latencia",
        "tecnica": "Stress & Performance Testing",
        "tests": ["tests/test_load.py"],
    },
    "06": {
        "id": "CP-06",
        "requisitos": "RNF2",
        "escenario": "Evaluación Estricta de Falsos Positivos",
        "tecnica": "Usability / Statistical Dataset Testing",
        "tests": ["tests/test_unit_deep.py"],
    },
    "07": {
        "id": "CP-07",
        "requisitos": "RNF4",
        "escenario": "Verificación de Regresión en CI/CD",
        "tecnica": "Automated Regression Testing",
        "tests": ["tests/test_sdlc.py"],
    },
    "08": {
        "id": "CP-08",
        "requisitos": "RE1",
        "escenario": "Pruebas de Sesgo Involuntario",
        "tecnica": "Algorithmic Fairness Testing",
        "tests": ["tests/test_bias_fairness.py"],
    },
    "09": {
        "id": "CP-09",
        "requisitos": "RE2",
        "escenario": "Flujo de Escape Transparente (MFA)",
        "tecnica": "UX Security Integration Testing",
        "tests": ["tests/test_e2e.py"],
    },
}


# =====================================================================
# SERVICE CHECKS
# =====================================================================
def reachable(url):
    try:
        urllib.request.urlopen(url, timeout=4)
        return True
    except Exception:
        return False


def check_services():
    print(f"\n{C}Checking services...{RS}")
    api  = reachable(f"{BASE}/")
    prom = reachable(f"{PROMETHEUS}/-/healthy")
    graf = reachable(f"{GRAFANA}/api/health")

    tag = lambda ok: f"{G}UP {RS}" if ok else f"{R}-- {RS}"
    print(f"  {tag(api)}  FastAPI    http://localhost:8000")
    print(f"  {tag(prom)}  Prometheus http://localhost:9090")
    print(f"  {tag(graf)}  Grafana    http://localhost:3000")

    if api:
        try:
            d = json.loads(urllib.request.urlopen(f"{BASE}/status", timeout=4).read())
            ml = d.get("ml_model_loaded", False)
            ml_tag = f"{G}loaded{RS}" if ml else f"{Y}NOT loaded -- docker compose up --build -d{RS}"
            print(f"  {'OK ' if ml else 'WRN'}  ML Model   {ml_tag}")
        except Exception:
            pass
    print()
    return api, prom, graf


# =====================================================================
# PYTEST RUNNER
# =====================================================================
def run_rbt_case(case_data, report=False, no_soak=False):
    """Run tests for a specific RBT case."""
    case_id = case_data["id"]
    tests = case_data["tests"]
    
    # Filter existing test files
    found_tests = [t for t in tests if Path(t).exists()]
    if not found_tests:
        return None  # Skip if no tests found
    
    cmd = [sys.executable, "-m", "pytest"] + found_tests + ["-v", "--tb=short"]
    
    if no_soak:
        cmd += ["-k", "not sustained_2_minutes"]
    
    # Add special flags for certain test types
    if "load" in found_tests[0] or "penetration" in found_tests[0] or "e2e" in found_tests[0]:
        cmd += ["-s"]
    
    if report:
        os.makedirs("reports", exist_ok=True)
        report_name = f"{case_id.replace('-', '_').lower()}"
        cmd += [f"--html=reports/{report_name}.html", "--self-contained-html"]
    
    try:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode == 0
    except Exception as e:
        print(f"{R}Error running {case_id}: {e}{RS}")
        return False


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        prog="python run_tests.py",
        description="RBT Security -- Master Test Runner (Ordered by Cases)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                   (all cases CP-01 to CP-09)
  python run_tests.py --cp 01           (CP-01 only)
  python run_tests.py --cp 01 05 06     (CP-01, CP-05, CP-06)
  python run_tests.py --quick           (CP-01, CP-05, CP-06 - quick suite)
  python run_tests.py --all --report    (all cases + HTML reports)
  python run_tests.py --no-api          (unit tests without Docker)
  python run_tests.py --grafana         (generate Grafana traffic)

RIGHT: python run_tests.py
        """,
    )

    # -- Case selection --
    parser.add_argument("--cp", nargs="+", type=str, 
                        help="Specific cases (e.g., --cp 01 02 03)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick suite: CP-01, CP-05, CP-06")
    parser.add_argument("--all", action="store_true",
                        help="All cases (CP-01 to CP-09)")
    parser.add_argument("--no-api", action="store_true",
                        help="Unit tests only (CP-06)")

    # -- Options --
    parser.add_argument("--no-soak", action="store_true",
                        help="Skip sustained load tests")
    parser.add_argument("--report", action="store_true",
                        help="Generate HTML reports in reports/")
    parser.add_argument("--grafana", action="store_true",
                        help="Generate Grafana traffic")

    args = parser.parse_args()

    print(f"\n{B}RBT Security -- Ordered Test Execution{RS}")
    print("-" * 70)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}")
    print(f"  Dir     : {Path.cwd()}")

    api_ok, prom_ok, graf_ok = check_services()

    # Determine which cases to run
    cases_to_run = []
    
    if args.cp:
        # Specific cases
        for cp_num in args.cp:
            cp_num = cp_num.zfill(2)
            if cp_num in RBT_CASES:
                cases_to_run.append(cp_num)
    elif args.quick:
        cases_to_run = ["01", "05", "06"]
    elif args.no_api:
        cases_to_run = ["06"]
    elif args.all or (not args.cp and not args.quick and not args.no_api and not args.grafana):
        # Default: all cases
        cases_to_run = sorted(RBT_CASES.keys())

    if not cases_to_run:
        print(f"{Y}No cases selected -- use --help to see options.{RS}")
        sys.exit(0)

    # Check API availability
    cases_need_api = ["01", "02", "03", "04", "05", "07", "08", "09"]
    if not api_ok and any(c in cases_to_run for c in cases_need_api):
        print(f"\n{R}API not reachable at http://localhost:8000{RS}")
        print("  Start with: docker compose up -d")
        print("  Or filter to: python run_tests.py --no-api")
        sys.exit(1)

    # Print test plan
    print(f"\n{B}{C}{'=' * 70}{RS}")
    print(f"{B}  TEST PLAN - CASOS DE PRUEBA RBT{RS}")
    print(f"{B}{C}{'=' * 70}{RS}\n")
    
    print(f"{'ID':<8} {'Requisito':<12} {'Escenario':<40} {'Técnica':<30}")
    print("-" * 90)
    
    results = {}
    t0 = time.time()

    for case_num in cases_to_run:
        if case_num not in RBT_CASES:
            continue
        
        case = RBT_CASES[case_num]
        print(f"{case['id']:<8} {case['requisitos']:<12} {case['escenario']:<40} {case['tecnica']:<30}")
        
        # Print test files
        print(f"        Tests: {', '.join(case['tests'])}\n")
        
        # Run the case
        print(f"{B}{C}{'=' * 70}{RS}")
        print(f"{B}  Ejecutando {case['id']}: {case['escenario']}{RS}")
        print(f"{B}{C}{'=' * 70}{RS}\n")
        
        report = args.report
        passed = run_rbt_case(case, report=report, no_soak=args.no_soak)
        results[case['id']] = {
            "passed": passed,
            "requisitos": case['requisitos'],
            "escenario": case['escenario'],
            "tecnica": case['tecnica'],
        }

    # Generate Grafana traffic if requested
    if args.grafana and api_ok and "01" in cases_to_run:
        print(f"\n{C}Generating Grafana traffic...{RS}")
        from tests.test_security import TestLoadGeneration
        try:
            TestLoadGeneration().generate_load()
        except Exception:
            pass

    # Print summary
    elapsed = time.time() - t0
    print(f"\n{'=' * 90}")
    print(f"{B}  RESUMEN DE EJECUCIÓN ({elapsed:.0f}s){RS}")
    print(f"{'=' * 90}\n")
    
    print(f"{'ID':<8} {'Requisito':<12} {'Resultado':<10} {'Escenario':<40}")
    print("-" * 70)
    
    all_passed = True
    for case_id, result in sorted(results.items()):
        if result["passed"] is None:
            status = f"{Y}SKIP{RS}"
        elif result["passed"]:
            status = f"{G}✓ PASS{RS}"
        else:
            status = f"{R}✗ FAIL{RS}"
            all_passed = False
        
        print(f"{case_id:<8} {result['requisitos']:<12} {status:<10} {result['escenario']:<40}")

    print()
    if args.report:
        print(f"  {C}Reports : reports/{RS}")
    if api_ok:
        print(f"  {C}Grafana : http://localhost:3000{RS}")
        print(f"  {C}Prometheus: http://localhost:9090{RS}")
    print()

    if all_passed and results:
        print(f"{G}{B}✓ Todos los casos de prueba aprobados.{RS}\n")
    elif results:
        print(f"{R}✗ Algunos casos de prueba fallaron -- revisa el output arriba.{RS}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
