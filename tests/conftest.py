"""
tests/conftest.py
-----------------
Fixtures compartidos + path setup para importar main.py y ml/
desde cualquier directorio de ejecucion.

Estructura del proyecto:
  D:\rbt_project\
    main.py          <- se importa con "from main import ..."
    risk_codes.py    <- se importa con "from risk_codes import ..."
    ml\
      train_model.py <- se importa con "from ml.train_model import ..."
    tests\
      conftest.py    <- este archivo
      test_sdlc.py

pytest puede ejecutarse desde D:/rbt_project/ o desde D:/rbt_project/tests/
conftest.py garantiza que el directorio raiz siempre este en sys.path.
"""

import sys
import os
import requests
import pytest
from pathlib import Path

# ── Path setup ────────────────────────────────────────────
# Resolve project root = parent of the directory containing this conftest.py
# Works whether pytest is run from the project root or from tests/
_HERE        = Path(__file__).resolve().parent   # .../tests/
_PROJECT_ROOT = _HERE.parent                      # .../rbt_project/

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Also ensure the tests/ directory itself is on the path
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Constants ─────────────────────────────────────────────
BASE = "http://localhost:8000"


# ── Fixtures ──────────────────────────────────────────────
@pytest.fixture(autouse=False)
def fresh_session():
    """Session with legit User-Agent — never blocked by the middleware."""
    s = requests.Session()
    s.headers.update({
        "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) pytest-fresh/1.0",
        "Accept-Language": "es-ES,es;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    })
    return s
