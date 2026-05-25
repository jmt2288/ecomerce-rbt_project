# 🛡 RBT Security Project

> **Prueba de Seguridad con Risk-Based Testing (RBT)**  
> FastAPI · Redis · Prometheus · Grafana · scikit-learn · Docker

[![Tests](https://github.com/TU_USUARIO/rbt-security/actions/workflows/tests.yml/badge.svg)](https://github.com/TU_USUARIO/rbt-security/actions)

---

## 🚀 Inicio en 1 minuto — GitHub Codespaces (sin instalar nada)

1. Haz clic en **`<> Code`** → **`Codespaces`** → **`Create codespace on main`**
2. Espera ~2 minutos — el entorno arranca automáticamente
3. Accede a los servicios desde la pestaña **Ports**:

| Servicio    | Puerto |
|-------------|--------|
| FastAPI     | 8000   |
| Grafana     | 3000   |
| Prometheus  | 9090   |

---

## 📁 Estructura

```
rbt-security/
├── main.py                    # FastAPI + middleware + ML
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .devcontainer/             # Config GitHub Codespaces
├── .github/workflows/         # CI automático
├── ml/train_model.py          # Entrena RandomForestClassifier
├── tests/test_security.py     # 6 suites, 15 tests
├── prometheus/prometheus.yml
└── grafana/provisioning/
```

---

## 🏃 Inicio local

```bash
git clone https://github.com/TU_USUARIO/rbt-security.git
cd rbt-security
docker compose up --build -d
curl http://localhost:8000/status
```

---

## 🧪 Pruebas

```bash
pip install pytest requests
pytest tests/test_security.py -v
```

---

## 📊 Métricas en Grafana (localhost:3000)

| Panel | Query |
|-------|-------|
| Requests/seg | `rate(http_requests_total[1m])` |
| Probabilidad bot ML | `bot_ml_probability` |
| Risk Score | `current_risk_score` |
| Falsos positivos | `sum(false_positive_blocks_total)` |

---

## 👥 Credenciales de prueba

- **API login:** `admin / secret123`
- **Grafana:** `admin / admin`
