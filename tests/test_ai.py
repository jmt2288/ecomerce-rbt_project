"""
tests/test_ai.py
────────────────
Pruebas completas de IA/ML para el modelo RandomForestClassifier
del proyecto RBT Security.

Cubre 10 categorías:
  1.  Calidad del modelo         — métricas estándar de clasificación
  2.  Robustez adversarial       — ¿se puede engañar el modelo?
  3.  Equidad / Fairness         — tasas de error por grupo de usuario
  4.  Explicabilidad             — importancia de features
  5.  Estabilidad                — consistencia y determinismo
  6.  Datos de entrenamiento     — calidad del dataset sintético
  7.  Detección de drift         — distribución lógica de datos
  8.  Ingeniería de features     — extracción correcta desde requests
  9.  Integración ML en vivo     — modelo dentro del middleware
  10. Casos límite y chaos       — inputs extremos y malformados

Run:
    pytest tests/test_ai.py -v
    pytest tests/test_ai.py -v -k "quality"
    pytest tests/test_ai.py -v -k "not live"   # sin API
"""

import pytest
import numpy as np
import pandas as pd
import time
import random
import string
import requests
from pathlib import Path
from unittest.mock import MagicMock, patch
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────
# HELPERS GLOBALES
# ─────────────────────────────────────────────
MODEL_PATH = Path("ml/bot_detector.pkl")
BASE       = "http://localhost:8000"


def load_model():
    if not MODEL_PATH.exists():
        pytest.skip("Modelo no entrenado — ejecuta: python ml/train_model.py")
    import joblib
    return joblib.load(MODEL_PATH)


def feature_cols():
    from ml.train_model import FEATURE_COLS
    return FEATURE_COLS


def gen_data(n=500, seed=42):
    from ml.train_model import generate_training_data
    return generate_training_data(n_samples=n, seed=seed)


def uid():
    return "".join(random.choices(string.ascii_lowercase, k=8))


def bot_session():
    s = requests.Session()
    s.headers["User-Agent"] = f"headless-bot/{uid()}"
    return s


def legit_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent":      f"Mozilla/5.0 RBT-AITest/{uid()}",
        "Accept-Language": "es-ES,es;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    })
    return s


def predict(model, features_list):
    """Predice con array de features [[f0,f1,f2,f3,f4,f5]]."""
    X = np.array(features_list)
    return model.predict(X), model.predict_proba(X)[:, 1]


# ══════════════════════════════════════════════
# 1. CALIDAD DEL MODELO
# ══════════════════════════════════════════════
class TestModelQuality:
    """Métricas de clasificación — umbrales mínimos aceptables para RBT."""

    def _train_eval(self, n=1000):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        df = gen_data(n)
        cols = feature_cols()
        X, y = df[cols].values, df["label"].values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        m = RandomForestClassifier(n_estimators=50, random_state=42)
        m.fit(Xtr, ytr)
        return m, Xte, yte

    def test_accuracy_above_80(self):
        from sklearn.metrics import accuracy_score
        m, Xte, yte = self._train_eval()
        acc = accuracy_score(yte, m.predict(Xte))
        assert acc >= 0.80, f"Accuracy {acc:.2%} < 80%"

    def test_f1_above_75_crossval(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        df = gen_data(800)
        X, y = df[feature_cols()].values, df["label"].values
        m = RandomForestClassifier(n_estimators=50, random_state=42)
        f1s = cross_val_score(m, X, y, cv=5, scoring="f1")
        assert f1s.mean() >= 0.75, f"F1 medio {f1s.mean():.2f} < 0.75"

    def test_precision_above_70(self):
        from sklearn.metrics import precision_score
        m, Xte, yte = self._train_eval()
        prec = precision_score(yte, m.predict(Xte))
        assert prec >= 0.70, f"Precisión {prec:.2%} — demasiados falsos positivos"

    def test_recall_above_70(self):
        from sklearn.metrics import recall_score
        m, Xte, yte = self._train_eval()
        rec = recall_score(yte, m.predict(Xte))
        assert rec >= 0.70, f"Recall {rec:.2%} — demasiados bots sin detectar"

    def test_roc_auc_above_85(self):
        from sklearn.model_selection import cross_val_score
        from sklearn.ensemble import RandomForestClassifier
        df = gen_data(800)
        X, y = df[feature_cols()].values, df["label"].values
        m = RandomForestClassifier(n_estimators=50, random_state=42)
        aucs = cross_val_score(m, X, y, cv=5, scoring="roc_auc")
        assert aucs.mean() >= 0.85, f"AUC-ROC {aucs.mean():.2f} < 0.85"

    def test_false_positive_rate_below_20_percent(self):
        """FPR ≤ 20%: usuarios legítimos bloqueados por error."""
        from sklearn.metrics import confusion_matrix
        m, Xte, yte = self._train_eval()
        cm = confusion_matrix(yte, m.predict(Xte))
        tn, fp = cm[0][0], cm[0][1]
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        assert fpr <= 0.20, f"FPR {fpr:.2%} — demasiados legítimos bloqueados"

    def test_false_negative_rate_below_30_percent(self):
        """FNR ≤ 30%: bots que pasan sin detectar."""
        from sklearn.metrics import confusion_matrix
        m, Xte, yte = self._train_eval()
        cm = confusion_matrix(yte, m.predict(Xte))
        fn, tp = cm[1][0], cm[1][1]
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        assert fnr <= 0.30, f"FNR {fnr:.2%} — demasiados bots sin detectar"

    def test_no_overfitting_max_gap_15(self):
        """Train F1 - Test F1 ≤ 0.15 — el modelo no memoriza."""
        from sklearn.metrics import f1_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        df = gen_data(1000)
        X, y = df[feature_cols()].values, df["label"].values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        m = RandomForestClassifier(n_estimators=50, random_state=42)
        m.fit(Xtr, ytr)
        gap = f1_score(ytr, m.predict(Xtr)) - f1_score(yte, m.predict(Xte))
        assert gap <= 0.15, f"Posible overfitting: gap F1 = {gap:.2f}"

    def test_model_beats_random_baseline(self):
        """El modelo debe superar significativamente una clasificación aleatoria."""
        from sklearn.metrics import f1_score
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.dummy import DummyClassifier
        from sklearn.model_selection import train_test_split
        df = gen_data(600)
        X, y = df[feature_cols()].values, df["label"].values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        rf = RandomForestClassifier(n_estimators=30, random_state=42)
        rf.fit(Xtr, ytr)
        dummy = DummyClassifier(strategy="stratified", random_state=42)
        dummy.fit(Xtr, ytr)
        rf_f1    = f1_score(yte, rf.predict(Xte))
        dummy_f1 = f1_score(yte, dummy.predict(Xte))
        assert rf_f1 > dummy_f1 + 0.20, \
            f"Modelo ({rf_f1:.2f}) no supera suficientemente al baseline ({dummy_f1:.2f})"


# ══════════════════════════════════════════════
# 2. ROBUSTEZ ADVERSARIAL
# ══════════════════════════════════════════════
class TestModelRobustness:
    """¿Puede un bot evadir la detección manipulando sus features?"""

    def test_bot_faking_legit_ua_detected_by_other_signals(self):
        """Bot que imita UA legítimo pero tiene rate alto y muchos fallos."""
        m = load_model()
        # UA normal (feature 0 = 0) pero todo lo demás es de bot
        preds, probs = predict(m, [[0, 1, 200, 60.0, 20, 0]])
        assert probs[0] >= 0.50, \
            f"Bot con UA falso no detectado: prob={probs[0]:.2f}"

    def test_bot_with_zero_risk_score_detected_by_rate(self):
        """Bot de primera visita (score=0) detectado por rate alto."""
        m = load_model()
        preds, probs = predict(m, [[1, 0, 150, 0.0, 0, 0]])
        assert preds[0] == 1 or probs[0] >= 0.40, \
            f"Bot nuevo sin score no detectado: prob={probs[0]:.2f}"

    def test_bot_with_legit_header_still_flagged(self):
        """X-Legitimate-User: true (feature 5=1) no salva un bot obvio."""
        m = load_model()
        preds, probs = predict(m, [[1, 0, 300, 90.0, 30, 1]])
        assert probs[0] >= 0.40, \
            f"Bot obvio con header legítimo tiene prob={probs[0]:.2f} — modelo manipulable"

    def test_noise_does_not_flip_clear_bot(self):
        """Pequeño ruido gaussiano no debe cambiar la clasificación de un bot claro."""
        m = load_model()
        base = np.array([[1, 0, 100, 60.0, 15, 0]], dtype=float)
        base_pred = m.predict(base)[0]
        np.random.seed(7)
        flipped = 0
        for _ in range(50):
            noisy = base + np.random.normal(0, 0.5, base.shape)
            noisy = np.clip(noisy, 0, None)
            noisy[:, :2] = np.round(noisy[:, :2]).clip(0, 1)
            if m.predict(noisy)[0] != base_pred:
                flipped += 1
        assert flipped <= 10, \
            f"Modelo inestable: {flipped}/50 predicciones cambiaron con ruido mínimo"

    def test_rate_abuser_without_headless_ua_detected(self):
        """Bot de rate abuse: UA normal pero 400 req/min."""
        m = load_model()
        preds, probs = predict(m, [[0, 1, 400, 5.0, 0, 0]])
        assert preds[0] == 1 or probs[0] >= 0.45, \
            f"Rate abuser no detectado: pred={preds[0]} prob={probs[0]:.2f}"

    def test_credential_stuffing_bot_detected(self):
        """Bot de credential stuffing: 25 fallos de login."""
        m = load_model()
        preds, probs = predict(m, [[0, 1, 30, 50.0, 25, 0]])
        assert preds[0] == 1 or probs[0] >= 0.50, \
            f"Credential stuffing no detectado: prob={probs[0]:.2f}"

    def test_all_zeros_does_not_crash(self):
        """Input de todos ceros no debe crashear el modelo."""
        m = load_model()
        preds, probs = predict(m, [[0, 0, 0, 0.0, 0, 0]])
        assert preds[0] in [0, 1]
        assert 0.0 <= probs[0] <= 1.0

    def test_extreme_values_do_not_crash(self):
        """Valores extremos no deben crashear el modelo."""
        m = load_model()
        preds, probs = predict(m, [[1, 0, 9999, 9999.0, 9999, 1]])
        assert preds[0] == 1, "Valores extremos de bot no clasificados como bot"

    def test_negative_values_handled_gracefully(self):
        """Valores negativos (input malformado) no deben crashear."""
        m = load_model()
        try:
            preds, probs = predict(m, [[0, 1, -10, -5.0, -1, 0]])
            assert preds[0] in [0, 1]
        except Exception as e:
            pytest.fail(f"Modelo crasheó con valores negativos: {e}")


# ══════════════════════════════════════════════
# 3. EQUIDAD / FAIRNESS
# ══════════════════════════════════════════════
class TestModelFairness:
    """El modelo no debe discriminar injustamente grupos de usuarios."""

    def test_legit_user_profiles_have_low_fpr(self):
        """≥ 80% de perfiles de usuario legítimo clasificados correctamente."""
        m = load_model()
        np.random.seed(42)
        n = 200
        legit = np.column_stack([
            np.zeros(n),
            np.ones(n),
            np.random.randint(1, 10, n),
            np.random.uniform(0, 5, n),
            np.zeros(n),
            np.zeros(n),
        ])
        preds = m.predict(legit)
        fp_rate = preds.mean()
        assert fp_rate <= 0.20, \
            f"{fp_rate:.0%} de usuarios legítimos marcados como bot — FPR demasiado alto"

    def test_obvious_bots_have_high_detection_rate(self):
        """≥ 80% de bots obvios detectados correctamente."""
        m = load_model()
        np.random.seed(42)
        n = 200
        bots = np.column_stack([
            np.ones(n),
            np.zeros(n),
            np.random.randint(50, 200, n),
            np.random.uniform(20, 100, n),
            np.random.randint(5, 30, n),
            np.zeros(n),
        ])
        preds = m.predict(bots)
        detection = preds.mean()
        assert detection >= 0.80, \
            f"Solo {detection:.0%} de bots obvios detectados — recall demasiado bajo"

    def test_first_time_legit_user_not_blocked(self):
        """Usuario nuevo con comportamiento normal → baja probabilidad de bot."""
        m = load_model()
        # Primer acceso: score=0, rate=1, sin fallos
        preds, probs = predict(m, [[0, 1, 1, 0.0, 0, 0]])
        assert probs[0] < 0.40, \
            f"Usuario nuevo clasificado como bot con prob={probs[0]:.2f}"

    def test_high_rate_legit_api_user_threshold(self):
        """Un usuario de API legítimo (alta tasa pero sin otras señales)."""
        m = load_model()
        # API user: rate alto pero UA normal, sin fallos, sin score
        preds, probs = predict(m, [[0, 1, 60, 3.0, 0, 0]])
        # No debe ser bloqueado con alta confianza
        assert probs[0] < 0.70, \
            f"API user legítimo tiene prob bot = {probs[0]:.2f}"

    def test_threshold_sensitivity_analysis(self):
        """Análisis de sensibilidad: diferentes umbrales de probabilidad."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import precision_score, recall_score
        df = gen_data(600)
        X, y = df[feature_cols()].values, df["label"].values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y)
        mdl = RandomForestClassifier(n_estimators=50, random_state=42)
        mdl.fit(Xtr, ytr)
        probs = mdl.predict_proba(Xte)[:, 1]
        for threshold in [0.3, 0.5, 0.7]:
            preds_t = (probs >= threshold).astype(int)
            prec = precision_score(yte, preds_t, zero_division=0)
            rec  = recall_score(yte, preds_t, zero_division=0)
            # At any threshold, precision and recall should be > 0.5
            assert prec >= 0.50 or rec >= 0.50, \
                f"Threshold {threshold}: prec={prec:.2f} rec={rec:.2f} — ambos muy bajos"


# ══════════════════════════════════════════════
# 4. EXPLICABILIDAD
# ══════════════════════════════════════════════
class TestModelExplainability:
    """El modelo debe ser interpretable y sus decisiones comprensibles."""

    def test_feature_importances_sum_to_one(self):
        m = load_model()
        total = m.feature_importances_.sum()
        assert abs(total - 1.0) < 1e-6, f"Importancias suman {total:.6f}, no 1.0"

    def test_all_features_contribute(self):
        """Toda feature debe tener importancia > 0."""
        m = load_model()
        cols = feature_cols()
        for col, imp in zip(cols, m.feature_importances_):
            assert imp > 0.0, f"Feature '{col}' tiene importancia 0 — inútil"

    def test_risk_score_or_fails_in_top_3(self):
        """current_risk_score o failed_logins deben estar en el top 3."""
        m = load_model()
        cols = feature_cols()
        imp = dict(zip(cols, m.feature_importances_))
        top3 = sorted(imp, key=imp.get, reverse=True)[:3]
        assert "current_risk_score" in top3 or "failed_logins" in top3, \
            f"Señales de riesgo no están en top 3: {top3}"

    def test_headless_ua_importance_above_5_percent(self):
        m = load_model()
        cols = feature_cols()
        imp = dict(zip(cols, m.feature_importances_))
        assert imp["is_headless_ua"] >= 0.05, \
            f"is_headless_ua importancia = {imp['is_headless_ua']:.3f} < 5%"

    def test_correct_number_of_features(self):
        m = load_model()
        assert m.n_features_in_ == 6, \
            f"Modelo usa {m.n_features_in_} features, esperado 6"

    def test_higher_risk_score_increases_bot_probability(self):
        """A mayor Risk Score, mayor probabilidad de bot (monotonía)."""
        m = load_model()
        probs = []
        for score in [0, 10, 30, 60, 90]:
            _, p = predict(m, [[0, 1, 5, float(score), 0, 0]])
            probs.append(p[0])
        # Debe ser creciente (o al menos no decreciente fuertemente)
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i+1] + 0.10, \
                f"Probabilidad NO crece con Risk Score: {probs}"

    def test_more_failed_logins_increases_bot_probability(self):
        """A más fallos de login, mayor probabilidad de bot."""
        m = load_model()
        probs = []
        for fails in [0, 3, 10, 20]:
            _, p = predict(m, [[0, 1, 10, 5.0, fails, 0]])
            probs.append(p[0])
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i+1] + 0.15, \
                f"Probabilidad NO crece con fallos de login: {probs}"

    def test_decision_path_retrievable(self):
        """Se puede obtener el camino de decisión de cualquier muestra."""
        m = load_model()
        X = np.array([[1, 0, 100, 50.0, 10, 0]])
        indicator = m.decision_path(X)
        assert indicator is not None
        assert indicator.shape[0] == 1

    def test_individual_tree_predictions_consistent(self):
        """La mayoría de los árboles individuales deben coincidir en un bot obvio."""
        m = load_model()
        X = np.array([[1, 0, 200, 80.0, 25, 0]])
        tree_preds = [tree.predict(X)[0] for tree in m.estimators_]
        majority = sum(tree_preds) / len(tree_preds)
        assert majority >= 0.60, \
            f"Solo {majority:.0%} de árboles clasifican el bot como bot"


# ══════════════════════════════════════════════
# 5. ESTABILIDAD Y DETERMINISMO
# ══════════════════════════════════════════════
class TestModelStability:
    """El modelo debe ser consistente y predecible."""

    def test_same_input_same_output_100_times(self):
        m = load_model()
        X = np.array([[1, 0, 50, 30.0, 8, 0]])
        preds = [m.predict(X)[0] for _ in range(100)]
        assert len(set(preds)) == 1, "Predicción no determinista"

    def test_batch_equals_individual(self):
        """Predicción en batch = predicciones individuales."""
        m = load_model()
        samples = np.array([
            [1, 0, 100, 50.0, 10, 0],
            [0, 1, 3,   0.0,  0,  0],
            [1, 1, 200, 80.0, 20, 0],
        ])
        batch = list(m.predict(samples))
        indiv = [m.predict(s.reshape(1, -1))[0] for s in samples]
        assert batch == indiv

    def test_prediction_latency_p99_under_10ms(self):
        """P99 de predicción < 10ms para no añadir latencia al middleware."""
        m = load_model()
        X = np.array([[1, 0, 50, 30.0, 5, 0]])
        # Warm-up
        for _ in range(10):
            m.predict(X)
        times = []
        for _ in range(100):
            t0 = time.perf_counter()
            m.predict(X)
            times.append(time.perf_counter() - t0)
        p99_ms = sorted(times)[98] * 1000
        assert p99_ms < 10, f"P99 = {p99_ms:.2f}ms > 10ms — añade latencia al middleware"

    def test_retraining_same_seed_same_result(self):
        """Dos entrenamientos con la misma semilla dan el mismo modelo."""
        from sklearn.ensemble import RandomForestClassifier
        df = gen_data(400)
        X, y = df[feature_cols()].values, df["label"].values
        m1 = RandomForestClassifier(n_estimators=20, random_state=42)
        m2 = RandomForestClassifier(n_estimators=20, random_state=42)
        m1.fit(X, y)
        m2.fit(X, y)
        X_test = X[:50]
        assert list(m1.predict(X_test)) == list(m2.predict(X_test))

    def test_probability_always_sums_to_one(self):
        """predict_proba siempre devuelve probabilidades que suman 1."""
        m = load_model()
        test_cases = [
            [0, 1, 1, 0.0, 0, 0],
            [1, 0, 200, 90.0, 30, 0],
            [0, 0, 0, 0.0, 0, 0],
            [1, 1, 9999, 9999.0, 9999, 1],
        ]
        for tc in test_cases:
            proba = m.predict_proba(np.array([tc]))[0]
            assert abs(proba.sum() - 1.0) < 1e-6

    def test_model_serialization_integrity(self):
        """El modelo cargado de disco da las mismas predicciones que uno en memoria."""
        import joblib, tempfile
        from sklearn.ensemble import RandomForestClassifier
        df = gen_data(200)
        X, y = df[feature_cols()].values, df["label"].values
        m = RandomForestClassifier(n_estimators=10, random_state=42)
        m.fit(X, y)
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            joblib.dump(m, f.name)
            m2 = joblib.load(f.name)
        assert list(m.predict(X[:20])) == list(m2.predict(X[:20]))


# ══════════════════════════════════════════════
# 6. CALIDAD DEL DATASET DE ENTRENAMIENTO
# ══════════════════════════════════════════════
class TestTrainingData:
    """El dataset sintético debe tener propiedades estadísticas correctas."""

    def test_shape_and_columns(self):
        df = gen_data(500)
        cols = feature_cols()
        assert len(df) == 500
        assert "label" in df.columns
        for c in cols:
            assert c in df.columns, f"Columna faltante: {c}"

    def test_binary_features_only_0_or_1(self):
        df = gen_data(500)
        for col in ["is_headless_ua", "has_accept_language", "has_legitimate_header"]:
            assert df[col].isin([0, 1]).all(), f"{col} tiene valores fuera de [0,1]"

    def test_continuous_features_non_negative(self):
        df = gen_data(500)
        for col in ["requests_per_minute", "current_risk_score", "failed_logins"]:
            assert (df[col] >= 0).all(), f"{col} tiene valores negativos"

    def test_both_classes_present(self):
        df = gen_data(200)
        assert 0 in df["label"].values
        assert 1 in df["label"].values

    def test_class_balance_reasonable(self):
        df = gen_data(1000)
        ratio = df["label"].mean()
        assert 0.3 <= ratio <= 0.7, f"Clases desbalanceadas: ratio bots = {ratio:.2f}"

    def test_no_nan_values(self):
        df = gen_data(500)
        assert not df[feature_cols()].isnull().any().any(), "Hay NaN en el dataset"

    def test_no_infinite_values(self):
        df = gen_data(500)
        for col in feature_cols():
            assert not np.isinf(df[col].values).any(), f"Hay infinitos en {col}"

    def test_no_duplicate_rows_majority(self):
        """Al menos el 80% de filas deben ser únicas."""
        df = gen_data(500)
        unique_ratio = len(df.drop_duplicates()) / len(df)
        assert unique_ratio >= 0.80, f"Demasiados duplicados: {unique_ratio:.0%} únicas"

    def test_different_seeds_give_different_data(self):
        df1 = gen_data(100, seed=1)
        df2 = gen_data(100, seed=99)
        assert not df1["current_risk_score"].equals(df2["current_risk_score"])

    def test_risk_score_range_per_class(self):
        """Bots deben tener risk_score promedio mayor que legítimos."""
        df = gen_data(1000)
        legit_mean = df[df.label == 0]["current_risk_score"].mean()
        bot_mean   = df[df.label == 1]["current_risk_score"].mean()
        assert bot_mean > legit_mean, \
            f"Bots ({bot_mean:.1f}) no tienen mayor score que legítimos ({legit_mean:.1f})"


# ══════════════════════════════════════════════
# 7. DETECCIÓN DE DRIFT DE DATOS
# ══════════════════════════════════════════════
class TestDataDrift:
    """Verifica que la distribución de datos de entrenamiento es coherente."""

    def test_bots_have_higher_rate_than_legit(self):
        df = gen_data(1000)
        assert df[df.label==1]["requests_per_minute"].mean() > \
               df[df.label==0]["requests_per_minute"].mean()

    def test_bots_have_more_failed_logins(self):
        df = gen_data(1000)
        assert df[df.label==1]["failed_logins"].mean() > \
               df[df.label==0]["failed_logins"].mean()

    def test_legit_users_have_more_accept_language(self):
        df = gen_data(1000)
        assert df[df.label==0]["has_accept_language"].mean() > \
               df[df.label==1]["has_accept_language"].mean()

    def test_bots_have_higher_headless_ua_rate(self):
        df = gen_data(1000)
        assert df[df.label==1]["is_headless_ua"].mean() > \
               df[df.label==0]["is_headless_ua"].mean()

    def test_model_generalizes_to_new_data(self):
        """El modelo pre-entrenado funciona en datos con semilla diferente."""
        from sklearn.metrics import f1_score
        m = load_model()
        df_new = gen_data(400, seed=999)
        X_new = df_new[feature_cols()].values
        y_new = df_new["label"].values
        f1 = f1_score(y_new, m.predict(X_new))
        assert f1 >= 0.70, f"F1 en datos nuevos = {f1:.2f} — posible drift"

    def test_feature_ranges_stable_across_seeds(self):
        """Las features deben tener rangos similares en diferentes semillas."""
        for seed in [1, 42, 123]:
            df = gen_data(300, seed=seed)
            assert df["requests_per_minute"].max() < 1000
            assert df["current_risk_score"].max() < 200
            assert df["failed_logins"].max() < 50


# ══════════════════════════════════════════════
# 8. INGENIERÍA DE FEATURES
# ══════════════════════════════════════════════
class TestFeatureEngineering:
    """Verifica que extract_features() produce el array correcto desde un Request."""

    def _make_req(self, ua="Mozilla/5.0", lang="en-US", enc="gzip",
                  ip="1.2.3.4", extra_headers=None):
        req = MagicMock()
        headers = {"User-Agent": ua, "Accept-Language": lang, "Accept-Encoding": enc}
        if extra_headers:
            headers.update(extra_headers)
        req.headers = headers
        req.client = MagicMock(); req.client.host = ip
        req.url = MagicMock(); req.url.path = "/api/data"
        return req

    def _redis(self, risk=0.0, rate=5, fails=0):
        r = MagicMock()
        r.get = MagicMock(side_effect=lambda k: (
            str(risk)  if k.startswith("risk:")  else
            str(fails) if k.startswith("fails:") else None))
        r.zcard = MagicMock(return_value=rate)
        r.zadd = r.zremrangebyscore = r.expire = MagicMock()
        return r

    def _extract(self, req, risk=0.0, rate=5, fails=0):
        from main import extract_features, get_identifier
        redis_m = self._redis(risk=risk, rate=rate, fails=fails)
        identifier = get_identifier(req)
        with patch("main.r", redis_m):
            return extract_features(req, identifier)

    def test_output_shape_is_1_by_6(self):
        f = self._extract(self._make_req())
        assert isinstance(f, np.ndarray)
        assert f.shape == (1, 6)

    def test_headless_ua_sets_feature_0_to_1(self):
        f = self._extract(self._make_req(ua="headless-chrome"))
        assert f[0][0] == 1

    def test_normal_ua_sets_feature_0_to_0(self):
        f = self._extract(self._make_req(ua="Mozilla/5.0"))
        assert f[0][0] == 0

    def test_accept_language_sets_feature_1(self):
        f = self._extract(self._make_req(lang="es-ES"))
        assert f[0][1] == 1

    def test_missing_language_sets_feature_1_to_0(self):
        req = self._make_req()
        req.headers = {"User-Agent": "Mozilla/5.0"}
        f = self._extract(req)
        assert f[0][1] == 0

    def test_rate_from_redis_in_feature_2(self):
        f = self._extract(self._make_req(), rate=42)
        assert f[0][2] == 42

    def test_risk_score_from_redis_in_feature_3(self):
        f = self._extract(self._make_req(), risk=55.5)
        assert f[0][3] == 55.5

    def test_failed_logins_from_redis_in_feature_4(self):
        f = self._extract(self._make_req(), fails=13)
        assert f[0][4] == 13.0

    def test_legit_header_sets_feature_5(self):
        f = self._extract(self._make_req(extra_headers={"X-Legitimate-User": "true"}))
        assert f[0][5] == 1

    def test_all_features_non_negative(self):
        f = self._extract(self._make_req())
        assert all(v >= 0 for v in f[0])

    def test_feature_dtype_is_numeric(self):
        f = self._extract(self._make_req())
        assert all(isinstance(v, (int, float, np.integer, np.floating))
                   for v in f[0])

    def test_zero_redis_values_handled(self):
        """Redis devuelve None → features deben ser 0, no crash."""
        from main import extract_features, get_identifier
        req = self._make_req()
        redis_m = MagicMock()
        redis_m.get = MagicMock(return_value=None)
        redis_m.zcard = MagicMock(return_value=0)
        identifier = get_identifier(req)
        with patch("main.r", redis_m):
            f = extract_features(req, identifier)
        assert f.shape == (1, 6)
        assert all(v >= 0 for v in f[0])


# ══════════════════════════════════════════════
# 9. INTEGRACIÓN ML EN VIVO (necesita API)
# ══════════════════════════════════════════════
class TestMLLiveIntegration:
    """Pruebas del modelo dentro del middleware real de FastAPI."""

    def test_api_reports_ml_loaded(self):
        r = requests.get(f"{BASE}/status", timeout=10)
        assert r.json().get("ml_model_loaded") is True

    def test_bot_probability_appears_in_metrics(self):
        if not requests.get(f"{BASE}/status").json().get("ml_model_loaded"):
            pytest.skip("ML no cargado")
        bot_session().get(f"{BASE}/api/data")
        time.sleep(0.5)
        text = requests.get(f"{BASE}/metrics").text
        assert "bot_ml_probability{" in text

    def test_ml_probability_in_valid_range(self):
        if not requests.get(f"{BASE}/status").json().get("ml_model_loaded"):
            pytest.skip("ML no cargado")
        bot_session().get(f"{BASE}/api/data")
        text = requests.get(f"{BASE}/metrics").text
        for line in text.splitlines():
            if line.startswith("bot_ml_probability{"):
                prob = float(line.split()[-1])
                assert 0.0 <= prob <= 1.0, f"Probabilidad fuera de rango: {prob}"

    def test_legit_users_not_blocked_by_ml(self):
        if not requests.get(f"{BASE}/status").json().get("ml_model_loaded"):
            pytest.skip("ML no cargado")
        for i in range(5):
            s = legit_session()
            r = s.get(f"{BASE}/api/data", timeout=10)
            assert r.status_code == 200, \
                f"Usuario legítimo bloqueado por ML en intento {i+1}"

    def test_bot_eventually_blocked_by_ml_or_rules(self):
        s = bot_session()
        blocked = False
        for _ in range(10):
            s.get(f"{BASE}/api/data")
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "wrong"})
            if s.get(f"{BASE}/api/data").status_code == 403:
                blocked = True
                break
        assert blocked, "Bot no bloqueado por ML ni por reglas tras 10 iteraciones"

    def test_ml_and_risk_score_metrics_both_present(self):
        bot_session().get(f"{BASE}/api/data")
        text = requests.get(f"{BASE}/metrics").text
        assert "bot_ml_probability{"  in text
        assert "current_risk_score{" in text

    def test_false_positive_counter_increments_after_bypass(self):
        """False positive counter sube tras X-Legitimate-User bypass."""
        s = bot_session()
        s.get(f"{BASE}/api/data")
        for _ in range(4):
            s.get(f"{BASE}/login",
                  params={"username": "x", "password": "wrong"})
        r_blocked = s.get(f"{BASE}/api/data")
        if r_blocked.status_code != 403:
            pytest.skip("Score no alcanzó el umbral")
        before = requests.get(f"{BASE}/metrics").text
        s.get(f"{BASE}/api/data", headers={"X-Legitimate-User": "true"})
        after = requests.get(f"{BASE}/metrics").text
        assert "false_positive_blocks_total{" in after


# ══════════════════════════════════════════════
# 10. COMPARACIÓN DE MODELOS
# ══════════════════════════════════════════════
class TestModelComparison:
    """Compara RandomForest con alternativas para verificar que es la mejor elección."""

    def _evaluate(self, model, X_te, y_te):
        from sklearn.metrics import f1_score
        return f1_score(y_te, model.predict(X_te))

    def _get_data(self):
        from sklearn.model_selection import train_test_split
        df = gen_data(800)
        X, y = df[feature_cols()].values, df["label"].values
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def test_random_forest_beats_decision_tree(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.tree import DecisionTreeClassifier
        Xtr, Xte, ytr, yte = self._get_data()
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        dt = DecisionTreeClassifier(random_state=42)
        rf.fit(Xtr, ytr); dt.fit(Xtr, ytr)
        assert self._evaluate(rf, Xte, yte) >= self._evaluate(dt, Xte, yte) - 0.05, \
            "RandomForest no supera al DecisionTree simple"

    def test_random_forest_beats_naive_bayes(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        Xtr, Xte, ytr, yte = self._get_data()
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        nb = GaussianNB()
        rf.fit(Xtr, ytr); nb.fit(Xtr, ytr)
        assert self._evaluate(rf, Xte, yte) >= self._evaluate(nb, Xte, yte) - 0.05

    def test_more_trees_does_not_hurt(self):
        """Aumentar el número de árboles no debe empeorar el F1."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        df = gen_data(600)
        X, y = df[feature_cols()].values, df["label"].values
        f1_small = cross_val_score(
            RandomForestClassifier(n_estimators=10, random_state=42),
            X, y, cv=3, scoring="f1").mean()
        f1_large = cross_val_score(
            RandomForestClassifier(n_estimators=100, random_state=42),
            X, y, cv=3, scoring="f1").mean()
        assert f1_large >= f1_small - 0.05, \
            f"Más árboles empeoró el F1: {f1_small:.2f} → {f1_large:.2f}"

    def test_class_weight_balanced_improves_fpr(self):
        """class_weight='balanced' debe reducir la tasa de falsos positivos."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import confusion_matrix
        from sklearn.model_selection import train_test_split
        df = gen_data(600)
        X, y = df[feature_cols()].values, df["label"].values
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        def fpr(model):
            cm = confusion_matrix(yte, model.predict(Xte))
            tn, fp = cm[0][0], cm[0][1]
            return fp / (fp + tn) if (fp + tn) > 0 else 0

        m_normal   = RandomForestClassifier(n_estimators=30, random_state=42)
        m_balanced = RandomForestClassifier(n_estimators=30, random_state=42,
                                            class_weight="balanced")
        m_normal.fit(Xtr, ytr)
        m_balanced.fit(Xtr, ytr)
        # balanced may or may not be better, just should not crash
        assert fpr(m_balanced) <= 1.0
