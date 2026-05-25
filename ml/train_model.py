"""
ml/train_model.py
─────────────────
Generates synthetic training data and trains a RandomForestClassifier
to detect bots based on request features.

Usage:
    python ml/train_model.py

Output:
    ml/bot_detector.pkl   ← trained model (loaded by main.py at startup)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ─────────────────────────────────────────────
# FEATURE COLUMNS (must match extract_features in main.py)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "is_headless_ua",       # 1 if UA has bot keywords
    "has_accept_language",  # 1 if Accept-Language header present
    "requests_per_minute",  # count in rate-limit window
    "current_risk_score",   # accumulated score in Redis
    "failed_logins",        # failed login attempts
    "has_legitimate_header",# 1 if X-Legitimate-User: true
]


def generate_training_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic labeled training data.
    label = 1 → bot
    label = 0 → legitimate user
    """
    rng = np.random.default_rng(seed)
    records = []

    # ── Legitimate users (50%) ────────────────────────────────
    n_legit = n_samples // 2
    for _ in range(n_legit):
        records.append({
            "is_headless_ua":        0,
            "has_accept_language":   1,
            "requests_per_minute":   rng.integers(1, 15),
            "current_risk_score":    rng.uniform(0, 10),
            "failed_logins":         rng.integers(0, 2),
            "has_legitimate_header": rng.integers(0, 2),
            "label": 0
        })

    # ── Headless bots (20%) ───────────────────────────────────
    n_headless = n_samples // 5
    for _ in range(n_headless):
        records.append({
            "is_headless_ua":        1,
            "has_accept_language":   rng.integers(0, 2),
            "requests_per_minute":   rng.integers(20, 100),
            "current_risk_score":    rng.uniform(15, 80),
            "failed_logins":         rng.integers(0, 5),
            "has_legitimate_header": 0,
            "label": 1
        })

    # ── Credential stuffing bots (15%) ────────────────────────
    n_cred = int(n_samples * 0.15)
    for _ in range(n_cred):
        records.append({
            "is_headless_ua":        rng.integers(0, 2),
            "has_accept_language":   rng.integers(0, 2),
            "requests_per_minute":   rng.integers(10, 60),
            "current_risk_score":    rng.uniform(20, 100),
            "failed_logins":         rng.integers(5, 30),
            "has_legitimate_header": 0,
            "label": 1
        })

    # ── Rate abusers (15%) ────────────────────────────────────
    n_rate = n_samples - n_legit - n_headless - n_cred
    for _ in range(n_rate):
        records.append({
            "is_headless_ua":        0,
            "has_accept_language":   1,
            "requests_per_minute":   rng.integers(80, 500),
            "current_risk_score":    rng.uniform(5, 40),
            "failed_logins":         rng.integers(0, 3),
            "has_legitimate_header": 0,
            "label": 1
        })

    df = pd.DataFrame(records)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def train(n_samples: int = 2000):
    print("=" * 55)
    print("  RBT Bot Detector — Model Training")
    print("=" * 55)

    # 1. Generate data
    print(f"\n📊 Generating {n_samples} synthetic samples...")
    df = generate_training_data(n_samples)
    print(f"   Legitimate: {(df.label == 0).sum()}  |  Bots: {(df.label == 1).sum()}")

    # 2. Save dataset for reference
    dataset_path = Path("ml/training_data.csv")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dataset_path, index=False)
    print(f"   Dataset saved → {dataset_path}")

    # 3. Split
    X = df[FEATURE_COLS].values
    y = df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train
    print("\n🤖 Training RandomForestClassifier (100 trees)...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    print("\n📈 Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Legitimate", "Bot"]))

    print("🔢 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   True Negatives  (legit→legit): {cm[0][0]}")
    print(f"   False Positives (legit→bot)  : {cm[0][1]}")
    print(f"   False Negatives (bot→legit)  : {cm[1][0]}")
    print(f"   True Positives  (bot→bot)    : {cm[1][1]}")

    # 6. Feature importance
    print("\n🔍 Feature Importance:")
    for feat, imp in sorted(zip(FEATURE_COLS, model.feature_importances_),
                             key=lambda x: x[1], reverse=True):
        bar = "█" * int(imp * 40)
        print(f"   {feat:<30} {bar} {imp:.3f}")

    # 7. Save model
    model_path = Path("ml/bot_detector.pkl")
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved → {model_path}")
    print("   Restart FastAPI to load the new model.")
    print("=" * 55)


if __name__ == "__main__":
    train()
