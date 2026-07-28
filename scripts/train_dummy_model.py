"""
Script d'entraînement avec tracking MLflow.

Usage:
    python scripts/train_dummy_model.py

Le modèle est loggué dans MLflow ET sauvegardé en joblib
pour que l'API puisse le charger.
"""

import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# ── Configuration ─────────────────────────────────────────────────────────────

OUTPUT_PATH = Path("model/anomaly_detector.joblib")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

FEATURES = [
    "duration", "protocol", "src_port", "dst_port",
    "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "fwd_pkt_len_mean", "bwd_pkt_len_mean",
    "flow_iat_mean", "flow_iat_std",
    "syn_flag", "fin_flag", "rst_flag",
]

# Hyperparamètres — c'est ces valeurs qu'on va faire varier et comparer
PARAMS = {
    "n_estimators": 100,
    "contamination": 0.10,
    "max_samples": "auto",
    "random_state": 42,
    "n_samples_normal": 2000,
    "n_samples_anomaly": 100,
}

# ── Génération des données ─────────────────────────────────────────────────────

print("Génération des données synthétiques...")

rng = np.random.default_rng(PARAMS["random_state"])

normal = rng.normal(
    loc=[0.5, 6, 40000, 443, 10, 8, 1500, 2000, 150, 200, 12, 3, 0.3, 0.3, 0.05],
    scale=[0.3, 2, 10000, 100, 5, 4, 500, 800, 50, 80, 5, 1, 0.4, 0.4, 0.2],
    size=(PARAMS["n_samples_normal"], 15),
)
normal = np.clip(normal, 0, None)

anomalies = rng.normal(
    loc=[0.01, 6, 54321, 80, 1000, 0, 64000, 0, 64, 0, 0.1, 0.01, 1, 0, 1],
    scale=[0.005, 0, 100, 0, 200, 0, 10000, 0, 5, 0, 0.05, 0.005, 0, 0, 0],
    size=(PARAMS["n_samples_anomaly"], 15),
)
anomalies = np.clip(anomalies, 0, None)

X_train = np.vstack([normal, anomalies])

# Labels réels : 0=normal, 1=anomalie
y_true = np.array(
    [0] * PARAMS["n_samples_normal"] + [1] * PARAMS["n_samples_anomaly"]
)

# ── Entraînement ───────────────────────────────────────────────────────────────

print("Entraînement du modèle...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = IsolationForest(
    n_estimators=PARAMS["n_estimators"],
    contamination=PARAMS["contamination"],
    max_samples=PARAMS["max_samples"],
    random_state=PARAMS["random_state"],
    n_jobs=-1,
)
model.fit(X_scaled)

# ── Calcul des métriques ───────────────────────────────────────────────────────

# IsolationForest : -1=anomalie → on convertit en 1, 1=normal → on convertit en 0
raw_preds = model.predict(X_scaled)
y_pred = np.where(raw_preds == -1, 1, 0)

# score_samples : plus proche de 0 = plus anormal → on normalise en [0,1]
raw_scores = model.score_samples(X_scaled)
y_scores = np.clip(0.5 - raw_scores, 0, 1)

precision = precision_score(y_true, y_pred, zero_division=0)
recall    = recall_score(y_true, y_pred, zero_division=0)
f1        = f1_score(y_true, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_true, y_scores)

print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")

# ── MLflow tracking ────────────────────────────────────────────────────────────

print("Logging dans MLflow...")

# Pointe vers le serveur MLflow local
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Crée l'experiment s'il n'existe pas
mlflow.set_experiment("anomaly-detection")

with mlflow.start_run(run_name="isolation-forest-v1"):

    # 1. Log des hyperparamètres
    mlflow.log_params(PARAMS)
    mlflow.log_param("n_features", len(FEATURES))

    # 2. Log des métriques
    mlflow.log_metrics({
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "roc_auc":   roc_auc,
    })

    # 3. Log du modèle sklearn (MLflow le sérialise lui-même)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_scaled[:1],
    )

    # 4. Log du scaler comme artefact
    scaler_path = "model/scaler.joblib"
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(scaler_path, artifact_path="scaler")

    # 5. Log de la liste des features (utile pour la reproductibilité)
    features_path = "model/features.txt"
    with open(features_path, "w") as f:
        f.write("\n".join(FEATURES))
    mlflow.log_artifact(features_path, artifact_path="metadata")

    run_id = mlflow.active_run().info.run_id
    print(f"Run ID : {run_id}")

# ── Sauvegarde joblib (pour l'API) ─────────────────────────────────────────────

bundle = {
    "model": model,
    "scaler": scaler,
    "features": FEATURES,
    "version": "2.0.0",
    "mlflow_run_id": run_id,
}
joblib.dump(bundle, OUTPUT_PATH)
print(f"Modèle sauvegardé : {OUTPUT_PATH}")
print("Termine.")
