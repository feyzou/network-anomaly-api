import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

INPUT_PATH  = Path("C:/Users/pc/Documents/ML-projects/network-anomaly-api/data/processed/cicids_processed.csv")
OUTPUT_PATH = Path("model/anomaly_detector.joblib")

FEATURES = [
    "duration", "protocol", "src_port", "dst_port",
    "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "fwd_pkt_len_mean", "bwd_pkt_len_mean",
    "flow_iat_mean", "flow_iat_std",
    "syn_flag", "fin_flag", "rst_flag",
]

PARAMS = {
    "n_estimators": 100,
    "random_state": 42,
    "test_size": 0.2,
}

print("Chargement des données...")
df = pd.read_csv(INPUT_PATH)
print(f"  Shape : {df.shape}")

X = df[FEATURES].values
y = df["label"].values

print("Split train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=PARAMS["test_size"],
    random_state=PARAMS["random_state"],
    stratify=y,
)
print(f"  Train : {X_train.shape[0]} lignes")
print(f"  Test  : {X_test.shape[0]} lignes")

print("Normalisation...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("Entrainement...")
model = RandomForestClassifier(
    n_estimators=100,
    random_state=PARAMS["random_state"],
    n_jobs=-1,
)
model.fit(X_train_scaled, y_train)

print("Evaluation...")
raw_preds  = model.predict(X_test_scaled)
y_pred   = model.predict(X_test_scaled)
raw_scores = model.score(X_test_scaled, y_pred, sample_weight=None)
y_scores = model.predict_proba(X_test_scaled)[:, 1]

precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
roc_auc   = roc_auc_score(y_test, y_scores)

print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall:.4f}")
print(f"  F1        : {f1:.4f}")
print(f"  ROC-AUC   : {roc_auc:.4f}")

print("Logging MLflow...")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("anomaly-detection")

with mlflow.start_run(run_name="random-forest-cicids2017"):

    mlflow.log_params(PARAMS)
    mlflow.log_param("n_features", len(FEATURES))
    mlflow.log_param("train_samples", X_train.shape[0])
    mlflow.log_param("test_samples", X_test.shape[0])

    mlflow.log_metrics({
        "precision": precision,
        "recall":    recall,
        "f1":        f1,
        "roc_auc":   roc_auc,
    })

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=X_train_scaled[:1],
    )

    scaler_path = "model/scaler.joblib"
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(scaler_path, artifact_path="scaler")

    run_id = mlflow.active_run().info.run_id
    print(f"  Run ID : {run_id}")

bundle = {
    "model":         model,
    "scaler":        scaler,
    "features":      FEATURES,
    "version":       "3.0.0-cicids2017",
    "mlflow_run_id": run_id,
}
joblib.dump(bundle, OUTPUT_PATH)
print(f"Modèle sauvegardé : {OUTPUT_PATH}")
print("Terminé.")