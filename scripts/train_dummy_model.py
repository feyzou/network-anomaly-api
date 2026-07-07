import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

OUTPUT_PATH = Path("model/anomaly_detector.joblib")
OUTPUT_PATH.parent.mkdir(exist_ok=True)

FEATURES = [
    "duration", "protocol", "src_port", "dst_port",
    "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "fwd_pkt_len_mean", "bwd_pkt_len_mean",
    "flow_iat_mean", "flow_iat_std",
    "syn_flag", "fin_flag", "rst_flag",
]

print(" Génération des données synthétiques...")

rng = np.random.default_rng(42)

# Trafic normal : HTTP, HTTPS, DNS
normal = rng.normal(
    loc=[0.5, 6, 40000, 443, 10, 8, 1500, 2000, 150, 200, 12, 3, 0.3, 0.3, 0.05],
    scale=[0.3, 2, 10000, 100, 5, 4, 500, 800, 50, 80, 5, 1, 0.4, 0.4, 0.2],
    size=(2000, 15),
)
normal = np.clip(normal, 0, None)

# Quelques anomalies injectées (DDoS-like : beaucoup de paquets, IAT faible)
anomalies = rng.normal(
    loc=[0.01, 6, 54321, 80, 1000, 0, 64000, 0, 64, 0, 0.1, 0.01, 1, 0, 1],
    scale=[0.005, 0, 100, 0, 200, 0, 10000, 0, 5, 0, 0.05, 0.005, 0, 0, 0],
    size=(100, 15),
)
anomalies = np.clip(anomalies, 0, None)

X_train = np.vstack([normal, anomalies])

print(" Entraînement du modèle...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

model = IsolationForest(
    n_estimators=200,
    max_samples="auto",
    contamination=0.05,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_scaled)

bundle = {
    "model": model,
    "scaler": scaler,
    "features": FEATURES,
    "version": "1.0.0",
}
joblib.dump(bundle, OUTPUT_PATH)

print(f" Modèle sauvegardé : {OUTPUT_PATH}")
print(f" Features : {len(FEATURES)}")
print(f" Samples d'entraînement : {len(X_train)}")
