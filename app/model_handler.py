import numpy as np
import joblib
import os
import logging
from pathlib import Path

from app.schemas import NetworkFlowInput

logger = logging.getLogger(__name__)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "model/anomaly_detector.joblib"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0-dummy")

FEATURE_ORDER = [
    "duration", "protocol", "src_port", "dst_port",
    "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes",
    "fwd_pkt_len_mean", "bwd_pkt_len_mean",
    "flow_iat_mean", "flow_iat_std",
    "syn_flag", "fin_flag", "rst_flag",
]


class ModelHandler:
    """
    Encapsule le cycle de vie du modèle ML :
    chargement, inférence, et métadonnées.

    Pour l'instant utilise un modèle dummy (IsolationForest).
    À remplacer par ton vrai modèle entraîné sur CICIDS2017.
    """

    def __init__(self):
        self._model = None
        self._scaler = None
        self.version = MODEL_VERSION

    def load(self):
        """Charge le modèle depuis le disque (ou crée un dummy)."""
        if MODEL_PATH.exists():
            bundle = joblib.load(MODEL_PATH)
            self._model = bundle["model"]
            self._scaler = bundle.get("scaler")
            self.version = bundle.get("version", MODEL_VERSION)
            logger.info(f"Modèle chargé depuis {MODEL_PATH} (v{self.version})")
        else:
            logger.warning("⚠️  Modèle non trouvé — utilisation du dummy IsolationForest")
            self._load_dummy()

    def _load_dummy(self):
        """
        Crée un IsolationForest entraîné sur des données synthétiques.
        Simule un vrai workflow : scaler + modèle.
        """
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        # Données synthétiques : trafic normal + quelques anomalies
        rng = np.random.default_rng(42)
        normal = rng.normal(loc=[0.5, 6, 443, 443, 10, 8, 1500, 2000,
                                  150, 200, 12, 3, 0.3, 0.3, 0.05],
                            scale=[0.3, 2, 100, 50, 5, 4, 500, 800,
                                   50, 80, 5, 1, 0.4, 0.4, 0.2],
                            size=(1000, 15))
        normal = np.clip(normal, 0, None)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(normal)

        self._model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42,
        )
        self._model.fit(X_scaled)
        self.version = "1.0.0-dummy"
        logger.info("Dummy IsolationForest entraîné")

    def is_loaded(self) -> bool:
        return self._model is not None

    def _flow_to_array(self, flow: NetworkFlowInput) -> np.ndarray:
        """Convertit un NetworkFlowInput en vecteur numpy dans le bon ordre."""
        return np.array([[getattr(flow, f) for f in FEATURE_ORDER]])

    def predict(self, flow: NetworkFlowInput) -> dict:
        """
        Retourne un dict avec :
        - label (0=normal, 1=anomalie)
        - anomaly_score [0.0, 1.0]
        - confidence [0.0, 1.0]
        """
        X = self._flow_to_array(flow)

        if self._scaler is not None:
            X = self._scaler.transform(X)

        # IsolationForest : -1=anomalie, 1=normal
        raw_pred = self._model.predict(X)[0]
        # score_samples retourne un score négatif ; plus proche de 0 = plus anormal
        raw_score = self._model.score_samples(X)[0]

        # Normalisation du score en [0, 1] (0=normal, 1=très anormal)
        # Les scores IsolationForest sont typiquement dans [-0.5, 0.5]
        anomaly_score = float(np.clip(0.5 - raw_score, 0, 1))

        label = 1 if raw_pred == -1 else 0
        confidence = anomaly_score if label == 1 else (1 - anomaly_score)

        return {
            "label": label,
            "anomaly_score": round(anomaly_score, 4),
            "confidence": round(confidence, 4),
        }
