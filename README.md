# Network Anomaly Detection API — Étape 1

API REST de détection d'anomalies réseau, construite avec **FastAPI** et containerisée avec **Docker**.

## Stack

| Composant | Techno |
|---|---|
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Modèle | IsolationForest entraîné sur des flux réseau synthétiques (inspiré CICIDS2017)(scikit-learn) |
| Container | Docker multi-stage |
| Tests | Pytest + httpx |
| Profil | Ingénieur télécom/sécurité — ce projet couvre le gap entre expérimentation ML et mise en production

## Architecture

```
POST /predict
     │
     ▼
 Pydantic v2          → validation des features réseau (15 features)
     │
     ▼
 StandardScaler       → normalisation
     │
     ▼
 IsolationForest      → score d'anomalie [0.0, 1.0]
     │
     ▼
 PredictionResponse   → label, score, confidence, latency_ms
```

## Démarrage rapide

```bash
git clone https://github.com/feyzou/network-anomaly-api
cd network-anomaly-api

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements-dev.txt
python scripts/train_dummy_model.py
pytest tests/ -v
uvicorn app.main:app --reload --port 8000
```
L'API est disponible sur **http://localhost:8000**  
Documentation Swagger : **http://localhost:8000/docs**

## Endpoints

| Méthode | Route | Description |
|---|---|---|
| GET | `/health` | Statut de l'API et du modèle |
| POST | `/predict` | Analyse un flux réseau |
| POST | `/predict/batch` | Analyse un lot de flux (max 100) |

## Exemple d'appel

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "duration": 0.5,
    "protocol": 6,
    "src_port": 54321,
    "dst_port": 80,
    "fwd_packets": 10,
    "bwd_packets": 8,
    "fwd_bytes": 1500,
    "bwd_bytes": 2048,
    "fwd_pkt_len_mean": 150.0,
    "bwd_pkt_len_mean": 256.0,
    "flow_iat_mean": 12.5,
    "flow_iat_std": 3.2,
    "syn_flag": 1,
    "fin_flag": 1,
    "rst_flag": 0
  }'
```

Réponse :
```json
{
  "label": 0,
  "label_text": "normal",
  "anomaly_score": 0.1823,
  "confidence": 0.8177,
  "latency_ms": 1.42
}
```
## Roadmap MLOps

- [x] **Étape 1** — FastAPI + Docker + tests (actuel)
- [ ] **Étape 2** — MLflow : tracking des expériences et versioning du modèle
- [ ] **Étape 3** — CI/CD avec GitHub Actions
- [ ] **Étape 4** — Monitoring drift avec Evidently AI + Grafana

## Tests

```bash
make test
```

## Remplacer le modèle dummy

1. Entraîner le modèle sur CICIDS2017
2. Sauvegarder avec `joblib.dump({"model": model, "scaler": scaler, "version": "2.0.0"}, "model/anomaly_detector.joblib")`
3. Redémarrer l'API — le modèle est chargé au démarrage

## Prochaines étapes (Étape 2)

- Intégration **MLflow** pour le tracking des expériences
- Versionning du modèle et des métriques d'entraînement
- Ajout d'un endpoint `/metrics` pour Prometheus
