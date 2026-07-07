"""
Tests unitaires et d'intégration de l'API.

Usage:
    pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app

# use_lifespan=True déclenche le chargement du modèle au démarrage du client
@pytest.fixture(scope="session")
def client():
    with TestClient(app) as c:
        yield c

# ── Fixture : payload valide ──────────────────────────────────────────────────

VALID_FLOW = {
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
    "rst_flag": 0,
}

ANOMALY_FLOW = {
    "duration": 0.01,
    "protocol": 6,
    "src_port": 54321,
    "dst_port": 80,
    "fwd_packets": 1000,
    "bwd_packets": 0,
    "fwd_bytes": 64000,
    "bwd_bytes": 0,
    "fwd_pkt_len_mean": 64.0,
    "bwd_pkt_len_mean": 0.0,
    "flow_iat_mean": 0.1,
    "flow_iat_std": 0.01,
    "syn_flag": 1,
    "fin_flag": 0,
    "rst_flag": 1,
}


# ── Tests /health ─────────────────────────────────────────────────────────────

def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "model_version" in data


# ── Tests /predict ────────────────────────────────────────────────────────────

def test_predict_valid_flow(client):
    response = client.post("/predict", json=VALID_FLOW)
    assert response.status_code == 200
    data = response.json()
    assert data["label"] in [0, 1]
    assert data["label_text"] in ["normal", "anomalie"]
    assert 0.0 <= data["anomaly_score"] <= 1.0
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["latency_ms"] >= 0


def test_predict_returns_label_and_score(client):
    response = client.post("/predict", json=VALID_FLOW)
    data = response.json()
    if data["label"] == 1:
        assert data["label_text"] == "anomalie"
    else:
        assert data["label_text"] == "normal"


def test_predict_missing_required_field(client):
    incomplete = {k: v for k, v in VALID_FLOW.items() if k != "duration"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_negative_value_rejected(client):
    bad = {**VALID_FLOW, "fwd_packets": -1}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_invalid_port_rejected(client):
    bad = {**VALID_FLOW, "src_port": 99999}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


# ── Tests /predict/batch ──────────────────────────────────────────────────────

def test_batch_predict(client):
    payload = {"flows": [VALID_FLOW, ANOMALY_FLOW]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 2
    assert data["anomaly_count"] + data["normal_count"] == data["total"]
    assert len(data["predictions"]) == 2


def test_batch_too_large(client):
    payload = {"flows": [VALID_FLOW] * 101}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 400


def test_batch_single_flow(client):
    payload = {"flows": [VALID_FLOW]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    assert response.json()["total"] == 1
