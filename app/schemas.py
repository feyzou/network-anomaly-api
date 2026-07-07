from pydantic import BaseModel, Field, field_validator
from typing import List


class NetworkFlowInput(BaseModel):
    """
    Features extraites d'un flux réseau (inspiré du dataset CICIDS2017).
    Toutes les valeurs doivent être >= 0.
    """

    duration: float = Field(..., ge=0, description="Durée du flux en secondes")
    protocol: int = Field(..., ge=0, le=255, description="Numéro de protocole (6=TCP, 17=UDP, 1=ICMP)")
    src_port: int = Field(..., ge=0, le=65535, description="Port source")
    dst_port: int = Field(..., ge=0, le=65535, description="Port destination")

    fwd_packets: int = Field(..., ge=0, description="Nombre de paquets envoyés")
    bwd_packets: int = Field(..., ge=0, description="Nombre de paquets reçus")
    fwd_bytes: int = Field(..., ge=0, description="Octets envoyés")
    bwd_bytes: int = Field(..., ge=0, description="Octets reçus")

    fwd_pkt_len_mean: float = Field(..., ge=0, description="Taille moyenne des paquets fwd")
    bwd_pkt_len_mean: float = Field(..., ge=0, description="Taille moyenne des paquets bwd")

    flow_iat_mean: float = Field(..., ge=0, description="IAT moyen du flux (ms)")
    flow_iat_std: float = Field(..., ge=0, description="Écart-type de l'IAT")

    syn_flag: int = Field(0, ge=0, le=1, description="Flag SYN présent")
    fin_flag: int = Field(0, ge=0, le=1, description="Flag FIN présent")
    rst_flag: int = Field(0, ge=0, le=1, description="Flag RST présent")

    @field_validator("protocol")
    @classmethod
    def protocol_known(cls, v):
        known = {1, 6, 17, 58}  # ICMP, TCP, UDP, ICMPv6
        if v not in known:
            # on accepte quand même, juste un warning implicite
            pass
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    label: int = Field(..., description="0=normal, 1=anomalie")
    label_text: str = Field(..., description="Libellé lisible")
    anomaly_score: float = Field(..., description="Score d'anomalie [0.0, 1.0]")
    confidence: float = Field(..., description="Confiance de la prédiction [0.0, 1.0]")
    latency_ms: float = Field(..., description="Latence d'inférence en millisecondes")


class BatchInput(BaseModel):
    flows: List[NetworkFlowInput] = Field(..., description="Liste de flux réseau (max 100)")


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    total: int
    anomaly_count: int
    normal_count: int
    total_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
