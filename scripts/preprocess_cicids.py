import pandas as pd
import numpy as np
from pathlib import Path

INPUT_PATH  = Path("C:/Users/pc/Documents/ML-projects/network-anomaly-api/data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
OUTPUT_PATH = Path("C:/Users/pc/Documents/ML-projects/network-anomaly-api/data/processed/cicids_processed.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Correspondance colonnes CICIDS2017 -> features de notre API
COLUMN_MAPPING = {
    "Destination Port":            "dst_port",
    "Flow Duration":               "duration",
    "Total Fwd Packets":           "fwd_packets",
    "Total Backward Packets":      "bwd_packets",
    "Total Length of Fwd Packets": "fwd_bytes",
    "Total Length of Bwd Packets": "bwd_bytes",
    "Fwd Packet Length Mean":      "fwd_pkt_len_mean",
    "Bwd Packet Length Mean":      "bwd_pkt_len_mean",
    "Flow IAT Mean":               "flow_iat_mean",
    "Flow IAT Std":                "flow_iat_std",
    "SYN Flag Count":              "syn_flag",
    "FIN Flag Count":              "fin_flag",
    "RST Flag Count":              "rst_flag",
}

FEATURES = list(COLUMN_MAPPING.values())

print("Chargement du dataset...")
df = pd.read_csv(INPUT_PATH)

# Nettoyage des espaces dans les noms de colonnes
df.columns = df.columns.str.strip()
print(f"  Shape initial : {df.shape}")

# Remplacement des valeurs infinies par NaN puis suppression
print("Nettoyage des valeurs infinies et manquantes...")
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
print(f"  Shape apres nettoyage : {df.shape}")

# Conversion du label : BENIGN=0, DDoS=1
print("Encodage des labels...")
df["label"] = (df["Label"] != "BENIGN").astype(int)
print(f"  Distribution : {df['label'].value_counts().to_dict()}")

# Selection et renommage des colonnes
df = df[list(COLUMN_MAPPING.keys()) + ["label"]]
df = df.rename(columns=COLUMN_MAPPING)

# Conversion duration : microsecondes -> secondes
df["duration"] = df["duration"] / 1_000_000

# Les flags sont des compteurs dans CICIDS2017, on les ramene a 0/1
for flag in ["syn_flag", "fin_flag", "rst_flag"]:
    df[flag] = (df[flag] > 0).astype(int)

# On ajoute src_port et protocol manquants avec des valeurs neutres
# (absents du dataset CICIDS2017 mais requis par notre API)
df["src_port"] = 0
df["protocol"] = 6  # TCP par defaut (DDoS utilise majoritairement TCP)

# Verification finale
print(f"\nFeatures finales : {FEATURES}")
print(f"Shape final : {df.shape}")
print(f"\nApercu :")
print(df.head(3))

df.to_csv(OUTPUT_PATH, index=False)
print(f"\nDataset sauvegardé : {OUTPUT_PATH}")