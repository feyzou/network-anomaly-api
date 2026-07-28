import pandas as pd

PATH = "C:/Users/pc/Documents/ML-projects/network-anomaly-api/data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"

df = pd.read_csv(PATH, nrows=5)

print("=== Colonnes ===")
for col in df.columns:
    print(f"  {col}")

print(f"\n=== Shape ===")
df_full = pd.read_csv(PATH)
print(f"  {df_full.shape[0]} lignes, {df_full.shape[1]} colonnes")

print(f"\n=== Distribution des labels ===")
print(df_full[" Label"].value_counts())

print(f"\n=== Types ===")
print(df_full.dtypes)

print(f"\n=== Valeurs manquantes ===")
print(df_full.isnull().sum()[df_full.isnull().sum() > 0])

print(f"\n=== Valeurs infinies ===")
import numpy as np
num_cols = df_full.select_dtypes(include=[np.number]).columns
inf_counts = np.isinf(df_full[num_cols]).sum()
print(inf_counts[inf_counts > 0])