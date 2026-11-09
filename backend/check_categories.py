import pandas as pd
from pathlib import Path

file_path = Path("data/highway_accidents_preprocessed.csv")

df = pd.read_csv(file_path)

print("\nDATASET SHAPE:")
print(df.shape)

print("\nSEVERITY DISTRIBUTION:")
print(df["Accident_Severity_C"].value_counts().sort_index())

for col in [
    "Causes_D",
    "Accident_Location_A_Chainage_km_RoadSide"
]:
    print(f"\n{col} CATEGORY COUNTS:")
    print(df[col].astype(str).value_counts().sort_index())