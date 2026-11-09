
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

INPUT_PATH = BASE_DIR / "data" / "uk_accidents" / "uk_collisions_2025.csv"
OUTPUT_PATH = BASE_DIR / "data" / "uk_accidents" / "uk_collisions_preprocessed.csv"

df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)

# Remove duplicate records
df = df.drop_duplicates(subset="collision_index").copy()

# Convert date and time
df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day

time_parts = df["time"].astype("string").str.extract(
    r"^\s*(\d{1,2}):(\d{2})"
)

df["hour"] = pd.to_numeric(time_parts[0], errors="coerce")
df["minute"] = pd.to_numeric(time_parts[1], errors="coerce")

# Remove original date and time text
df.drop(columns=["date", "time"], inplace=True)

# Save as clean dataset
df.to_csv(OUTPUT_PATH, index=False)

print("\nPREPROCESSING COMPLETED")
print("Final shape:", df.shape)
print("Duplicate records:", df.duplicated().sum())

print("\nSeverity distribution:")
print(df["collision_severity"].value_counts().sort_index())

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))

print("\nSaved:", OUTPUT_PATH)
