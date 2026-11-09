
import pandas as pd
from pathlib import Path

# 1. Load cleaned dataset
DATA_PATH = Path("data/highway_accidents_clean.csv")

df = pd.read_csv(DATA_PATH)

print("Original dataset shape:", df.shape)

# 2. Check required columns
required_columns = [
    "Date",
    "Day_of_Week",
    "Time_of_Accident",
    "Accident_Location_A",
    "Accident_Location_A_Chainage_km",
    "Accident_Location_A_Chainage_km_RoadSide",
    "Accident_Severity_C",
    "Causes_D",
    "Road_Feature_E",
    "Road_Condition_F",
    "Weather_Conditions_H",
    "Vehicle_Type_Involved_J_V1",
    "Vehicle_Type_Involved_J_V2"
]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# 3. Process date
df["Date"] = pd.to_datetime(
    df["Date"],
    format="mixed",
    dayfirst=True,
    errors="coerce"
)

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# 4. Process accident time
time_parts = df["Time_of_Accident"].astype("string").str.extract(
    r"^\s*(\d{1,2}):(\d{2})"
)

df["Hour"] = pd.to_numeric(time_parts[0], errors="coerce")
df["Minute"] = pd.to_numeric(time_parts[1], errors="coerce")

# 5. Convert chainage into numeric format
chainage_column = "Accident_Location_A_Chainage_km"

df[chainage_column] = pd.to_numeric(
    df[chainage_column],
    errors="coerce"
)

# 6. Preserve all categorical codes
categorical_columns = [
    "Day_of_Week",
    "Accident_Location_A",
    "Accident_Location_A_Chainage_km_RoadSide",
    "Causes_D",
    "Road_Feature_E",
    "Road_Condition_F",
    "Weather_Conditions_H",
    "Vehicle_Type_Involved_J_V1",
    "Vehicle_Type_Involved_J_V2"
]

# Missing vehicle type is kept as Unknown
df["Vehicle_Type_Involved_J_V2"] = (
    df["Vehicle_Type_Involved_J_V2"].fillna("Unknown")
)

for col in categorical_columns:
    df[col] = df[col].astype("string")

# 7. Keep severity as the prediction target
df["Accident_Severity_C"] = pd.to_numeric(
    df["Accident_Severity_C"],
    errors="coerce"
)

# 8. Remove original date and time after extracting features
df.drop(
    columns=["Date", "Time_of_Accident"],
    inplace=True
)

# 9. Save preprocessed dataset
output_path = Path("data/highway_accidents_preprocessed.csv")

df.to_csv(output_path, index=False)

# 10. Display verification
print("\nPREPROCESSING COMPLETED")
print("Final dataset shape:", df.shape)
print("Remaining exact duplicates:", df.duplicated().sum())

print("\nMissing values:")
print(df.isnull().sum().to_string())

print("\nSeverity distribution:")
print(df["Accident_Severity_C"].value_counts().sort_index())

print("\nVehicle V2 categories:")
print(df["Vehicle_Type_Involved_J_V2"].value_counts())

print("\nSaved file:", output_path)
