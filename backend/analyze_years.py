
import pandas as pd

df = pd.read_csv("data/highway_accidents_preprocessed.csv")

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df["Accident_Severity_C"] = pd.to_numeric(
    df["Accident_Severity_C"], errors="coerce"
)

df = df.dropna(subset=["Year", "Accident_Severity_C"])

df["Year"] = df["Year"].astype(int)
df["Accident_Severity_C"] = df["Accident_Severity_C"].astype(int)

print("\nYEAR-WISE RECORD COUNTS")
print("=" * 60)

print(df.groupby("Year").size().to_string())

print("\nYEAR-WISE SEVERITY COUNTS")
print("=" * 60)

table = pd.crosstab(
    df["Year"],
    df["Accident_Severity_C"]
)

table = table.reindex(columns=[1, 2, 3, 4], fill_value=0)
table.columns = [
    "Fatal",
    "Grievous",
    "Minor",
    "No Injury"
]

print(table.to_string())

print("\nYEAR-WISE SEVERITY PERCENTAGES")
print("=" * 60)

percentages = table.div(table.sum(axis=1), axis=0) * 100
print(percentages.round(2).to_string())

print("\nTOTAL RECORDS:", len(df))
print("YEAR RANGE:", df["Year"].min(), "-", df["Year"].max())
