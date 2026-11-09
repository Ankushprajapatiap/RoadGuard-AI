
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

# Load dataset
df = pd.read_csv("data/highway_accidents_preprocessed.csv")

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

# Latest two years for testing
years = sorted(df["Year"].dropna().unique())
test_years = years[-2:]

train_mask = ~df["Year"].isin(test_years)
test_mask = df["Year"].isin(test_years)

y_train = df.loc[train_mask, "Accident_Severity_C"].astype(int)
y_test = df.loc[test_mask, "Accident_Severity_C"].astype(int)

# Baseline model
model = DummyClassifier(strategy="most_frequent")

model.fit(
    [[0]] * len(y_train),
    y_train
)

predictions = model.predict(
    [[0]] * len(y_test)
)

# Results
print("\nBASELINE MODEL RESULTS")
print("=" * 40)

print("Test years:", test_years)

print(
    "Accuracy:",
    round(accuracy_score(y_test, predictions) * 100, 2),
    "%"
)

print(
    "Balanced Accuracy:",
    round(balanced_accuracy_score(y_test, predictions) * 100, 2),
    "%"
)

print(
    "Macro F1:",
    round(f1_score(y_test, predictions, average="macro") * 100, 2),
    "%"
)
