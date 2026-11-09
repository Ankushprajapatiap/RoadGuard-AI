
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load data
df = pd.read_csv("data/highway_accidents_preprocessed.csv")

# Sort by year
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year", "Accident_Severity_C"])
df["Year"] = df["Year"].astype(int)

years = sorted(df["Year"].unique())

# Keep the latest 20% of years for testing
test_year_count = max(1, round(len(years) * 0.20))
test_years = years[-test_year_count:]

train = df[~df["Year"].isin(test_years)].copy()
test = df[df["Year"].isin(test_years)].copy()

print("TRAIN YEARS:", sorted(train["Year"].unique()))
print("TEST YEARS:", sorted(test["Year"].unique()))
print("TRAIN RECORDS:", len(train))
print("TEST RECORDS:", len(test))

X_train = train.drop(columns=["Accident_Severity_C"])
y_train = train["Accident_Severity_C"].astype(int)

X_test = test.drop(columns=["Accident_Severity_C"])
y_test = test["Accident_Severity_C"].astype(int)

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

numeric_columns = [
    "Accident_Location_A_Chainage_km",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute"
]

for col in categorical_columns:
    X_train[col] = X_train[col].astype(str)
    X_test[col] = X_test[col].astype(str)

preprocessor = ColumnTransformer([
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical_columns
    ),
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]),
        numeric_columns
    )
])

model = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features="sqrt",
    max_depth=25,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

print("\nTraining on older years...")
pipeline.fit(X_train, y_train)

print("\nTesting on newer years...")
predictions = pipeline.predict(X_test)

print("\nTEMPORAL TEST RESULTS")
print("=" * 50)

print("Accuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

print("Balanced Accuracy:", round(
    balanced_accuracy_score(y_test, predictions) * 100, 2
), "%")

print("Macro F1:", round(
    f1_score(y_test, predictions, average="macro") * 100, 2
), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    labels=[1, 2, 3, 4],
    target_names=[
        "Fatal",
        "Grievous Injury",
        "Minor Injury",
        "No Injury"
    ],
    zero_division=0
))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions, labels=[1, 2, 3, 4]))
