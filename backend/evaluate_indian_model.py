
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Load dataset
df = pd.read_csv("data/highway_accidents_preprocessed.csv")

X = df.drop(columns=["Accident_Severity_C"])
y = df["Accident_Severity_C"].astype(int)

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
    X[col] = X[col].astype(str)

# Preprocessing
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

# Best parameters from tuning
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

# Cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nEvaluating tuned Indian model...")

predictions = cross_val_predict(
    pipeline,
    X,
    y,
    cv=cv,
    n_jobs=1
)

print("\nCROSS-VALIDATION RESULTS")
print("=" * 55)

print("Accuracy:", round(accuracy_score(y, predictions) * 100, 2), "%")
print("Balanced Accuracy:", round(
    balanced_accuracy_score(y, predictions) * 100, 2
), "%")
print("Macro F1:", round(
    f1_score(y, predictions, average="macro") * 100, 2
), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y,
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
print(confusion_matrix(y, predictions, labels=[1, 2, 3, 4]))

# Train final candidate on all available records
print("\nTraining final candidate model on all records...")
pipeline.fit(X, y)

# Save separately; do not replace the live model yet
output_path = Path("models/accident_severity_model_v3_candidate.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, output_path)

print("\nCandidate model saved at:", output_path)
print("Existing production model has NOT been replaced.")
