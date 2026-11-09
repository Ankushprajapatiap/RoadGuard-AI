
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
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

# File paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "highway_accidents_preprocessed.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

# Target
target = "Accident_Severity_C"

# Explicit categorical columns
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

# Convert category codes into categorical strings
for col in categorical_columns:
    df[col] = df[col].astype(str)

# Separate features and target
X = df.drop(columns=[target])
y = df[target].astype(int)

# Numerical columns
numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

print("\nCategorical columns:", categorical_columns)
print("\nNumerical columns:", numerical_columns)

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_columns
        ),
        (
            "numerical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numerical_columns
        )
    ]
)

# Model
model = RandomForestClassifier(
    n_estimators=400,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Complete pipeline
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", model)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# Train
print("\nTraining improved Random Forest...")
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

# Results
print("\n========== IMPROVED MODEL RESULTS ==========")
print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Balanced Accuracy:", round(balanced_accuracy_score(y_test, y_pred) * 100, 2), "%")
print("Macro F1 Score:", round(f1_score(y_test, y_pred, average="macro") * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[1, 2, 3, 4],
    target_names=["Fatal", "Grievous Injury", "Minor Injury", "No Injury"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4]))

# Save model
model_path = MODEL_DIR / "accident_severity_model_v2.pkl"
joblib.dump(pipeline, model_path)

print("\nImproved model saved at:", model_path)
