
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV
)

# Load dataset
df = pd.read_csv(
    Path("data/highway_accidents_preprocessed.csv")
)

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

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        random_state=42,
        n_jobs=-1
    ))
])

parameters = {
    "model__n_estimators": [200, 300, 400],
    "model__max_depth": [10, 15, 20, 25, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", 0.7],
    "model__class_weight": [
        "balanced",
        "balanced_subsample"
    ]
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameters,
    n_iter=15,
    scoring="f1_macro",
    cv=cv,
    random_state=42,
    n_jobs=1,
    verbose=2
)

print("\nStarting Random Forest tuning...")

search.fit(X, y)

print("\nBEST PARAMETERS:")
print(search.best_params_)

print("\nBEST CROSS-VALIDATED MACRO F1:")
print(round(search.best_score_ * 100, 2), "%")

print("\nTUNING COMPLETED.")
