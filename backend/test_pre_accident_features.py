
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

df = pd.read_csv("data/highway_accidents_preprocessed.csv")

# Only information potentially available before an accident
features = [
    "Day_of_Week",
    "Accident_Location_A",
    "Accident_Location_A_Chainage_km",
    "Accident_Location_A_Chainage_km_RoadSide",
    "Road_Feature_E",
    "Road_Condition_F",
    "Weather_Conditions_H",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute"
]

X = df[features].copy()
y = df["Accident_Severity_C"].astype(int)

categorical = [
    "Day_of_Week",
    "Accident_Location_A",
    "Accident_Location_A_Chainage_km_RoadSide",
    "Road_Feature_E",
    "Road_Condition_F",
    "Weather_Conditions_H"
]

numeric = [
    "Accident_Location_A_Chainage_km",
    "Year", "Month", "Day", "Hour", "Minute"
]

for col in categorical:
    X[col] = X[col].astype(str)

preprocessor = ColumnTransformer([
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    ),
    (
        "num",
        SimpleImputer(strategy="median"),
        numeric
    )
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        max_depth=25,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\nPRE-ACCIDENT FEATURES MODEL TEST")
print("=" * 50)

scores = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=[
        "accuracy",
        "balanced_accuracy",
        "f1_macro"
    ],
    n_jobs=1
)

print("Accuracy:",
      round(scores["test_accuracy"].mean() * 100, 2), "%")

print("Balanced Accuracy:",
      round(scores["test_balanced_accuracy"].mean() * 100, 2), "%")

print("Macro F1:",
      round(scores["test_f1_macro"].mean() * 100, 2), "%")

print("\nTEMPORAL TEST")

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
years = sorted(df["Year"].dropna().unique())

test_years = years[-2:]

train_mask = ~df["Year"].isin(test_years)
test_mask = df["Year"].isin(test_years)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]

y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print("Test years:", test_years)
print("Accuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")
print("Balanced Accuracy:", round(
    balanced_accuracy_score(y_test, predictions) * 100, 2
), "%")
print("Macro F1:", round(
    f1_score(y_test, predictions, average="macro") * 100, 2
), "%")
