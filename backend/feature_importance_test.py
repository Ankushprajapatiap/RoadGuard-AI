
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score

# Load dataset
df = pd.read_csv("data/highway_accidents_preprocessed.csv")

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
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical),
    ("num", SimpleImputer(strategy="median"), numeric)
])

model = Pipeline([
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

# Keep the latest two years for testing
years = sorted(df["Year"].dropna().unique())
test_years = years[-2:]

train_mask = ~df["Year"].isin(test_years)
test_mask = df["Year"].isin(test_years)

X_train = X.loc[train_mask]
X_test = X.loc[test_mask]
y_train = y.loc[train_mask]
y_test = y.loc[test_mask]

print("Training model...")
model.fit(X_train, y_train)

print("Calculating feature importance...")
result = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="f1_macro",
    n_repeats=10,
    random_state=42,
    n_jobs=1
)

importance = pd.DataFrame({
    "Feature": features,
    "Importance": result.importances_mean
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("=" * 50)
print(importance.to_string(index=False))

importance.to_csv("feature_importance_results.csv", index=False)

print("\nSaved: feature_importance_results.csv")
