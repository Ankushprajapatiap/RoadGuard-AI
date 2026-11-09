
import pandas as pd

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

# Load cleaned dataset
file_path = Path("data/highway_accidents_preprocessed.csv")
df = pd.read_csv(file_path)

# Separate input features and target
X = df.drop(columns=["Accident_Severity_C"])
y = df["Accident_Severity_C"].astype(int)

# Feature types
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

# Ensure categorical values are treated as categories
for col in categorical_columns:
    X[col] = X[col].astype(str)

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
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ]),
            numeric_columns
        )
    ]
)

# Algorithms to compare
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=400,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    )
}

# Stratified 5-fold cross-validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro"
}

print("\nINDIAN ACCIDENT SEVERITY MODEL COMPARISON")
print("=" * 65)

results = []

for name, model in models.items():
    print(f"\nTraining and evaluating: {name}")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1
    )

    accuracy = scores["test_accuracy"].mean() * 100
    balanced = scores["test_balanced_accuracy"].mean() * 100
    macro_f1 = scores["test_macro_f1"].mean() * 100

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 2),
        "Balanced Accuracy": round(balanced, 2),
        "Macro F1": round(macro_f1, 2)
    })

    print(f"Accuracy:          {accuracy:.2f}%")
    print(f"Balanced Accuracy: {balanced:.2f}%")
    print(f"Macro F1:          {macro_f1:.2f}%")

print("\nFINAL COMPARISON")
print("=" * 65)

result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

result_df.to_csv("model_comparison_results.csv", index=False)

print("\nResults saved to model_comparison_results.csv")
