import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    BASE_DIR / "data" / "uk_accidents"
    / "uk_collisions_preprocessed.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "uk_accident_severity_model_v3.pkl"


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded:", df.shape)

TARGET = "collision_severity"

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found.")

df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET]).copy()
df[TARGET] = df[TARGET].astype(int)

# STATS19 severity labels:
# 1 = Fatal, 2 = Serious, 3 = Slight

print("\nSeverity distribution:")
print(df[TARGET].value_counts().sort_index())


# --------------------------------------------------
# 3. REMOVE TARGET LEAKAGE / UNSUITABLE FEATURES
# --------------------------------------------------

drop_columns = [
    # Record identifiers
    "collision_index",
    "collision_ref_no",

    # Information known after the collision
    "number_of_casualties",
    "number_of_vehicles",
    "did_police_officer_attend_scene_of_accident",

    # Outcome-derived fields
    "enhanced_severity_collision",
    "collision_injury_based",
    "collision_adjusted_severity_serious",
    "collision_adjusted_severity_slight",

    # Redundant or high-cardinality fields
    "collision_year",
    "year",
    "first_road_number",
    "second_road_number",
    "local_authority_ons_district",
    "local_authority_highway",
    "local_authority_highway_current",
    "lsoa_of_accident_location"
]

X = df.drop(
    columns=[TARGET] + drop_columns,
    errors="ignore"
)

y = df[TARGET]


# --------------------------------------------------
# 4. DEFINE CATEGORICAL FEATURES
# --------------------------------------------------

categorical_columns = [
    "police_force",
    "day_of_week",
    "local_authority_district",

    "first_road_class",
    "road_type",

    "junction_detail",
    "junction_detail_historic",
    "junction_control",
    "second_road_class",

    "pedestrian_crossing",
    "pedestrian_crossing_human_control_historic",
    "pedestrian_crossing_physical_facilities_historic",

    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "special_conditions_at_site",

    "carriageway_hazards",
    "carriageway_hazards_historic",

    "urban_or_rural_area",
    "trunk_road_flag"
]

# Keep only columns that actually exist.
categorical_columns = [
    col for col in categorical_columns
    if col in X.columns
]

# Automatically include any remaining text columns.
for col in X.select_dtypes(include=["object", "string"]).columns:
    if col not in categorical_columns:
        categorical_columns.append(col)

# Convert category values to strings.
for col in categorical_columns:
    X[col] = X[col].astype("string")

numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

print("\nCategorical features:", categorical_columns)
print("\nNumerical features:", numerical_columns)

if not numerical_columns and not categorical_columns:
    raise ValueError("No usable features found.")


# --------------------------------------------------
# 5. SPLIT DATA: TRAIN / VALIDATION / TEST
# --------------------------------------------------

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=(0.15 / 0.85),
    random_state=42,
    stratify=y_train_full
)

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Testing records:", len(X_test))


# --------------------------------------------------
# 6. PREPROCESSING PIPELINE
# --------------------------------------------------

def create_pipeline(classifier):

    transformers = []

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore")
                )
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if numerical_columns:
        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]
        )

        transformers.append(
            ("numerical", numerical_pipeline, numerical_columns)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )


# --------------------------------------------------
# 7. COMPARE FOUR MODEL CONFIGURATIONS
# --------------------------------------------------

models = {
    "Extra Trees - No Class Weight": ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight=None,
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees - Balanced": ExtraTreesClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest - No Class Weight": RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight=None,
        random_state=42,
        n_jobs=-1
    ),

    "Random Forest - Balanced": RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 8. MAJORITY-CLASS BASELINE
# --------------------------------------------------

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_val)

print("\n========== BASELINE RESULTS ==========")
print(
    "Baseline validation accuracy:",
    round(accuracy_score(y_val, baseline_predictions) * 100, 2),
    "%"
)
print(
    "Baseline validation Macro F1:",
    round(f1_score(y_val, baseline_predictions, average="macro") * 100, 2),
    "%"
)


# --------------------------------------------------
# 9. TRAIN AND SELECT BEST MODEL ON VALIDATION
# --------------------------------------------------

best_model = None
best_model_name = None
best_macro_f1 = -1

print("\n========== MODEL COMPARISON ==========")

for name, classifier in models.items():

    print(f"\nTraining: {name}")

    pipeline = create_pipeline(classifier)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    accuracy = accuracy_score(y_val, predictions)
    balanced_accuracy = balanced_accuracy_score(y_val, predictions)
    macro_f1 = f1_score(
        y_val,
        predictions,
        average="macro",
        zero_division=0
    )

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("Balanced Accuracy:", round(balanced_accuracy * 100, 2), "%")
    print("Macro F1:", round(macro_f1 * 100, 2), "%")

    if macro_f1 > best_macro_f1:
        best_macro_f1 = macro_f1
        best_model = pipeline
        best_model_name = name


# --------------------------------------------------
# 10. REFIT SELECTED MODEL ON TRAIN + VALIDATION
# --------------------------------------------------

print("\nBest configuration:", best_model_name)

best_model.fit(X_train_full, y_train_full)


# --------------------------------------------------
# 11. FINAL TEST EVALUATION
# --------------------------------------------------

test_predictions = best_model.predict(X_test)

print("\n========== FINAL UK MODEL V3 RESULTS ==========")

print(
    "Accuracy:",
    round(accuracy_score(y_test, test_predictions) * 100, 2),
    "%"
)

print(
    "Balanced Accuracy:",
    round(balanced_accuracy_score(y_test, test_predictions) * 100, 2),
    "%"
)

print(
    "Macro F1:",
    round(
        f1_score(
            y_test,
            test_predictions,
            average="macro",
            zero_division=0
        ) * 100,
        2
    ),
    "%"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        test_predictions,
        labels=[1, 2, 3],
        target_names=["Fatal", "Serious", "Slight"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        test_predictions,
        labels=[1, 2, 3]
    )
)


# --------------------------------------------------
# 12. SAVE FINAL MODEL
# --------------------------------------------------

joblib.dump(best_model, MODEL_PATH)

print("\nModel saved successfully!")
print("Selected model:", best_model_name)
print("Location:", MODEL_PATH)
print("\nUK MODEL V3 TRAINING COMPLETED!")