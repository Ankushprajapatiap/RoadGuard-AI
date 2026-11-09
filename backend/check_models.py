import joblib
from pathlib import Path

model_dir = Path(__file__).resolve().parent / "models"

files = [
    "accident_severity_model_v2.pkl",
    "uk_accident_severity_model_v3.pkl"
]

for filename in files:
    print("\nMODEL:", filename)

    path = model_dir / filename

    if not path.exists():
        print("NOT FOUND:", path)
        continue

    model = joblib.load(path)

    print("FEATURES:", list(model.feature_names_in_))
    print("TOTAL:", len(model.feature_names_in_))