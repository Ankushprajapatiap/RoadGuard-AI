
from datetime import datetime
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func
from sqlalchemy.orm import Session
from pydantic import BaseModel, ConfigDict

from database import engine, Base, get_db
import models


# --------------------------------------------------
# DATABASE
# --------------------------------------------------

Base.metadata.create_all(bind=engine)


# --------------------------------------------------
# LOAD MACHINE LEARNING MODELS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

india_model = None
uk_model = None

try:
    india_model = joblib.load(
        MODEL_DIR / "accident_severity_model_v2.pkl"
    )
    print("Indian ML model loaded successfully.")

except Exception as error:
    print("Indian model loading error:", error)

try:
    uk_model = joblib.load(
        MODEL_DIR / "uk_accident_severity_model_v3.pkl"
    )
    print("UK ML model loaded successfully.")

except Exception as error:
    print("UK model loading error:", error)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="RoadGuard AI API",
    description="Traffic Accident Prediction and Road Safety API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# ACCIDENT DATABASE SCHEMAS
# --------------------------------------------------

class AccidentCreate(BaseModel):
    location: str
    latitude: float | None = None
    longitude: float | None = None
    road_type: str | None = None
    weather: str | None = None
    traffic_density: str | None = None
    visibility: str | None = None
    accident_occurred: bool = True
    severity: str | None = None


class AccidentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    accident_date: datetime
    location: str
    latitude: float | None = None
    longitude: float | None = None
    road_type: str | None = None
    weather: str | None = None
    traffic_density: str | None = None
    visibility: str | None = None
    accident_occurred: bool
    severity: str | None = None
    data_source: str | None = None

    # ML prediction fields
    predicted_severity: str | None = None
    prediction_confidence: float | None = None

    # Human verification fields
    verified_severity: str | None = None
    verification_status: str | None = None


# --------------------------------------------------
# ACCIDENT VERIFICATION SCHEMA
# --------------------------------------------------

class AccidentVerificationRequest(BaseModel):
    status: Literal["Verified", "Rejected"]
    verified_severity: str | None = None


# --------------------------------------------------
# INDIAN MODEL INPUT SCHEMA
# --------------------------------------------------

class IndiaPredictionRequest(BaseModel):

    Day_of_Week: str | int | None = None
    Accident_Location_A: str | int | None = None
    Accident_Location_A_Chainage_km: float | None = None
    Accident_Location_A_Chainage_km_RoadSide: str | int | None = None
    Causes_D: str | int | None = None
    Road_Feature_E: str | int | None = None
    Road_Condition_F: str | int | None = None
    Weather_Conditions_H: str | int | None = None
    Vehicle_Type_Involved_J_V1: str | int | None = None
    Vehicle_Type_Involved_J_V2: str | int | None = None
    Year: float | None = None
    Month: float | None = None
    Day: float | None = None
    Hour: float | None = None
    Minute: float | None = None


# --------------------------------------------------
# UK MODEL INPUT SCHEMA
# --------------------------------------------------

class UKPredictionRequest(BaseModel):

    location_easting_osgr: float | None = None
    location_northing_osgr: float | None = None
    longitude: float | None = None
    latitude: float | None = None

    police_force: str | int | None = None
    day_of_week: str | int | None = None
    local_authority_district: str | int | None = None
    first_road_class: str | int | None = None
    road_type: str | int | None = None
    speed_limit: float | None = None

    junction_detail_historic: str | int | None = None
    junction_detail: str | int | None = None
    junction_control: str | int | None = None
    second_road_class: str | int | None = None

    pedestrian_crossing_human_control_historic: str | int | None = None
    pedestrian_crossing_physical_facilities_historic: str | int | None = None
    pedestrian_crossing: str | int | None = None

    light_conditions: str | int | None = None
    weather_conditions: str | int | None = None
    road_surface_conditions: str | int | None = None
    special_conditions_at_site: str | int | None = None
    carriageway_hazards_historic: str | int | None = None
    carriageway_hazards: str | int | None = None
    urban_or_rural_area: str | int | None = None
    trunk_road_flag: str | int | None = None

    month: float | None = None
    day: float | None = None
    hour: float | None = None
    minute: float | None = None


# --------------------------------------------------
# SEVERITY LABELS
# --------------------------------------------------

INDIA_LABELS = {
    1: "Fatal",
    2: "Grievous Injury",
    3: "Minor Injury",
    4: "No Injury"
}

UK_LABELS = {
    1: "Fatal",
    2: "Serious",
    3: "Slight"
}


# --------------------------------------------------
# COMMON PREDICTION FUNCTION
# --------------------------------------------------

def make_prediction(model, request_data, labels, convert_categories=False):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="The requested ML model is not loaded. Check the backend model files."
        )

    features = list(model.feature_names_in_)
    input_data = request_data.model_dump()

    # Preserve exact model feature order
    row = {
        feature: input_data.get(feature)
        for feature in features
    }

    # Convert Indian categorical features to strings
    if convert_categories:
        categorical_features = [
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

        for feature in categorical_features:
            if row[feature] is not None:
                row[feature] = str(row[feature])

    input_df = pd.DataFrame([row], columns=features)

    try:
        prediction = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]

    except Exception as error:
        raise HTTPException(
            status_code=422,
            detail=f"Prediction failed: {str(error)}"
        )

    class_probabilities = {
        labels.get(int(class_value), "Unknown"): round(
            float(probability), 4
        )
        for class_value, probability in zip(
            model.classes_,
            probabilities
        )
    }

    highest_score = max(probabilities)

    return {
        "predicted_class": prediction,
        "predicted_severity": labels.get(prediction, "Unknown"),
        "model_score_percent": round(
            float(highest_score) * 100, 2
        ),
        "class_probabilities": class_probabilities,
        "note": "Model scores are estimates, not calibrated real-world probabilities."
    }


# --------------------------------------------------
# GENERAL ENDPOINTS
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "RoadGuard AI Backend is running",
        "status": "success"
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/database-status")
def database_status():
    return {
        "database": "SQLite",
        "status": "connected",
        "table": "accidents"
    }


# --------------------------------------------------
# ACCIDENT DATABASE ENDPOINTS
# --------------------------------------------------

@app.post("/accidents", response_model=AccidentResponse)
def create_accident(
    accident: AccidentCreate,
    db: Session = Depends(get_db)
):
    new_accident = models.Accident(**accident.model_dump())

    db.add(new_accident)
    db.commit()
    db.refresh(new_accident)

    return new_accident


@app.get("/accidents", response_model=list[AccidentResponse])
def get_accidents(db: Session = Depends(get_db)):
    return db.query(models.Accident).order_by(
        models.Accident.id.desc()
    ).all()


# --------------------------------------------------
# GOVERNMENT OFFICER VERIFICATION API
# --------------------------------------------------

@app.put("/accidents/{accident_id}/verify")
def verify_accident(
    accident_id: int,
    verification: AccidentVerificationRequest,
    db: Session = Depends(get_db)
):

    accident = db.query(models.Accident).filter(
        models.Accident.id == accident_id
    ).first()

    if accident is None:
        raise HTTPException(
            status_code=404,
            detail="Accident record not found."
        )

    # Update verification status
    accident.verification_status = verification.status

    if verification.status == "Verified":
        accident.verified_severity = verification.verified_severity
    else:
        accident.verified_severity = None

    db.commit()
    db.refresh(accident)

    return {
        "message": "Accident verification updated successfully.",
        "accident_id": accident.id,
        "verification_status": accident.verification_status,
        "verified_severity": accident.verified_severity
    }


# --------------------------------------------------
# DASHBOARD STATISTICS API
# --------------------------------------------------

@app.get("/dashboard/stats")
def get_dashboard_stats(db: Session = Depends(get_db)):

    # 1. Total accident records
    total_accidents = db.query(models.Accident).filter(
        models.Accident.accident_occurred.is_(True)
    ).count()

    # 2. Actual severity counts
    severity_rows = db.query(
        func.lower(func.trim(models.Accident.severity)),
        func.count(models.Accident.id)
    ).filter(
        models.Accident.accident_occurred.is_(True)
    ).group_by(
        func.lower(func.trim(models.Accident.severity))
    ).all()

    severity_counts = {
        "Fatal": 0,
        "Grievous Injury": 0,
        "Minor Injury": 0,
        "No Injury": 0,
        "Other": 0
    }

    severity_mapping = {
        "fatal": "Fatal",
        "grievous injury": "Grievous Injury",
        "grievous": "Grievous Injury",
        "serious": "Grievous Injury",
        "minor injury": "Minor Injury",
        "minor": "Minor Injury",
        "no injury": "No Injury"
    }

    for severity, count in severity_rows:
        label = severity_mapping.get(severity, "Other") if severity else "Other"
        severity_counts[label] += count

    # 3. ML predicted severity distribution
    prediction_rows = db.query(
        func.lower(func.trim(models.Accident.predicted_severity)),
        func.count(models.Accident.id)
    ).filter(
        models.Accident.predicted_severity.isnot(None),
        models.Accident.accident_occurred.is_(True)
    ).group_by(
        func.lower(func.trim(models.Accident.predicted_severity))
    ).all()

    predicted_severity_counts = {
        "Fatal": 0,
        "Grievous Injury": 0,
        "Minor Injury": 0,
        "No Injury": 0,
        "Other": 0
    }

    for severity, count in prediction_rows:
        label = severity_mapping.get(severity, "Other") if severity else "Other"
        predicted_severity_counts[label] += count

    # 4. Year-wise accident statistics
    year_rows = db.query(
        func.strftime("%Y", models.Accident.accident_date).label("year"),
        func.count(models.Accident.id).label("total")
    ).filter(
        models.Accident.accident_occurred.is_(True),
        models.Accident.accident_date.isnot(None)
    ).group_by(
        func.strftime("%Y", models.Accident.accident_date)
    ).order_by(
        func.strftime("%Y", models.Accident.accident_date)
    ).all()

    yearly_accidents = [
        {
            "year": year,
            "total": count
        }
        for year, count in year_rows
        if year is not None
    ]

    # 5. Recent accidents
    recent = db.query(models.Accident).filter(
        models.Accident.accident_occurred.is_(True)
    ).order_by(
        models.Accident.id.desc()
    ).limit(5).all()

    recent_accidents = [
        {
            "id": item.id,
            "location": item.location,
            "severity": item.severity,
            "predicted_severity": item.predicted_severity,
            "verification_status": item.verification_status,
            "road_type": item.road_type,
            "weather": item.weather,
            "accident_date": (
                item.accident_date.isoformat()
                if item.accident_date else None
            )
        }
        for item in recent
    ]

    # 6. ML verification statistics
    pending_verification = db.query(models.Accident).filter(
        models.Accident.predicted_severity.isnot(None),
        models.Accident.verification_status == "Pending"
    ).count()

    classified_records = db.query(models.Accident).filter(
        models.Accident.predicted_severity.isnot(None)
    ).count()

    verified_records = db.query(models.Accident).filter(
        models.Accident.verification_status == "Verified"
    ).count()

    rejected_records = db.query(models.Accident).filter(
        models.Accident.verification_status == "Rejected"
    ).count()

    return {
        "total_accidents": total_accidents,
        "severity_counts": severity_counts,
        "predicted_severity_counts": predicted_severity_counts,
        "yearly_accidents": yearly_accidents,
        "classified_records": classified_records,
        "pending_verification": pending_verification,
        "verified_records": verified_records,
        "rejected_records": rejected_records,
        "recent_accidents": recent_accidents
    }


# --------------------------------------------------
# INDIAN ML PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict/india")
def predict_indian_severity(data: IndiaPredictionRequest):

    return make_prediction(
        india_model,
        data,
        INDIA_LABELS,
        convert_categories=True
    )


# --------------------------------------------------
# INDIAN ML PREDICTION + DATABASE SAVE
# --------------------------------------------------

@app.post("/accidents/{accident_id}/predict/india")
def predict_and_save_indian_severity(
    accident_id: int,
    data: IndiaPredictionRequest,
    db: Session = Depends(get_db)
):

    # Find accident record
    accident = db.query(models.Accident).filter(
        models.Accident.id == accident_id
    ).first()

    if accident is None:
        raise HTTPException(
            status_code=404,
            detail="Accident record not found."
        )

    # Generate ML prediction
    result = make_prediction(
        india_model,
        data,
        INDIA_LABELS,
        convert_categories=True
    )

    # Save prediction
    accident.predicted_severity = result["predicted_severity"]

    accident.prediction_confidence = result["model_score_percent"]

    # Human verification is pending
    accident.verification_status = "Pending"
    accident.verified_severity = None

    db.commit()
    db.refresh(accident)

    return {
        "message": "ML prediction generated and saved successfully.",
        "accident_id": accident.id,
        "predicted_severity": accident.predicted_severity,
        "prediction_confidence": accident.prediction_confidence,
        "verification_status": accident.verification_status,
        "class_probabilities": result["class_probabilities"],
        "note": result["note"]
    }


# --------------------------------------------------
# UK ML PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict/uk")
def predict_uk_severity(data: UKPredictionRequest):

    return make_prediction(
        uk_model,
        data,
        UK_LABELS
    )
