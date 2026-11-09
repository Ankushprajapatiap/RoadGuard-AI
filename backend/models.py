
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime
from datetime import datetime

from database import Base


class Accident(Base):
    __tablename__ = "accidents"

    id = Column(Integer, primary_key=True, index=True)

    accident_date = Column(DateTime, default=datetime.utcnow)

    location = Column(String(200), nullable=False)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    road_type = Column(String(100))
    weather = Column(String(100))
    traffic_density = Column(String(100))
    visibility = Column(String(100))

    accident_occurred = Column(Boolean, default=True)

    # Existing field - preserved for backward compatibility
    severity = Column(String(50))

    data_source = Column(String(100), default="user_entered")

    # New ML prediction fields
    predicted_severity = Column(String(50), nullable=True)

    prediction_confidence = Column(Float, nullable=True)

    # Human verification fields
    verified_severity = Column(String(50), nullable=True)

    verification_status = Column(
        String(30),
        default="Pending",
        nullable=False,
        server_default="Pending"
    )
