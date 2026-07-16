from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.database import Base

class FleetAssetTransition(Base):
    """
    SQL Schema storing feasibility analysis metrics for electrified fleet vehicles.
    """
    __tablename__ = "fleet_asset_transitions"

    id = Column(Integer, primary_key=True, index=True)
    vehicle_id = Column(String, unique=True, index=True, nullable=False)
    vehicle_type = Column(String, nullable=False)
    ev_readiness_score = Column(Float, nullable=False)
    recommended_ev_replacement = Column(String, nullable=True)
    estimated_cost_usd = Column(Float, nullable=True)
    lead_time_months = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class BatteryHealthLog(Base):
    """
    SQL Schema tracking historical ML predictions for EV battery states of health.
    """
    __tablename__ = "battery_health_logs"

    id = Column(Integer, primary_key=True, index=True)
    battery_id = Column(String, index=True, nullable=False)
    cycle_number = Column(Integer, nullable=False)
    predicted_soh = Column(Float, nullable=False)
    predicted_rul_cycles = Column(Integer, nullable=False)
    health_zone = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
