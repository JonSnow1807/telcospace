"""Router database model."""

from sqlalchemy import Column, String, Integer, Numeric, ARRAY, CheckConstraint, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class Router(Base):
    """
    Router model representing available WiFi router specifications.

    Stores technical specifications needed for RF propagation calculations.
    """
    __tablename__ = "routers"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String(255), nullable=False, index=True)
    manufacturer = Column(String(100), nullable=False, index=True)
    frequency_bands = Column(ARRAY(String), nullable=False)
    max_tx_power_dbm = Column(Integer, nullable=False)
    antenna_gain_dbi = Column(Numeric(4, 2), nullable=False)
    wifi_standard = Column(String(50), index=True)
    max_range_meters = Column(Integer)
    coverage_area_sqm = Column(Integer)
    price_usd = Column(Numeric(10, 2), index=True)
    specs = Column(JSONB)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    __table_args__ = (
        CheckConstraint('max_tx_power_dbm BETWEEN 0 AND 30', name='valid_tx_power'),
        CheckConstraint('antenna_gain_dbi BETWEEN -10 AND 20', name='valid_gain'),
    )

    def __repr__(self):
        return f"<Router(id={self.id}, model={self.model_name}, manufacturer={self.manufacturer})>"
