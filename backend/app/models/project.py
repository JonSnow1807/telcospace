"""Project database model."""

from sqlalchemy import Column, String, Text, Numeric, Integer, Float, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class Project(Base):
    """
    Project model representing a floor plan optimization project.

    Contains the uploaded floor plan image and extracted map data.
    """
    __tablename__ = "projects"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    map_image_path = Column(Text)
    map_data = Column(JSONB, nullable=False)
    scale_meters_per_pixel = Column(Numeric(10, 6), nullable=False)

    # Processing status fields
    processing_status = Column(String(50), default="pending", nullable=False)
    # Values: pending, processing, completed, failed, manual
    processing_progress = Column(Integer, default=0, nullable=False)  # 0-100
    detected_scale = Column(Float, nullable=True)  # Auto-detected scale
    scale_confidence = Column(Float, nullable=True)  # 0-1 confidence score
    processing_error = Column(Text, nullable=True)  # Error message if failed

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

    # Relationships
    optimization_jobs = relationship(
        "OptimizationJob",
        back_populates="project",
        cascade="all, delete-orphan"
    )

    def __repr__(self):
        return f"<Project(id={self.id}, name={self.name})>"
