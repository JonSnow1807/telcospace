"""Optimization job and solution database models."""

from sqlalchemy import Column, String, Text, Integer, Numeric, ForeignKey, CheckConstraint, DateTime
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from app.db.base import Base


class OptimizationJob(Base):
    """
    Optimization job model for tracking optimization runs.

    Jobs are processed asynchronously by Celery workers.
    """
    __tablename__ = "optimization_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    project_id = Column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    status = Column(String(50), nullable=False, default="queued", index=True)
    constraints = Column(JSONB, nullable=False)
    progress_percent = Column(Integer, default=0)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    project = relationship("Project", back_populates="optimization_jobs")
    solutions = relationship(
        "Solution",
        back_populates="job",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'failed')",
            name='valid_status'
        ),
        CheckConstraint(
            'progress_percent BETWEEN 0 AND 100',
            name='valid_progress'
        ),
    )

    def __repr__(self):
        return f"<OptimizationJob(id={self.id}, status={self.status})>"


class Solution(Base):
    """
    Solution model for storing optimization results.

    Each job can produce multiple solutions (Pareto front).
    """
    __tablename__ = "solutions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(
        UUID(as_uuid=True),
        ForeignKey("optimization_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    router_placements = Column(JSONB, nullable=False)
    coverage_percentage = Column(Numeric(5, 2), nullable=False)
    total_cost = Column(Numeric(10, 2))
    average_signal_strength = Column(Numeric(6, 2))
    min_signal_strength = Column(Numeric(6, 2))
    signal_heatmap_path = Column(Text)
    metrics = Column(JSONB)
    rank = Column(Integer, default=1, index=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    job = relationship("OptimizationJob", back_populates="solutions")
    feedback = relationship(
        "Feedback",
        back_populates="solution",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint(
            'coverage_percentage BETWEEN 0 AND 100',
            name='valid_coverage'
        ),
    )

    def __repr__(self):
        return f"<Solution(id={self.id}, coverage={self.coverage_percentage}%)>"


class Feedback(Base):
    """
    Feedback model for user feedback on solutions.

    Used for continuous improvement and future ML training.
    """
    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    solution_id = Column(
        UUID(as_uuid=True),
        ForeignKey("solutions.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )
    rating = Column(Integer)
    accuracy_score = Column(Numeric(3, 2))
    actual_measurements = Column(JSONB)
    comments = Column(Text)
    submitted_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # Relationships
    solution = relationship("Solution", back_populates="feedback")

    __table_args__ = (
        CheckConstraint('rating BETWEEN 1 AND 5', name='valid_rating'),
        CheckConstraint('accuracy_score BETWEEN 0 AND 1', name='valid_accuracy'),
    )

    def __repr__(self):
        return f"<Feedback(id={self.id}, rating={self.rating})>"
