"""Optimization Pydantic schemas for API validation."""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class OptimizationConstraints(BaseModel):
    """User-defined constraints for optimization."""
    max_routers: Optional[int] = Field(None, ge=1, le=20)
    max_budget: Optional[float] = Field(None, ge=0)
    min_coverage_percent: float = Field(80.0, ge=0, le=100)
    min_signal_strength_dbm: float = Field(-70.0, le=0)
    allowed_router_ids: Optional[List[UUID]] = None
    preferred_frequency: Optional[str] = Field(
        None,
        description="Preferred frequency: '2.4GHz', '5GHz', '6GHz', or 'dual-band'"
    )
    # Additional constraints
    prioritize_cost: bool = Field(False, description="Prioritize lower cost over coverage")
    require_mesh_support: bool = Field(False, description="Only use mesh-capable routers")


class RouterPlacement(BaseModel):
    """Single router placement in a solution."""
    router_id: UUID
    x: float = Field(..., description="X position in pixels")
    y: float = Field(..., description="Y position in pixels")
    rotation: float = Field(0.0, ge=0, lt=360, description="Rotation in degrees")

    # Computed fields (optional, filled by backend)
    router_model: Optional[str] = None
    router_manufacturer: Optional[str] = None


class OptimizationJobCreate(BaseModel):
    """Schema for creating a new optimization job."""
    project_id: UUID
    constraints: OptimizationConstraints


class OptimizationJobUpdate(BaseModel):
    """Schema for updating job status (internal use)."""
    status: Optional[str] = None
    progress_percent: Optional[int] = Field(None, ge=0, le=100)
    error_message: Optional[str] = None


class OptimizationJob(BaseModel):
    """Schema for optimization job response."""
    id: UUID
    project_id: UUID
    status: str
    constraints: OptimizationConstraints
    progress_percent: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SolutionMetrics(BaseModel):
    """Additional metrics for a solution."""
    fitness_score: Optional[float] = None
    num_routers: Optional[int] = None
    signal_uniformity: Optional[float] = None
    weak_spots_count: Optional[int] = None
    overlap_percentage: Optional[float] = None


class Solution(BaseModel):
    """Schema for optimization solution response."""
    id: UUID
    job_id: UUID
    router_placements: List[RouterPlacement]
    coverage_percentage: float
    total_cost: Optional[float] = None
    average_signal_strength: float
    min_signal_strength: float
    signal_heatmap_path: Optional[str] = None
    metrics: Optional[SolutionMetrics] = None
    rank: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SolutionList(BaseModel):
    """Schema for list of solutions."""
    items: List[Solution]
    job_id: UUID


class FeedbackCreate(BaseModel):
    """Schema for submitting feedback on a solution."""
    solution_id: UUID
    rating: Optional[int] = Field(None, ge=1, le=5)
    accuracy_score: Optional[float] = Field(None, ge=0, le=1)
    actual_measurements: Optional[Dict[str, Any]] = None
    comments: Optional[str] = None


class Feedback(BaseModel):
    """Schema for feedback response."""
    id: UUID
    solution_id: UUID
    rating: Optional[int] = None
    accuracy_score: Optional[float] = None
    actual_measurements: Optional[Dict[str, Any]] = None
    comments: Optional[str] = None
    submitted_at: datetime

    model_config = ConfigDict(from_attributes=True)


class JobProgress(BaseModel):
    """WebSocket message for job progress updates."""
    job_id: UUID
    status: str
    progress_percent: int
    message: Optional[str] = None
    current_generation: Optional[int] = None
    best_fitness: Optional[float] = None
