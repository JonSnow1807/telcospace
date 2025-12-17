"""Project Pydantic schemas for API validation."""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class Point(BaseModel):
    """2D point coordinates."""
    x: float
    y: float


class WallSegment(BaseModel):
    """Wall segment in the floor plan."""
    start: Point
    end: Point
    thickness: float = Field(0.2, gt=0, description="Wall thickness in meters")
    material: str = Field(
        "concrete",
        description="Wall material: concrete, brick, glass, wood, drywall, metal"
    )
    attenuation_db: float = Field(
        15.0,
        description="RF signal attenuation in dB when passing through this wall"
    )


class Room(BaseModel):
    """Room definition in the floor plan."""
    name: str
    area: float = Field(..., gt=0, description="Room area in square meters")
    polygon: List[List[float]] = Field(
        ...,
        description="Room boundary as list of [x, y] coordinates"
    )


class ForbiddenZone(BaseModel):
    """Zone where routers cannot be placed."""
    name: str
    polygon: List[List[float]]
    reason: Optional[str] = None


class MapDimensions(BaseModel):
    """Floor plan dimensions in pixels."""
    width: int = Field(..., gt=0)
    height: int = Field(..., gt=0)


class MapData(BaseModel):
    """Structured representation of a floor plan."""
    dimensions: MapDimensions
    walls: List[WallSegment] = Field(default_factory=list)
    rooms: List[Room] = Field(default_factory=list)
    forbidden_zones: List[ForbiddenZone] = Field(default_factory=list)


class ProjectBase(BaseModel):
    """Base schema for project data."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    scale_meters_per_pixel: float = Field(..., gt=0)


class ProjectCreate(ProjectBase):
    """Schema for creating a new project."""
    user_id: UUID
    map_data: MapData


class ProjectUpdate(BaseModel):
    """Schema for updating an existing project."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    scale_meters_per_pixel: Optional[float] = Field(None, gt=0)


class ProjectMapUpdate(BaseModel):
    """Schema for updating project map data."""
    map_data: MapData


class Project(ProjectBase):
    """Schema for project response with all fields."""
    id: UUID
    user_id: UUID
    map_image_path: Optional[str] = None
    map_data: MapData

    # Processing status fields
    processing_status: str = "pending"
    processing_progress: int = 0
    detected_scale: Optional[float] = None
    scale_confidence: Optional[float] = None
    processing_error: Optional[str] = None

    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProcessingStatus(BaseModel):
    """Schema for processing status response."""
    status: str
    progress: int
    detected_scale: Optional[float] = None
    scale_confidence: Optional[float] = None
    error: Optional[str] = None


class ProjectList(BaseModel):
    """Schema for paginated project list response."""
    items: List[Project]
    total: int
    skip: int
    limit: int


class ProjectSummary(BaseModel):
    """Brief project summary for lists."""
    id: UUID
    name: str
    description: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
