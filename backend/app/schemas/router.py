"""Router Pydantic schemas for API validation."""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime


class RouterBase(BaseModel):
    """Base schema for router data."""
    model_name: str = Field(..., min_length=1, max_length=255)
    manufacturer: str = Field(..., min_length=1, max_length=100)
    frequency_bands: List[str] = Field(..., min_length=1)
    max_tx_power_dbm: int = Field(..., ge=0, le=30)
    antenna_gain_dbi: float = Field(..., ge=-10, le=20)
    wifi_standard: Optional[str] = Field(None, max_length=50)
    max_range_meters: Optional[int] = Field(None, ge=0)
    coverage_area_sqm: Optional[int] = Field(None, ge=0)
    price_usd: Optional[float] = Field(None, ge=0)
    specs: Optional[Dict[str, Any]] = None


class RouterCreate(RouterBase):
    """Schema for creating a new router."""
    pass


class RouterUpdate(BaseModel):
    """Schema for updating an existing router."""
    model_name: Optional[str] = Field(None, min_length=1, max_length=255)
    manufacturer: Optional[str] = Field(None, min_length=1, max_length=100)
    frequency_bands: Optional[List[str]] = Field(None, min_length=1)
    max_tx_power_dbm: Optional[int] = Field(None, ge=0, le=30)
    antenna_gain_dbi: Optional[float] = Field(None, ge=-10, le=20)
    wifi_standard: Optional[str] = Field(None, max_length=50)
    max_range_meters: Optional[int] = Field(None, ge=0)
    coverage_area_sqm: Optional[int] = Field(None, ge=0)
    price_usd: Optional[float] = Field(None, ge=0)
    specs: Optional[Dict[str, Any]] = None


class Router(RouterBase):
    """Schema for router response with all fields."""
    id: UUID
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class RouterList(BaseModel):
    """Schema for paginated router list response."""
    items: List[Router]
    total: int
    skip: int
    limit: int
