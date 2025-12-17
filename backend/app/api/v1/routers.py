"""Router API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID

from app.db.session import get_db
from app.crud import router as router_crud
from app.schemas.router import Router, RouterCreate, RouterUpdate, RouterList

router = APIRouter()


@router.get("/", response_model=RouterList)
async def list_routers(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=100, description="Maximum records to return"),
    manufacturer: Optional[str] = Query(None, description="Filter by manufacturer"),
    wifi_standard: Optional[str] = Query(None, description="Filter by WiFi standard"),
    min_price: Optional[float] = Query(None, ge=0, description="Minimum price"),
    max_price: Optional[float] = Query(None, ge=0, description="Maximum price"),
    search: Optional[str] = Query(None, description="Search in model name and manufacturer"),
    db: Session = Depends(get_db)
):
    """
    List all available routers with optional filtering.

    - **manufacturer**: Filter by manufacturer name (partial match)
    - **wifi_standard**: Filter by exact WiFi standard (WiFi 5, WiFi 6, WiFi 6E)
    - **min_price, max_price**: Price range filter
    - **search**: Full-text search on model name and manufacturer
    """
    routers = router_crud.get_routers(
        db,
        skip=skip,
        limit=limit,
        manufacturer=manufacturer,
        wifi_standard=wifi_standard,
        min_price=min_price,
        max_price=max_price,
        search=search
    )

    total = router_crud.get_routers_count(
        db,
        manufacturer=manufacturer,
        wifi_standard=wifi_standard,
        min_price=min_price,
        max_price=max_price,
        search=search
    )

    return RouterList(items=routers, total=total, skip=skip, limit=limit)


@router.get("/{router_id}", response_model=Router)
async def get_router(
    router_id: UUID,
    db: Session = Depends(get_db)
):
    """Get a specific router by ID."""
    db_router = router_crud.get_router(db, router_id)
    if not db_router:
        raise HTTPException(status_code=404, detail="Router not found")
    return db_router


@router.post("/", response_model=Router, status_code=201)
async def create_router(
    router_data: RouterCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new router entry.

    Router specifications are used for RF propagation calculations.
    Required fields:
    - **model_name**: Router model name
    - **manufacturer**: Manufacturer name
    - **frequency_bands**: List of supported frequencies (2.4GHz, 5GHz, 6GHz)
    - **max_tx_power_dbm**: Maximum transmit power in dBm (0-30)
    - **antenna_gain_dbi**: Antenna gain in dBi (-10 to 20)
    """
    return router_crud.create_router(db, router_data)


@router.put("/{router_id}", response_model=Router)
async def update_router(
    router_id: UUID,
    router_data: RouterUpdate,
    db: Session = Depends(get_db)
):
    """Update an existing router."""
    db_router = router_crud.update_router(db, router_id, router_data)
    if not db_router:
        raise HTTPException(status_code=404, detail="Router not found")
    return db_router


@router.delete("/{router_id}", status_code=204)
async def delete_router(
    router_id: UUID,
    db: Session = Depends(get_db)
):
    """Delete a router."""
    success = router_crud.delete_router(db, router_id)
    if not success:
        raise HTTPException(status_code=404, detail="Router not found")


@router.get("/by-ids/batch", response_model=List[Router])
async def get_routers_by_ids(
    ids: List[UUID] = Query(..., description="List of router IDs"),
    db: Session = Depends(get_db)
):
    """Get multiple routers by their IDs."""
    return router_crud.get_routers_by_ids(db, ids)
