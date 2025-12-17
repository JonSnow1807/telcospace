"""CRUD operations for routers."""

from sqlalchemy.orm import Session
from sqlalchemy import func, or_
from typing import List, Optional
from uuid import UUID

from app.models.router import Router
from app.schemas.router import RouterCreate, RouterUpdate


def get_router(db: Session, router_id: UUID) -> Optional[Router]:
    """Get a single router by ID."""
    return db.query(Router).filter(Router.id == router_id).first()


def get_routers(
    db: Session,
    skip: int = 0,
    limit: int = 100,
    manufacturer: Optional[str] = None,
    wifi_standard: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    search: Optional[str] = None
) -> List[Router]:
    """
    Get list of routers with optional filtering.

    Args:
        db: Database session
        skip: Number of records to skip
        limit: Maximum number of records to return
        manufacturer: Filter by manufacturer
        wifi_standard: Filter by WiFi standard
        min_price: Minimum price filter
        max_price: Maximum price filter
        search: Search term for model name or manufacturer
    """
    query = db.query(Router)

    # Apply filters
    if manufacturer:
        query = query.filter(Router.manufacturer.ilike(f"%{manufacturer}%"))

    if wifi_standard:
        query = query.filter(Router.wifi_standard == wifi_standard)

    if min_price is not None:
        query = query.filter(Router.price_usd >= min_price)

    if max_price is not None:
        query = query.filter(Router.price_usd <= max_price)

    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                Router.model_name.ilike(search_term),
                Router.manufacturer.ilike(search_term)
            )
        )

    # Order by price (lower first)
    query = query.order_by(Router.price_usd.asc().nullslast())

    return query.offset(skip).limit(limit).all()


def get_routers_count(
    db: Session,
    manufacturer: Optional[str] = None,
    wifi_standard: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    search: Optional[str] = None
) -> int:
    """Get total count of routers matching filters."""
    query = db.query(func.count(Router.id))

    if manufacturer:
        query = query.filter(Router.manufacturer.ilike(f"%{manufacturer}%"))

    if wifi_standard:
        query = query.filter(Router.wifi_standard == wifi_standard)

    if min_price is not None:
        query = query.filter(Router.price_usd >= min_price)

    if max_price is not None:
        query = query.filter(Router.price_usd <= max_price)

    if search:
        search_term = f"%{search}%"
        query = query.filter(
            or_(
                Router.model_name.ilike(search_term),
                Router.manufacturer.ilike(search_term)
            )
        )

    return query.scalar()


def create_router(db: Session, router: RouterCreate) -> Router:
    """Create a new router."""
    db_router = Router(
        model_name=router.model_name,
        manufacturer=router.manufacturer,
        frequency_bands=router.frequency_bands,
        max_tx_power_dbm=router.max_tx_power_dbm,
        antenna_gain_dbi=router.antenna_gain_dbi,
        wifi_standard=router.wifi_standard,
        max_range_meters=router.max_range_meters,
        coverage_area_sqm=router.coverage_area_sqm,
        price_usd=router.price_usd,
        specs=router.specs
    )
    db.add(db_router)
    db.commit()
    db.refresh(db_router)
    return db_router


def update_router(
    db: Session,
    router_id: UUID,
    router_update: RouterUpdate
) -> Optional[Router]:
    """Update an existing router."""
    db_router = get_router(db, router_id)
    if not db_router:
        return None

    update_data = router_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_router, field, value)

    db.commit()
    db.refresh(db_router)
    return db_router


def delete_router(db: Session, router_id: UUID) -> bool:
    """Delete a router."""
    db_router = get_router(db, router_id)
    if not db_router:
        return False

    db.delete(db_router)
    db.commit()
    return True


def get_routers_by_ids(db: Session, router_ids: List[UUID]) -> List[Router]:
    """Get multiple routers by their IDs."""
    return db.query(Router).filter(Router.id.in_(router_ids)).all()
