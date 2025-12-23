"""CRUD operations for projects."""

from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from uuid import UUID

from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectUpdate, MapData


def get_project(db: Session, project_id: UUID) -> Optional[Project]:
    """Get a single project by ID."""
    return db.query(Project).filter(Project.id == project_id).first()


def get_projects(
    db: Session,
    user_id: UUID,
    skip: int = 0,
    limit: int = 100
) -> List[Project]:
    """Get list of projects for a user."""
    return (
        db.query(Project)
        .filter(Project.user_id == user_id)
        .order_by(Project.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_projects_count(db: Session, user_id: UUID) -> int:
    """Get total count of projects for a user."""
    return (
        db.query(func.count(Project.id))
        .filter(Project.user_id == user_id)
        .scalar()
    )


def create_project(
    db: Session,
    project: ProjectCreate,
    image_path: Optional[str] = None
) -> Project:
    """Create a new project."""
    db_project = Project(
        user_id=project.user_id,
        name=project.name,
        description=project.description,
        map_image_path=image_path,
        map_data=project.map_data.model_dump(),
        scale_meters_per_pixel=project.scale_meters_per_pixel
    )
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project


def update_project(
    db: Session,
    project_id: UUID,
    project_update: ProjectUpdate
) -> Optional[Project]:
    """Update an existing project's basic info."""
    db_project = get_project(db, project_id)
    if not db_project:
        return None

    update_data = project_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_project, field, value)

    db.commit()
    db.refresh(db_project)
    return db_project


def update_project_map(
    db: Session,
    project_id: UUID,
    map_data: MapData
) -> Optional[Project]:
    """Update a project's map data."""
    db_project = get_project(db, project_id)
    if not db_project:
        return None

    db_project.map_data = map_data.model_dump()
    db.commit()
    db.refresh(db_project)
    return db_project


def update_project_image(
    db: Session,
    project_id: UUID,
    image_path: str
) -> Optional[Project]:
    """Update a project's map image path."""
    db_project = get_project(db, project_id)
    if not db_project:
        return None

    db_project.map_image_path = image_path
    db.commit()
    db.refresh(db_project)
    return db_project


def delete_project(db: Session, project_id: UUID) -> bool:
    """Delete a project."""
    db_project = get_project(db, project_id)
    if not db_project:
        return False

    db.delete(db_project)
    db.commit()
    return True


def update_processing_status(
    db: Session,
    project_id: UUID,
    status: str,
    progress: Optional[int] = None,
    detected_scale: Optional[float] = None,
    scale_confidence: Optional[float] = None,
    error_message: Optional[str] = None
) -> Optional[Project]:
    """Update the processing status of a project."""
    db_project = get_project(db, project_id)
    if not db_project:
        return None

    db_project.processing_status = status

    if progress is not None:
        db_project.processing_progress = int(progress)

    if detected_scale is not None:
        # Convert numpy floats to native Python floats
        db_project.detected_scale = float(detected_scale)

    if scale_confidence is not None:
        # Convert numpy floats to native Python floats
        db_project.scale_confidence = float(scale_confidence)

    if error_message is not None:
        db_project.processing_error = error_message

    db.commit()
    db.refresh(db_project)
    return db_project


def update_project_llm_layout(
    db: Session,
    project_id: UUID,
    html_layout: str,
    svg_layout: str
) -> Optional[Project]:
    """Update a project's LLM-generated layout data."""
    db_project = get_project(db, project_id)
    if not db_project:
        return None

    # Store layouts in a metadata field or extended map_data
    current_map_data = db_project.map_data or {}
    current_map_data["llm_layout"] = {
        "html": html_layout,
        "svg": svg_layout
    }
    db_project.map_data = current_map_data

    db.commit()
    db.refresh(db_project)
    return db_project
