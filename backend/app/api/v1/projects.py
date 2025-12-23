"""Project API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import os
import aiofiles
import json
from datetime import datetime

from app.db.session import get_db
from app.crud import project as project_crud
from app.schemas.project import (
    Project, ProjectCreate, ProjectUpdate, ProjectMapUpdate,
    ProjectList, ProjectSummary, MapData, ProcessingStatus
)
from app.core.config import settings

router = APIRouter()


# Extended file type support for CAD and image files
ALLOWED_FILE_TYPES = {
    # Images
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/svg+xml": ".svg",
    "application/pdf": ".pdf",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
    # CAD formats (DXF)
    "application/dxf": ".dxf",
    "image/vnd.dxf": ".dxf",
    "image/x-dxf": ".dxf",
    "application/x-dxf": ".dxf",
    # CAD formats (DWG)
    "application/acad": ".dwg",
    "application/x-acad": ".dwg",
    "application/dwg": ".dwg",
    "image/vnd.dwg": ".dwg",
    # CAD formats (IFC)
    "application/x-step": ".ifc",
    "model/step": ".ifc",
    "application/ifc": ".ifc",
}

# File extensions that are allowed (for fallback detection)
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.bmp', '.tiff', '.webp',
                      '.dxf', '.dwg', '.ifc'}


async def save_uploaded_file(upload_file: UploadFile, project_id: str) -> tuple[str, str]:
    """
    Save uploaded file and return both paths.

    Returns:
        Tuple of (full_file_path, static_url_path)
    """
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".png"
    filename = f"{project_id}_{timestamp}{ext}"
    filepath = os.path.join(settings.UPLOAD_PATH, filename)

    # Ensure directory exists
    os.makedirs(settings.UPLOAD_PATH, exist_ok=True)

    # Save file
    async with aiofiles.open(filepath, "wb") as f:
        content = await upload_file.read()
        await f.write(content)

    return filepath, f"/static/uploads/{filename}"


def validate_file_type(upload_file: UploadFile) -> bool:
    """Validate that the uploaded file is an allowed type."""
    # Check content type
    if upload_file.content_type and upload_file.content_type in ALLOWED_FILE_TYPES:
        return True

    # Fallback: check extension
    if upload_file.filename:
        ext = os.path.splitext(upload_file.filename)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            return True

    return False


@router.post("/", response_model=Project, status_code=201)
async def create_project(
    name: str = Form(..., min_length=1, max_length=255),
    description: Optional[str] = Form(None),
    scale: Optional[float] = Form(None, gt=0, description="Meters per pixel (optional - auto-detected if not provided)"),
    auto_process: bool = Form(True, description="Automatically process floor plan"),
    map_image: UploadFile = File(..., description="Floor plan file (PNG, JPG, PDF, SVG, DXF, DWG, IFC)"),
    db: Session = Depends(get_db)
):
    """
    Create a new project with a floor plan image or CAD file.

    The floor plan will be automatically processed to detect:
    - **Walls** - Using computer vision or CAD geometry extraction
    - **Rooms** - Segmented from enclosed spaces
    - **Scale** - Auto-detected from scale bars, dimensions, or reference objects
    - **Room labels** - Via OCR text extraction

    Supported formats:
    - **Images**: PNG, JPG, PDF, SVG, BMP, TIFF, WebP
    - **CAD**: DXF (AutoCAD Drawing Exchange), DWG (AutoCAD), IFC (BIM)

    - **name**: Project name
    - **description**: Optional description
    - **scale**: Scale factor (meters per pixel) - optional, auto-detected if not provided
    - **auto_process**: Whether to automatically process the floor plan (default: true)
    - **map_image**: Floor plan file
    """
    # Validate file type
    if not validate_file_type(map_image):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Check file size
    content = await map_image.read()
    await map_image.seek(0)  # Reset file pointer

    if len(content) > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE // (1024*1024)}MB"
        )

    # Create project with placeholder data first to get ID
    from uuid import uuid4
    project_id = str(uuid4())

    # Save uploaded file
    full_path, static_path = await save_uploaded_file(map_image, project_id)

    # Create empty map data (will be populated by processing)
    map_data = MapData(
        dimensions={"width": 800, "height": 600},  # Will be updated after image processing
        walls=[],
        rooms=[],
        forbidden_zones=[]
    )

    # Create project with pending status
    project_data = ProjectCreate(
        user_id=UUID(settings.DEFAULT_USER_ID),
        name=name,
        description=description,
        map_data=map_data,
        scale_meters_per_pixel=scale if scale else 0.05  # Default scale if not provided
    )

    project = project_crud.create_project(db, project_data, static_path)

    # Trigger async processing if enabled
    if auto_process:
        from app.tasks.map_processing_task import process_floor_plan
        process_floor_plan.delay(str(project.id), full_path, scale)

    return project


@router.get("/", response_model=ProjectList)
async def list_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    List all projects for the current user.

    Returns projects ordered by creation date (newest first).
    """
    user_id = UUID(settings.DEFAULT_USER_ID)  # MVP: hardcoded user

    projects = project_crud.get_projects(db, user_id, skip, limit)
    total = project_crud.get_projects_count(db, user_id)

    return ProjectList(items=projects, total=total, skip=skip, limit=limit)


@router.get("/{project_id}", response_model=Project)
async def get_project(
    project_id: UUID,
    db: Session = Depends(get_db)
):
    """Get a specific project by ID."""
    project = project_crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.put("/{project_id}", response_model=Project)
async def update_project(
    project_id: UUID,
    project_data: ProjectUpdate,
    db: Session = Depends(get_db)
):
    """Update project basic information (name, description, scale)."""
    project = project_crud.update_project(db, project_id, project_data)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.put("/{project_id}/map", response_model=Project)
async def update_project_map(
    project_id: UUID,
    map_update: ProjectMapUpdate,
    db: Session = Depends(get_db)
):
    """
    Update the map data for a project.

    Use this endpoint after manually editing walls in the frontend.
    The map_data includes:
    - **dimensions**: Width and height in pixels
    - **walls**: List of wall segments with materials
    - **rooms**: List of detected/defined rooms
    - **forbidden_zones**: Areas where routers cannot be placed
    """
    project = project_crud.update_project_map(db, project_id, map_update.map_data)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    project_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Delete a project.

    This will also delete all associated optimization jobs and solutions.
    """
    # Get project to find image path
    project = project_crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Delete the project (cascade deletes jobs and solutions)
    success = project_crud.delete_project(db, project_id)

    # Optionally delete the image file
    if project.map_image_path:
        file_path = os.path.join(settings.STATIC_PATH, project.map_image_path.lstrip("/static/"))
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass  # Non-critical error


@router.get("/{project_id}/processing-status", response_model=ProcessingStatus)
async def get_processing_status(
    project_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Get the current processing status for a project.

    Returns:
    - **status**: Current status (pending, processing, completed, failed, manual)
    - **progress**: Processing progress percentage (0-100)
    - **detected_scale**: Auto-detected scale (meters per pixel)
    - **scale_confidence**: Confidence score for detected scale (0-1)
    - **error**: Error message if processing failed
    """
    project = project_crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProcessingStatus(
        status=project.processing_status,
        progress=project.processing_progress,
        detected_scale=project.detected_scale,
        scale_confidence=project.scale_confidence,
        error=project.processing_error
    )


@router.post("/{project_id}/reprocess", response_model=ProcessingStatus)
async def reprocess_project(
    project_id: UUID,
    scale: Optional[float] = Form(None, description="Override scale (meters per pixel)"),
    db: Session = Depends(get_db)
):
    """
    Trigger reprocessing of a project's floor plan.

    Useful when:
    - Initial processing failed
    - You want to use a different scale
    - You've updated the floor plan image

    - **scale**: Optional scale override (meters per pixel)
    """
    project = project_crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.map_image_path:
        raise HTTPException(
            status_code=400,
            detail="Project has no floor plan file"
        )

    # Reset processing status
    project_crud.update_processing_status(db, project_id, "pending", 0)

    # Get full file path from static path
    full_path = os.path.join(settings.STATIC_PATH, project.map_image_path.lstrip("/static/"))

    # Trigger async processing
    from app.tasks.map_processing_task import process_floor_plan
    process_floor_plan.delay(str(project.id), full_path, scale)

    return ProcessingStatus(
        status="pending",
        progress=0,
        detected_scale=None,
        scale_confidence=None,
        error=None
    )


@router.post("/{project_id}/complete-walls-preview")
async def preview_complete_walls(
    project_id: UUID,
    max_gap: int = Query(100, ge=10, le=200, description="Maximum gap to auto-close (pixels)"),
    db: Session = Depends(get_db)
):
    """
    Preview wall completion without saving changes.

    Returns the original walls, processed walls, and new segments
    so the frontend can show a preview before applying.
    """
    project = project_crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.map_data or not project.map_data.get('walls'):
        raise HTTPException(
            status_code=400,
            detail="Project has no walls to complete"
        )

    from app.services.wall_post_processor import WallPostProcessor
    from app.schemas.project import WallSegment, Point

    # Get original walls
    original_walls = project.map_data.get('walls', [])

    # Convert dict walls to WallSegment objects
    walls = []
    for w in original_walls:
        if isinstance(w, dict):
            walls.append(WallSegment(
                start=Point(x=w['start']['x'], y=w['start']['y']),
                end=Point(x=w['end']['x'], y=w['end']['y']),
                thickness=w.get('thickness', 0.15),
                material=w.get('material', 'concrete'),
                attenuation_db=w.get('attenuation_db', 15.0)
            ))
        else:
            walls.append(w)

    # Run post-processing
    processor = WallPostProcessor(
        gap_threshold=max_gap,
        min_wall_length=30,
        isolation_threshold=max_gap + 20
    )
    processed = processor.process(walls)

    # Convert processed to dict format
    processed_walls = [
        {
            'start': {'x': w.start.x, 'y': w.start.y},
            'end': {'x': w.end.x, 'y': w.end.y},
            'thickness': w.thickness,
            'material': w.material,
            'attenuation_db': w.attenuation_db
        }
        for w in processed
    ]

    # Find new/changed segments by comparing
    def walls_match(w1, w2, tolerance=5):
        """Check if two walls are approximately the same."""
        return (
            abs(w1['start']['x'] - w2['start']['x']) < tolerance and
            abs(w1['start']['y'] - w2['start']['y']) < tolerance and
            abs(w1['end']['x'] - w2['end']['x']) < tolerance and
            abs(w1['end']['y'] - w2['end']['y']) < tolerance
        )

    # Find walls in processed that don't match any original
    new_segments = []
    for pw in processed_walls:
        is_new = True
        for ow in original_walls:
            if walls_match(pw, ow):
                is_new = False
                break
        if is_new:
            new_segments.append(pw)

    return {
        "original_walls": original_walls,
        "processed_walls": processed_walls,
        "new_segments": new_segments,
        "stats": {
            "original_count": len(original_walls),
            "final_count": len(processed_walls),
            "gaps_closed": len(new_segments)
        }
    }


@router.post("/{project_id}/complete-walls", response_model=Project)
async def complete_walls(
    project_id: UUID,
    max_gap: int = Query(100, ge=10, le=200, description="Maximum gap to auto-close (pixels)"),
    db: Session = Depends(get_db)
):
    """
    Auto-complete walls by closing detected gaps.

    This runs the gap-closing algorithm on existing walls without
    re-processing the floor plan image. Useful for:
    - Closing small gaps between wall segments
    - Connecting T-junctions where walls should meet
    - Creating closed room polygons

    - **max_gap**: Maximum gap size to bridge (10-200 pixels, default 100)
    """
    project = project_crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    if not project.map_data or not project.map_data.get('walls'):
        raise HTTPException(
            status_code=400,
            detail="Project has no walls to complete"
        )

    from app.services.wall_post_processor import WallPostProcessor
    from app.schemas.project import WallSegment, Point

    # Convert dict walls to WallSegment objects
    walls = []
    for w in project.map_data.get('walls', []):
        if isinstance(w, dict):
            walls.append(WallSegment(
                start=Point(x=w['start']['x'], y=w['start']['y']),
                end=Point(x=w['end']['x'], y=w['end']['y']),
                thickness=w.get('thickness', 0.15),
                material=w.get('material', 'concrete'),
                attenuation_db=w.get('attenuation_db', 15.0)
            ))
        else:
            walls.append(w)

    # Run post-processing with aggressive gap closure
    processor = WallPostProcessor(
        gap_threshold=max_gap,
        min_wall_length=30,
        isolation_threshold=max_gap + 20
    )
    processed = processor.process(walls)

    # Convert back to dict format
    processed_walls = [
        {
            'start': {'x': w.start.x, 'y': w.start.y},
            'end': {'x': w.end.x, 'y': w.end.y},
            'thickness': w.thickness,
            'material': w.material,
            'attenuation_db': w.attenuation_db
        }
        for w in processed
    ]

    # Update project map data
    new_map_data = {**project.map_data, 'walls': processed_walls}
    from app.schemas.project import MapData
    updated_project = project_crud.update_project_map(db, project_id, MapData(**new_map_data))

    return updated_project


@router.post("/{project_id}/image", response_model=Project)
async def upload_project_image(
    project_id: UUID,
    auto_process: bool = Form(True, description="Automatically process floor plan"),
    map_image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload or replace the floor plan image for a project.

    Supports all floor plan formats:
    - **Images**: PNG, JPG, PDF, SVG, BMP, TIFF, WebP
    - **CAD**: DXF, DWG, IFC
    """
    # Validate project exists
    project = project_crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate file type
    if not validate_file_type(map_image):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    # Delete old image if exists
    if project.map_image_path:
        old_path = os.path.join(settings.STATIC_PATH, project.map_image_path.lstrip("/static/"))
        if os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass

    # Save new file
    full_path, static_path = await save_uploaded_file(map_image, str(project_id))

    # Update project
    updated_project = project_crud.update_project_image(db, project_id, static_path)

    # Trigger processing if enabled
    if auto_process:
        project_crud.update_processing_status(db, project_id, "pending", 0)
        from app.tasks.map_processing_task import process_floor_plan
        process_floor_plan.delay(str(project_id), full_path, None)

    return updated_project
