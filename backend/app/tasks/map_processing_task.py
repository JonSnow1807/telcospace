"""Background task for processing floor plan images and CAD files."""

import asyncio
import logging
from uuid import UUID
from typing import Optional

from celery import Task, states

from app.tasks.celery_app import celery_app
from app.db.session import SessionLocal
from app.crud import project as project_crud
from app.services.floor_plan_processor import FloorPlanProcessor, ProcessingResult

logger = logging.getLogger(__name__)


class MapProcessingTask(Task):
    """Base task with database session management."""

    _db = None
    _processor = None

    @property
    def db(self):
        """Lazy database session."""
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    @property
    def processor(self):
        """Lazy floor plan processor."""
        if self._processor is None:
            self._processor = FloorPlanProcessor()
        return self._processor

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """Cleanup after task completion."""
        if self._db:
            self._db.close()
            self._db = None


@celery_app.task(
    bind=True,
    base=MapProcessingTask,
    name='app.tasks.map_processing_task.process_floor_plan',
    max_retries=3,
    default_retry_delay=60
)
def process_floor_plan(
    self,
    project_id: str,
    file_path: str,
    user_scale: Optional[float] = None
):
    """
    Background task to process a floor plan file.

    Args:
        project_id: UUID of the project to update
        file_path: Path to the floor plan file
        user_scale: Optional user-provided scale (meters per pixel)

    Returns:
        Dictionary with processing results
    """
    project_uuid = UUID(project_id)
    db = self.db

    try:
        # Update status to processing
        project_crud.update_processing_status(
            db, project_uuid, "processing", 0
        )

        def progress_callback(percent: int, message: str):
            """Update progress in database and Celery state."""
            project_crud.update_processing_status(
                db, project_uuid, "processing", percent
            )
            self.update_state(
                state='PROGRESS',
                meta={
                    'percent': percent,
                    'message': message,
                    'project_id': project_id
                }
            )

        # Run async processing in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(
                self.processor.process(file_path, progress_callback, user_scale)
            )
        finally:
            loop.close()

        # Update project with results
        if result.map_data:
            project_crud.update_project_map(
                db, project_uuid, result.map_data
            )

        # Update processing status to completed
        project_crud.update_processing_status(
            db, project_uuid, "completed", 100,
            detected_scale=result.detected_scale,
            scale_confidence=result.scale_confidence
        )

        return {
            "status": "completed",
            "project_id": project_id,
            "detected_scale": result.detected_scale,
            "scale_confidence": result.scale_confidence,
            "scale_method": result.scale_method,
            "walls_detected": len(result.map_data.walls) if result.map_data else 0,
            "rooms_detected": len(result.map_data.rooms) if result.map_data else 0,
            "warnings": result.warnings
        }

    except Exception as e:
        logger.error(f"Floor plan processing failed for project {project_id}: {e}")

        # Update status to failed
        project_crud.update_processing_status(
            db, project_uuid, "failed",
            error_message=str(e)
        )

        # Retry on certain errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

        raise


@celery_app.task(
    bind=True,
    base=MapProcessingTask,
    name='app.tasks.map_processing_task.reprocess_floor_plan'
)
def reprocess_floor_plan(
    self,
    project_id: str,
    user_scale: Optional[float] = None
):
    """
    Reprocess an existing project's floor plan.

    Used when user wants to re-run processing with different settings.

    Args:
        project_id: UUID of the project
        user_scale: Optional new scale to use

    Returns:
        Dictionary with processing results
    """
    project_uuid = UUID(project_id)
    db = self.db

    # Get project to find file path
    project = project_crud.get_project(db, project_uuid)
    if not project:
        raise ValueError(f"Project not found: {project_id}")

    if not project.image_path:
        raise ValueError(f"Project has no floor plan file: {project_id}")

    # Process with existing file
    return process_floor_plan(self, project_id, project.image_path, user_scale)


@celery_app.task(name='app.tasks.map_processing_task.get_processing_status')
def get_processing_status(project_id: str) -> dict:
    """
    Get the current processing status for a project.

    Args:
        project_id: UUID of the project

    Returns:
        Dictionary with status information
    """
    project_uuid = UUID(project_id)
    db = SessionLocal()

    try:
        project = project_crud.get_project(db, project_uuid)
        if not project:
            return {"error": "Project not found"}

        return {
            "project_id": project_id,
            "status": project.processing_status,
            "progress": project.processing_progress,
            "detected_scale": project.detected_scale,
            "scale_confidence": project.scale_confidence,
            "error": project.processing_error
        }
    finally:
        db.close()
