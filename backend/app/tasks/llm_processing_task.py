"""Background task for Simple LLM floor plan processing.

Simple approach: Pass image to LLM, get HTML back.
"""

import logging
from uuid import UUID
from typing import Optional

from celery import Task

from app.tasks.celery_app import celery_app
from app.db.session import SessionLocal
from app.crud import project as project_crud

logger = logging.getLogger(__name__)


class SimpleLLMTask(Task):
    """Base task for simple LLM floor plan processing."""

    _db = None
    _llm = None

    @property
    def db(self):
        """Lazy database session."""
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    @property
    def llm(self):
        """Lazy LLM initialization."""
        if self._llm is None:
            from app.services.simple_llm_floor_plan import get_simple_llm
            self._llm = get_simple_llm()
        return self._llm

    def after_return(self, status, retval, task_id, args, kwargs, einfo):
        """Cleanup after task completion."""
        if self._db:
            self._db.close()
            self._db = None


@celery_app.task(
    bind=True,
    base=SimpleLLMTask,
    name='app.tasks.llm_processing_task.process_floor_plan_llm',
    max_retries=3,
    default_retry_delay=60
)
def process_floor_plan_llm(
    self,
    project_id: str,
    file_path: str,
    user_scale: Optional[float] = None
):
    """
    Process floor plan using Simple LLM approach.
    
    Just pass the image to Gemini and ask it to generate HTML.

    Args:
        project_id: UUID of the project to update
        file_path: Path to the floor plan image
        user_scale: Optional user-provided scale (meters per pixel)

    Returns:
        Dictionary with processing results including HTML
    """
    project_uuid = UUID(project_id)
    db = self.db

    try:
        # Update status to processing
        project_crud.update_processing_status(
            db, project_uuid, "processing", 10
        )

        logger.info(f"Processing floor plan {project_id} with Simple LLM...")

        # Simple: pass image to LLM, get HTML back
        result = self.llm.analyze(file_path, user_scale=user_scale)

        project_crud.update_processing_status(
            db, project_uuid, "processing", 70
        )

        # Convert to MapData
        map_data = self.llm.to_map_data(result)

        # Update project with results
        if map_data and map_data.walls:
            project_crud.update_project_map(db, project_uuid, map_data)
            
        # Store HTML layout
        try:
            project_crud.update_project_llm_layout(
                db, project_uuid, result.html, result.html
            )
        except Exception as e:
            logger.warning(f"Could not store HTML layout: {e}")

        # Update processing status to completed
        project_crud.update_processing_status(
            db, project_uuid, "completed", 100,
            detected_scale=result.scale,
            scale_confidence=0.8
        )

        return {
            "status": "completed",
            "project_id": project_id,
            "detected_scale": result.scale,
            "scale_confidence": 0.8,
            "scale_method": "llm_gemini",
            "walls_detected": len(result.walls),
            "rooms_detected": len(result.rooms),
            "html_layout": result.html,
            "width_m": result.width_m,
            "height_m": result.height_m,
            "warnings": []
        }

    except Exception as e:
        logger.error(f"LLM floor plan processing failed for project {project_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())

        # Update status to failed
        project_crud.update_processing_status(
            db, project_uuid, "failed",
            error_message=str(e)
        )

        # Retry on certain errors
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)

        raise
