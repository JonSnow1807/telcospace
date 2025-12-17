"""Optimization API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from uuid import UUID
import asyncio
import json

from app.db.session import get_db
from app.crud import project as project_crud
from app.crud import optimization as optimization_crud
from app.crud import router as router_crud
from app.schemas.optimization import (
    OptimizationJob, OptimizationJobCreate, Solution, SolutionList,
    Feedback, FeedbackCreate, JobProgress
)

router = APIRouter()


@router.post("/jobs", response_model=OptimizationJob, status_code=201)
async def create_optimization_job(
    job_data: OptimizationJobCreate,
    db: Session = Depends(get_db)
):
    """
    Create and queue a new optimization job.

    The job will be processed asynchronously by a Celery worker.
    Use the WebSocket endpoint or polling to track progress.

    Constraints:
    - **max_routers**: Maximum number of routers to use
    - **max_budget**: Maximum total cost
    - **min_coverage_percent**: Minimum coverage percentage required (default: 80%)
    - **min_signal_strength_dbm**: Minimum signal strength in dBm (default: -70)
    - **allowed_router_ids**: Restrict to specific router models
    - **preferred_frequency**: Prefer specific frequency band
    """
    # Validate project exists
    project = project_crud.get_project(db, job_data.project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Validate allowed routers if specified
    if job_data.constraints.allowed_router_ids:
        routers = router_crud.get_routers_by_ids(db, job_data.constraints.allowed_router_ids)
        if len(routers) != len(job_data.constraints.allowed_router_ids):
            raise HTTPException(
                status_code=400,
                detail="One or more specified routers not found"
            )

    # Create job
    job = optimization_crud.create_job(db, job_data)

    # Queue the optimization task (Celery)
    try:
        from app.tasks.optimization_task import run_optimization_task
        run_optimization_task.delay(str(job.id))
    except Exception as e:
        # If Celery not available, mark as failed
        optimization_crud.update_job_status(
            db, job.id, "failed",
            error_message=f"Failed to queue task: {str(e)}"
        )
        raise HTTPException(
            status_code=503,
            detail="Background task system unavailable"
        )

    return job


@router.get("/jobs", response_model=List[OptimizationJob])
async def list_jobs(
    project_id: UUID = Query(..., description="Project ID to list jobs for"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """List all optimization jobs for a project."""
    # Validate project exists
    project = project_crud.get_project(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return optimization_crud.get_jobs_by_project(db, project_id, skip, limit)


@router.get("/jobs/{job_id}", response_model=OptimizationJob)
async def get_job(
    job_id: UUID,
    db: Session = Depends(get_db)
):
    """Get the status of an optimization job."""
    job = optimization_crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.get("/jobs/{job_id}/solutions", response_model=SolutionList)
async def get_job_solutions(
    job_id: UUID,
    rank: Optional[int] = Query(None, ge=1, description="Filter by rank"),
    db: Session = Depends(get_db)
):
    """
    Get all solutions for a completed optimization job.

    Solutions are ranked by fitness (rank 1 = best).
    Each solution includes:
    - Router placements with positions
    - Coverage percentage
    - Total cost
    - Signal strength metrics
    - Path to heatmap visualization
    """
    job = optimization_crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job not yet completed. Current status: {job.status}"
        )

    solutions = optimization_crud.get_solutions(db, job_id, rank)
    return SolutionList(items=solutions, job_id=job_id)


@router.get("/solutions/{solution_id}", response_model=Solution)
async def get_solution(
    solution_id: UUID,
    db: Session = Depends(get_db)
):
    """Get a specific solution by ID."""
    solution = optimization_crud.get_solution(db, solution_id)
    if not solution:
        raise HTTPException(status_code=404, detail="Solution not found")
    return solution


@router.post("/solutions/{solution_id}/feedback", response_model=Feedback, status_code=201)
async def submit_feedback(
    solution_id: UUID,
    feedback_data: FeedbackCreate,
    db: Session = Depends(get_db)
):
    """
    Submit feedback on a solution.

    Feedback helps improve the optimization algorithm over time.

    - **rating**: 1-5 star rating
    - **accuracy_score**: 0.0-1.0 accuracy compared to real measurements
    - **actual_measurements**: JSON with real signal strength readings
    - **comments**: Free-form feedback text
    """
    # Validate solution exists
    solution = optimization_crud.get_solution(db, solution_id)
    if not solution:
        raise HTTPException(status_code=404, detail="Solution not found")

    # Ensure solution_id matches
    feedback_data.solution_id = solution_id

    return optimization_crud.create_feedback(db, feedback_data)


@router.delete("/jobs/{job_id}", status_code=204)
async def cancel_job(
    job_id: UUID,
    db: Session = Depends(get_db)
):
    """
    Cancel a running or queued optimization job.

    Note: Jobs that are already running may complete before cancellation.
    """
    job = optimization_crud.get_job(db, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in ("completed", "failed"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job with status: {job.status}"
        )

    # Update status to failed with cancellation message
    optimization_crud.update_job_status(
        db, job_id, "failed",
        error_message="Job cancelled by user"
    )

    # TODO: Revoke Celery task if running
    # celery_app.control.revoke(str(job_id), terminate=True)


# ==================== WebSocket for Progress ====================

class ConnectionManager:
    """Manage WebSocket connections for job progress updates."""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]

    async def broadcast(self, job_id: str, message: dict):
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


manager = ConnectionManager()


@router.websocket("/ws/jobs/{job_id}")
async def websocket_job_progress(
    websocket: WebSocket,
    job_id: UUID
):
    """
    WebSocket endpoint for real-time job progress updates.

    Connect to receive progress updates as the optimization runs.
    Messages include:
    - progress_percent: 0-100
    - status: queued, running, completed, failed
    - message: Current operation description
    """
    job_id_str = str(job_id)
    await manager.connect(websocket, job_id_str)

    try:
        # Get initial status
        db = next(get_db())
        job = optimization_crud.get_job(db, job_id)

        if job:
            await websocket.send_json({
                "job_id": job_id_str,
                "status": job.status,
                "progress_percent": job.progress_percent,
                "message": "Connected to job progress stream"
            })

        # Keep connection alive and wait for updates
        # In production, this would subscribe to Redis pub/sub
        while True:
            try:
                # Wait for client messages (heartbeat)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )

                # Refresh job status
                db = next(get_db())
                job = optimization_crud.get_job(db, job_id)

                if job:
                    await websocket.send_json({
                        "job_id": job_id_str,
                        "status": job.status,
                        "progress_percent": job.progress_percent,
                        "message": "Status update"
                    })

                    # Close if job is done
                    if job.status in ("completed", "failed"):
                        break

            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket, job_id_str)
