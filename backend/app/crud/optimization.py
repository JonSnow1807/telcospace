"""CRUD operations for optimization jobs and solutions."""

from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from app.models.optimization import OptimizationJob, Solution, Feedback
from app.schemas.optimization import (
    OptimizationJobCreate,
    OptimizationConstraints,
    RouterPlacement,
    FeedbackCreate
)


# ==================== Optimization Jobs ====================

def get_job(db: Session, job_id: UUID) -> Optional[OptimizationJob]:
    """Get a single optimization job by ID."""
    return db.query(OptimizationJob).filter(OptimizationJob.id == job_id).first()


def get_jobs_by_project(
    db: Session,
    project_id: UUID,
    skip: int = 0,
    limit: int = 100
) -> List[OptimizationJob]:
    """Get list of optimization jobs for a project."""
    return (
        db.query(OptimizationJob)
        .filter(OptimizationJob.project_id == project_id)
        .order_by(OptimizationJob.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def create_job(
    db: Session,
    job_data: OptimizationJobCreate
) -> OptimizationJob:
    """Create a new optimization job."""
    db_job = OptimizationJob(
        project_id=job_data.project_id,
        constraints=job_data.constraints.model_dump(),
        status="queued",
        progress_percent=0
    )
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job


def update_job_status(
    db: Session,
    job_id: UUID,
    status: str,
    progress_percent: Optional[int] = None,
    error_message: Optional[str] = None
) -> Optional[OptimizationJob]:
    """Update job status and progress."""
    db_job = get_job(db, job_id)
    if not db_job:
        return None

    db_job.status = status

    if progress_percent is not None:
        db_job.progress_percent = progress_percent

    if error_message is not None:
        db_job.error_message = error_message

    # Set timestamps based on status
    if status == "running" and db_job.started_at is None:
        db_job.started_at = datetime.utcnow()
    elif status in ("completed", "failed"):
        db_job.completed_at = datetime.utcnow()

    db.commit()
    db.refresh(db_job)
    return db_job


def update_job_progress(
    db: Session,
    job_id: UUID,
    progress_percent: int
) -> Optional[OptimizationJob]:
    """Update job progress percentage."""
    db_job = get_job(db, job_id)
    if not db_job:
        return None

    db_job.progress_percent = progress_percent
    db.commit()
    db.refresh(db_job)
    return db_job


# ==================== Solutions ====================

def get_solution(db: Session, solution_id: UUID) -> Optional[Solution]:
    """Get a single solution by ID."""
    return db.query(Solution).filter(Solution.id == solution_id).first()


def get_solutions(
    db: Session,
    job_id: UUID,
    rank: Optional[int] = None
) -> List[Solution]:
    """Get solutions for an optimization job."""
    query = db.query(Solution).filter(Solution.job_id == job_id)

    if rank is not None:
        query = query.filter(Solution.rank == rank)

    return query.order_by(Solution.rank.asc()).all()


def create_solution(
    db: Session,
    job_id: UUID,
    router_placements: List[RouterPlacement],
    coverage_percentage: float,
    average_signal_strength: float,
    min_signal_strength: float,
    total_cost: Optional[float] = None,
    signal_heatmap_path: Optional[str] = None,
    rank: int = 1,
    metrics: Optional[dict] = None
) -> Solution:
    """Create a new solution."""
    # Convert to dict and ensure UUIDs are strings for JSON serialization
    placements_data = []
    for p in router_placements:
        d = p.model_dump()
        # Convert any UUID to string
        if 'router_id' in d and d['router_id'] is not None:
            d['router_id'] = str(d['router_id'])
        placements_data.append(d)

    db_solution = Solution(
        job_id=job_id,
        router_placements=placements_data,
        coverage_percentage=coverage_percentage,
        total_cost=total_cost,
        average_signal_strength=average_signal_strength,
        min_signal_strength=min_signal_strength,
        signal_heatmap_path=signal_heatmap_path,
        rank=rank,
        metrics=metrics
    )
    db.add(db_solution)
    db.commit()
    db.refresh(db_solution)
    return db_solution


def delete_solutions_for_job(db: Session, job_id: UUID) -> int:
    """Delete all solutions for a job. Returns count deleted."""
    count = db.query(Solution).filter(Solution.job_id == job_id).delete()
    db.commit()
    return count


# ==================== Feedback ====================

def get_feedback(db: Session, feedback_id: UUID) -> Optional[Feedback]:
    """Get a single feedback by ID."""
    return db.query(Feedback).filter(Feedback.id == feedback_id).first()


def get_feedback_for_solution(db: Session, solution_id: UUID) -> List[Feedback]:
    """Get all feedback for a solution."""
    return (
        db.query(Feedback)
        .filter(Feedback.solution_id == solution_id)
        .order_by(Feedback.submitted_at.desc())
        .all()
    )


def create_feedback(
    db: Session,
    feedback_data: FeedbackCreate
) -> Feedback:
    """Create new feedback for a solution."""
    db_feedback = Feedback(
        solution_id=feedback_data.solution_id,
        rating=feedback_data.rating,
        accuracy_score=feedback_data.accuracy_score,
        actual_measurements=feedback_data.actual_measurements,
        comments=feedback_data.comments
    )
    db.add(db_feedback)
    db.commit()
    db.refresh(db_feedback)
    return db_feedback
