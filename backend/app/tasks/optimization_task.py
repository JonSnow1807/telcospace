"""Celery task for running optimization jobs."""

from celery import Task
from sqlalchemy.orm import Session
from uuid import UUID
from typing import List
import os

from app.tasks.celery_app import celery_app
from app.db.session import SessionLocal
from app.crud import optimization as optimization_crud
from app.crud import project as project_crud
from app.crud import router as router_crud
from app.services.optimization import GeneticOptimizer
from app.services.rf_propagation import get_rf_engine, calculate_coverage_percentage
from app.services.heatmap_generator import generate_heatmap_image
from app.schemas.optimization import RouterPlacement, OptimizationConstraints
from app.schemas.project import MapData
from app.core.config import settings


class OptimizationTask(Task):
    """Base task class with database session management."""

    _db = None

    @property
    def db(self) -> Session:
        """Get database session, creating if needed."""
        if self._db is None:
            self._db = SessionLocal()
        return self._db

    def after_return(self, *args, **kwargs):
        """Clean up database session after task completes."""
        if self._db is not None:
            self._db.close()
            self._db = None


@celery_app.task(bind=True, base=OptimizationTask, name='app.tasks.optimization_task.run_optimization_task')
def run_optimization_task(self, job_id: str):
    """
    Background task to run optimization.

    Updates job status and stores results in database.
    """
    job_uuid = UUID(job_id)
    db = self.db

    try:
        # Get job
        job = optimization_crud.get_job(db, job_uuid)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        # Update status to running
        optimization_crud.update_job_status(db, job_uuid, "running", 0)

        # Get project data
        project = project_crud.get_project(db, job.project_id)
        if not project:
            raise ValueError(f"Project {job.project_id} not found")

        # Parse map data
        map_data = MapData(**project.map_data)

        # Parse constraints
        constraints = OptimizationConstraints(**job.constraints)

        # Get available routers
        if constraints.allowed_router_ids:
            routers = router_crud.get_routers_by_ids(db, constraints.allowed_router_ids)
        else:
            routers = router_crud.get_routers(db, limit=100)

        if not routers:
            raise ValueError("No routers available for optimization")

        # Create optimizer
        optimizer = GeneticOptimizer(
            map_data=map_data,
            available_routers=routers,
            constraints=constraints,
            scale=float(project.scale_meters_per_pixel),
            population_size=settings.GA_POPULATION_SIZE,
            generations=settings.GA_GENERATIONS,
            mutation_rate=settings.GA_MUTATION_RATE,
            crossover_rate=settings.GA_CROSSOVER_RATE
        )

        # Progress callback
        def progress_callback(gen: int, total_gen: int, fitness: float):
            progress = int((gen / total_gen) * 100)
            optimization_crud.update_job_progress(db, job_uuid, progress)

            # Update Celery task state
            self.update_state(
                state='PROGRESS',
                meta={
                    'percent': progress,
                    'generation': gen,
                    'fitness': fitness
                }
            )

        # Run optimization
        best_solutions = optimizer.optimize(progress_callback=progress_callback)

        # Get RF engine for final evaluation
        rf_engine = get_rf_engine()

        # Save solutions to database
        for rank, solution in enumerate(best_solutions, start=1):
            # Get router objects
            routers_used = [
                optimizer.router_map.get(rid)
                for rid in solution.router_ids
                if optimizer.router_map.get(rid) is not None
            ]

            # Calculate final signal coverage (using coarser grid for speed)
            signal_grid = rf_engine.predict_signal(
                map_data,
                solution.router_positions,
                routers_used,
                float(project.scale_meters_per_pixel),
                grid_resolution=20  # Coarser grid for faster computation
            )

            coverage = calculate_coverage_percentage(
                signal_grid,
                constraints.min_signal_strength_dbm
            )

            total_cost = sum(float(r.price_usd or 0) for r in routers_used)

            # Generate heatmap
            heatmap_filename = f"{job_id}_{rank}.png"
            heatmap_path = os.path.join(settings.HEATMAP_PATH, heatmap_filename)

            try:
                generate_heatmap_image(
                    signal_grid=signal_grid,
                    output_path=heatmap_path,
                    background_image=project.map_image_path.lstrip('/') if project.map_image_path else None,
                    router_positions=solution.router_positions
                )
            except Exception as e:
                # Log but don't fail the optimization
                import logging
                logging.warning(f"Heatmap generation failed: {e}")

            # Prepare router placements
            placements = []
            for rid, pos in zip(solution.router_ids, solution.router_positions):
                router = optimizer.router_map.get(rid)
                placements.append(RouterPlacement(
                    router_id=str(rid),  # Convert UUID to string for JSON serialization
                    x=float(pos[0]),
                    y=float(pos[1]),
                    rotation=0.0,
                    router_model=router.model_name if router else None,
                    router_manufacturer=router.manufacturer if router else None
                ))

            # Calculate signal statistics - ensure Python floats not numpy
            avg_signal = float(signal_grid.grid.mean())
            min_signal = float(signal_grid.grid.min())
            coverage_float = float(coverage)
            total_cost_float = float(total_cost) if total_cost else 0.0

            # Save solution
            optimization_crud.create_solution(
                db,
                job_id=job_uuid,
                router_placements=placements,
                coverage_percentage=coverage_float,
                total_cost=total_cost_float,
                average_signal_strength=avg_signal,
                min_signal_strength=min_signal,
                signal_heatmap_path=f"/static/heatmaps/{heatmap_filename}",
                rank=rank,
                metrics={
                    "fitness": float(solution.fitness),  # Convert numpy to Python float
                    "num_routers": len(routers_used),
                    "generation_count": settings.GA_GENERATIONS
                }
            )

        # Mark job as completed
        optimization_crud.update_job_status(db, job_uuid, "completed", 100)

        return {
            "job_id": job_id,
            "status": "completed",
            "solutions_count": len(best_solutions)
        }

    except Exception as e:
        # Mark job as failed
        error_message = str(e)
        try:
            optimization_crud.update_job_status(
                db,
                job_uuid,
                "failed",
                error_message=error_message
            )
        except Exception:
            pass  # Ignore errors during error handling

        # Re-raise to mark Celery task as failed
        raise


@celery_app.task(name='app.tasks.optimization_task.cleanup_old_jobs')
def cleanup_old_jobs():
    """
    Periodic task to clean up old optimization jobs.

    Removes jobs older than 30 days.
    """
    # TODO: Implement cleanup logic
    pass
