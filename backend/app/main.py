"""Main FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from app.core.config import settings, ensure_directories
from app.api.v1 import routers as router_routes
from app.api.v1 import projects as project_routes
from app.api.v1 import optimization as optimization_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    ensure_directories()
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    yield
    # Shutdown
    print("Shutting down application")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Optimize WiFi router placements for maximum coverage using physics-based RF simulation",
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving uploaded images and heatmaps
if os.path.exists(settings.STATIC_PATH):
    app.mount("/static", StaticFiles(directory=settings.STATIC_PATH), name="static")

# Include API routers
app.include_router(
    router_routes.router,
    prefix="/api/v1/routers",
    tags=["routers"]
)
app.include_router(
    project_routes.router,
    prefix="/api/v1/projects",
    tags=["projects"]
)
app.include_router(
    optimization_routes.router,
    prefix="/api/v1/optimization",
    tags=["optimization"]
)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint for container orchestration."""
    return {
        "status": "healthy",
        "version": settings.APP_VERSION
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
