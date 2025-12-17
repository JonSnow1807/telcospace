"""Application configuration settings."""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "WiFi Router Placement Optimizer"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # AI/LLM (for AI-powered wall detection)
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None

    # AI Model Selection: "opus" for max accuracy, "sonnet" for speed/cost balance
    AI_WALL_DETECTION_MODEL: str = "opus"  # Options: "opus", "sonnet"

    # Database
    DATABASE_URL: str = "postgresql://router_user:router_password@localhost:5432/router_optimizer"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:3000,http://localhost:5173"

    @property
    def allowed_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.ALLOWED_ORIGINS.split(",")]

    # File storage
    STATIC_PATH: str = "./static"
    UPLOAD_PATH: str = "./static/uploads"
    HEATMAP_PATH: str = "./static/heatmaps"
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB

    # RF Propagation
    DEFAULT_FREQUENCY_GHZ: float = 2.4
    GRID_RESOLUTION: int = 2  # pixels per grid cell

    # Optimization
    GA_POPULATION_SIZE: int = 20
    GA_GENERATIONS: int = 20
    GA_MUTATION_RATE: float = 0.15
    GA_CROSSOVER_RATE: float = 0.7

    # Default user ID for MVP (no auth)
    DEFAULT_USER_ID: str = "00000000-0000-0000-0000-000000000001"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs(settings.UPLOAD_PATH, exist_ok=True)
    os.makedirs(settings.HEATMAP_PATH, exist_ok=True)
