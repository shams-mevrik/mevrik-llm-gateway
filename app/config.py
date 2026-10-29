"""Server-level settings from environment variables."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Service identity
    service_name: str = "LLM Gateway"
    service_version: str = "0.0.0"
    service_description: str = "OpenAI-compatible gateway for vLLM backends"

    # Auth
    api_key: str = ""
    # Optional separate key for model management (add/update/delete), load
    # tests, stats reset and backend details. Empty = API_KEY has full access.
    admin_api_key: str = ""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "info"
    root_path: str = ""

    # CORS
    allowed_origins: str = "*"

    # Proxy tuning
    request_timeout: int = 300
    max_connections: int = 500
    # Timeout for each backend probe in GET /health/backends
    health_timeout: float = 5.0

    # Data paths
    data_dir: str = "data"
    models_path: str = "data/models.json"
    stats_path: str = "data/stats.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def cors_origins(self) -> List[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    def ensure_dirs(self):
        """Create data directories on startup."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.models_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
