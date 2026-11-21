import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("ACOUSTICASE_DATA_DIR", "data")))
    model_dir: Path | None = field(default_factory=lambda: Path(os.environ["ACOUSTICASE_YAMNET_DIR"]) if os.getenv("ACOUSTICASE_YAMNET_DIR") else None)
    max_upload_bytes: int = 100 * 1024 * 1024
    max_duration_s: float = 300
    max_pending_jobs: int = 4
    cors_origins: list[str] = field(default_factory=lambda: os.getenv(
        "ACOUSTICASE_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://localhost:8501,http://127.0.0.1:3000,http://127.0.0.1:5173,http://127.0.0.1:8501",
    ).split(","))
