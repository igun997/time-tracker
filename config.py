from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Application
    app_name: str = "YOLO Time Tracker"
    debug: bool = True

    # Paths
    base_dir: Path = Path(__file__).parent
    models_dir: Path = base_dir / "models"
    uploads_dir: Path = base_dir / "uploads"
    data_dir: Path = base_dir / "data"

    # Database
    database_url: str = f"sqlite+aiosqlite:///{base_dir / 'data' / 'timetracker.db'}"

    # YOLO Detection
    yolo_model_path: Optional[str] = None  # Will look for models/yolov8n.pt if None
    yolo_confidence: float = 0.5
    yolo_device: str = "auto"  # "auto", "cpu", "cuda", or "cuda:0"

    # Face Recognition
    face_recognition_tolerance: float = 0.6  # Lower = stricter matching
    face_recognition_model: str = "hog"  # "hog" (CPU) or "cnn" (GPU)

    # Tracking
    track_timeout_seconds: int = 30  # Mark person as "exited" after this
    min_session_seconds: int = 5  # Ignore sessions shorter than this

    # Video Processing
    frame_skip: int = 2  # Process every Nth frame
    max_fps: int = 30

    # Streaming
    jpeg_quality: int = 80

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Ensure directories exist
settings.models_dir.mkdir(exist_ok=True)
settings.uploads_dir.mkdir(exist_ok=True)
settings.data_dir.mkdir(exist_ok=True)
