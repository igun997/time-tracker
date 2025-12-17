from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional, List


# COCO class names for YOLOv8
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "tv", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush"
}

# Reverse mapping: name -> id
COCO_CLASS_IDS = {v: k for k, v in COCO_CLASSES.items()}

# Predefined class groups for convenience
CLASS_GROUPS = {
    "persons": [0],  # person only
    "vehicles": [2, 3, 5, 7],  # car, motorcycle, bus, truck
    "all_vehicles": [1, 2, 3, 5, 6, 7, 8],  # bicycle, car, motorcycle, bus, train, truck, boat
    "traffic": [0, 1, 2, 3, 5, 7, 9, 11],  # person, bicycle, car, motorcycle, bus, truck, traffic light, stop sign
    "animals": [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # bird through giraffe
}


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

    # Detection classes - comma-separated class names or IDs, or a group name
    # Examples: "person", "person,car,motorcycle", "vehicles", "traffic", "0,2,3,5,7"
    yolo_classes: str = "person"

    # Face Recognition
    face_recognition_tolerance: float = 0.6  # Lower = stricter matching
    face_recognition_model: str = "hog"  # "hog" (CPU) or "cnn" (GPU)

    # Tracking Mode
    # "presence" = track duration (how long objects stay)
    # "counter" = count entries/exits
    # "both" = track both duration and counts
    tracking_mode: str = "presence"
    track_all_objects: bool = True  # Track all detected classes, not just identified persons

    # Tracking Parameters
    track_timeout_seconds: int = 30  # Mark object as "exited" after this
    min_session_seconds: int = 3  # Ignore sessions shorter than this

    # Counter Mode Settings
    counter_hourly: bool = True  # Track hourly counts (False = daily only)

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


def parse_detection_classes(classes_str: str) -> List[int]:
    """
    Parse detection classes string into list of class IDs.

    Args:
        classes_str: Can be:
            - A group name: "vehicles", "traffic", "persons"
            - Comma-separated class names: "person,car,motorcycle"
            - Comma-separated class IDs: "0,2,3"
            - Mixed: "person,2,motorcycle"

    Returns:
        List of class IDs
    """
    classes_str = classes_str.strip().lower()

    # Check if it's a predefined group
    if classes_str in CLASS_GROUPS:
        return CLASS_GROUPS[classes_str]

    # Parse comma-separated values
    class_ids = []
    for item in classes_str.split(","):
        item = item.strip()
        if not item:
            continue

        # Try as number first
        try:
            class_id = int(item)
            if 0 <= class_id <= 79:
                class_ids.append(class_id)
        except ValueError:
            # Try as class name
            if item in COCO_CLASS_IDS:
                class_ids.append(COCO_CLASS_IDS[item])

    return class_ids if class_ids else [0]  # Default to person if nothing valid


def get_class_name(class_id: int) -> str:
    """Get class name from class ID"""
    return COCO_CLASSES.get(class_id, f"class_{class_id}")
