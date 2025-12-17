from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


@dataclass
class PersonDetection:
    """Represents a detected person in a frame"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    track_id: Optional[int] = None

    @property
    def x1(self) -> int:
        return self.bbox[0]

    @property
    def y1(self) -> int:
        return self.bbox[1]

    @property
    def x2(self) -> int:
        return self.bbox[2]

    @property
    def y2(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def crop_from_frame(self, frame: np.ndarray, padding: int = 0) -> np.ndarray:
        """Extract the person region from frame with optional padding"""
        h, w = frame.shape[:2]
        x1 = max(0, self.x1 - padding)
        y1 = max(0, self.y1 - padding)
        x2 = min(w, self.x2 + padding)
        y2 = min(h, self.y2 + padding)
        return frame[y1:y2, x1:x2]


class YOLODetector:
    """YOLO-based person detector"""

    # COCO class ID for person
    PERSON_CLASS_ID = 0

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.5,
        device: str = "auto"
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._is_loaded = False

    def load_model(self) -> bool:
        """Load the YOLO model"""
        if not YOLO_AVAILABLE:
            print("Warning: ultralytics not installed. Running in demo mode.")
            return False

        try:
            if self.model_path and Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
            else:
                # Try to find model in models directory
                models_dir = Path(__file__).parent.parent.parent / "models"
                default_model = models_dir / "yolov8n.pt"

                if default_model.exists():
                    self.model = YOLO(str(default_model))
                else:
                    print(f"Warning: No YOLO model found at {default_model}")
                    print("Please download yolov8n.pt and place it in the models/ directory")
                    return False

            # Set device
            if self.device == "auto":
                # Let ultralytics decide (will use CUDA if available)
                pass
            else:
                self.model.to(self.device)

            self._is_loaded = True
            return True

        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            return False

    def detect(self, frame: np.ndarray, use_tracking: bool = False) -> List[PersonDetection]:
        """
        Detect persons in a frame

        Args:
            frame: BGR image as numpy array
            use_tracking: If True, use YOLO's built-in tracking

        Returns:
            List of PersonDetection objects
        """
        if not self._is_loaded or self.model is None:
            return self._demo_detect(frame)

        try:
            if use_tracking:
                results = self.model.track(
                    frame,
                    persist=True,
                    classes=[self.PERSON_CLASS_ID],
                    conf=self.confidence_threshold,
                    verbose=False
                )
            else:
                results = self.model(
                    frame,
                    classes=[self.PERSON_CLASS_ID],
                    conf=self.confidence_threshold,
                    verbose=False
                )

            detections = []
            for result in results:
                if result.boxes is None:
                    continue

                for i, box in enumerate(result.boxes):
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])

                    # Get track ID if available
                    track_id = None
                    if use_tracking and box.id is not None:
                        track_id = int(box.id[0])

                    detections.append(PersonDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        track_id=track_id
                    ))

            return detections

        except Exception as e:
            print(f"Detection error: {e}")
            return []

    def _demo_detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """
        Demo detection for testing without YOLO model.
        Returns simulated detections.
        """
        # Return empty list in demo mode
        # In real usage, you might want to return mock detections for UI testing
        return []

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    @property
    def device_info(self) -> str:
        if self.model is not None:
            return str(self.model.device)
        return "not loaded"


class DemoDetector:
    """
    Demo detector that generates fake detections for UI testing
    when YOLO model is not available.
    """

    def __init__(self):
        self._frame_count = 0
        self._mock_persons = []
        self._is_loaded = True

    def load_model(self) -> bool:
        return True

    def detect(self, frame: np.ndarray, use_tracking: bool = False) -> List[PersonDetection]:
        """Generate mock detections for testing"""
        self._frame_count += 1

        # Generate some mock detections that move around
        h, w = frame.shape[:2]
        detections = []

        # Simulate 1-3 persons with semi-random positions
        import math

        for i in range(2):
            # Create oscillating positions
            t = self._frame_count * 0.05
            offset = i * 2

            cx = int(w * 0.3 + w * 0.2 * math.sin(t + offset))
            cy = int(h * 0.5 + h * 0.1 * math.cos(t * 0.7 + offset))

            # Person-sized bounding box
            bw, bh = 100, 200

            detections.append(PersonDetection(
                bbox=(cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2),
                confidence=0.85 + 0.1 * math.sin(t),
                track_id=i + 1 if use_tracking else None
            ))

        return detections

    @property
    def is_loaded(self) -> bool:
        return True

    @property
    def device_info(self) -> str:
        return "demo"


def create_detector(
    model_path: Optional[str] = None,
    confidence_threshold: float = 0.5,
    device: str = "auto",
    demo_mode: bool = False
) -> YOLODetector:
    """
    Factory function to create appropriate detector

    Args:
        model_path: Path to YOLO model file
        confidence_threshold: Detection confidence threshold
        device: Device to use ("auto", "cpu", "cuda")
        demo_mode: If True, return demo detector for testing

    Returns:
        Detector instance
    """
    if demo_mode:
        return DemoDetector()

    detector = YOLODetector(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        device=device
    )

    return detector
