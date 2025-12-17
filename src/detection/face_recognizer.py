from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False


@dataclass
class FaceMatch:
    """Result of face recognition"""
    employee_id: Optional[int]
    employee_name: Optional[str]
    confidence: float  # 1.0 - distance (higher is better match)
    face_location: Tuple[int, int, int, int]  # (top, right, bottom, left)
    is_known: bool

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Convert face_location to (x1, y1, x2, y2) format"""
        top, right, bottom, left = self.face_location
        return (left, top, right, bottom)


class FaceRecognizer:
    """Face recognition for employee identification"""

    def __init__(self, tolerance: float = 0.6, model: str = "hog"):
        """
        Initialize face recognizer

        Args:
            tolerance: Face matching tolerance (lower = stricter)
            model: "hog" (CPU) or "cnn" (GPU)
        """
        self.tolerance = tolerance
        self.model = model
        self._known_encodings: List[np.ndarray] = []
        self._known_employee_ids: List[int] = []
        self._known_names: List[str] = []
        self._is_available = FACE_RECOGNITION_AVAILABLE

    @property
    def is_available(self) -> bool:
        return self._is_available

    def load_known_faces(self, faces_data: List[Tuple[int, str, np.ndarray]]):
        """
        Load known faces from database

        Args:
            faces_data: List of (employee_id, name, encoding_array) tuples
        """
        self._known_encodings = []
        self._known_employee_ids = []
        self._known_names = []

        for employee_id, name, encoding in faces_data:
            self._known_encodings.append(encoding)
            self._known_employee_ids.append(employee_id)
            self._known_names.append(name)

    def add_known_face(self, employee_id: int, name: str, encoding: np.ndarray):
        """Add a single known face"""
        self._known_encodings.append(encoding)
        self._known_employee_ids.append(employee_id)
        self._known_names.append(name)

    def encode_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Get face encoding from an image

        Args:
            image: RGB image containing a face

        Returns:
            Face encoding array or None if no face found
        """
        if not self._is_available:
            return None

        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = image[:, :, ::-1]  # BGR to RGB
            else:
                rgb_image = image

            # Find face locations
            face_locations = face_recognition.face_locations(rgb_image, model=self.model)

            if not face_locations:
                return None

            # Get encoding for the first (largest) face
            encodings = face_recognition.face_encodings(rgb_image, face_locations)

            if encodings:
                return encodings[0]
            return None

        except Exception as e:
            print(f"Error encoding face: {e}")
            return None

    def recognize_faces_in_frame(self, frame: np.ndarray) -> List[FaceMatch]:
        """
        Detect and recognize all faces in a frame

        Args:
            frame: BGR image

        Returns:
            List of FaceMatch objects
        """
        if not self._is_available:
            return []

        try:
            # Convert BGR to RGB
            rgb_frame = frame[:, :, ::-1]

            # Find all faces in frame
            face_locations = face_recognition.face_locations(rgb_frame, model=self.model)

            if not face_locations:
                return []

            # Get encodings for all faces
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            results = []
            for face_location, face_encoding in zip(face_locations, face_encodings):
                match = self._match_face(face_encoding, face_location)
                results.append(match)

            return results

        except Exception as e:
            print(f"Error recognizing faces: {e}")
            return []

    def recognize_face_in_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[FaceMatch]:
        """
        Recognize face in a specific region (from person detection)

        Args:
            frame: BGR image
            bbox: (x1, y1, x2, y2) bounding box of person

        Returns:
            FaceMatch or None if no face found
        """
        if not self._is_available:
            return None

        try:
            x1, y1, x2, y2 = bbox
            # Crop the region
            person_crop = frame[y1:y2, x1:x2]

            if person_crop.size == 0:
                return None

            # Convert to RGB
            rgb_crop = person_crop[:, :, ::-1]

            # Find faces in the cropped region
            face_locations = face_recognition.face_locations(rgb_crop, model=self.model)

            if not face_locations:
                return None

            # Get encoding for the first face
            try:
                face_encodings = face_recognition.face_encodings(rgb_crop, face_locations, num_jitters=1)
            except Exception as enc_error:
                # Fallback: try without specifying locations
                try:
                    face_encodings = face_recognition.face_encodings(rgb_crop)
                except Exception:
                    print(f"Face encoding failed: {enc_error}")
                    return None

            if not face_encodings:
                return None

            # Adjust face location to frame coordinates
            top, right, bottom, left = face_locations[0]
            adjusted_location = (top + y1, right + x1, bottom + y1, left + x1)

            return self._match_face(face_encodings[0], adjusted_location)

        except Exception as e:
            print(f"Error recognizing face in region: {e}")
            return None

    def _match_face(
        self,
        face_encoding: np.ndarray,
        face_location: Tuple[int, int, int, int]
    ) -> FaceMatch:
        """Match a face encoding against known faces"""
        if not self._known_encodings:
            return FaceMatch(
                employee_id=None,
                employee_name=None,
                confidence=0.0,
                face_location=face_location,
                is_known=False
            )

        # Calculate distances to all known faces
        distances = face_recognition.face_distance(self._known_encodings, face_encoding)

        # Find the best match
        best_idx = np.argmin(distances)
        best_distance = distances[best_idx]

        if best_distance <= self.tolerance:
            # Match found
            confidence = 1.0 - best_distance
            return FaceMatch(
                employee_id=self._known_employee_ids[best_idx],
                employee_name=self._known_names[best_idx],
                confidence=confidence,
                face_location=face_location,
                is_known=True
            )
        else:
            # No match - unknown person
            return FaceMatch(
                employee_id=None,
                employee_name=None,
                confidence=1.0 - best_distance,
                face_location=face_location,
                is_known=False
            )

    @property
    def known_faces_count(self) -> int:
        return len(self._known_encodings)


class DemoFaceRecognizer:
    """Demo face recognizer for testing without face_recognition library"""

    def __init__(self, tolerance: float = 0.6, model: str = "hog"):
        self.tolerance = tolerance
        self.model = model
        self._known_names = ["Employee A", "Employee B"]
        self._frame_count = 0

    @property
    def is_available(self) -> bool:
        return True

    def load_known_faces(self, faces_data: List[Tuple[int, str, np.ndarray]]):
        self._known_names = [name for _, name, _ in faces_data]

    def add_known_face(self, employee_id: int, name: str, encoding: np.ndarray):
        self._known_names.append(name)

    def encode_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        # Return a random encoding for demo purposes
        return np.random.randn(128)

    def recognize_faces_in_frame(self, frame: np.ndarray) -> List[FaceMatch]:
        # Demo: return mock results
        return []

    def recognize_face_in_region(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> Optional[FaceMatch]:
        self._frame_count += 1

        # Cycle through known names for demo
        if self._known_names:
            name = self._known_names[self._frame_count % len(self._known_names)]
            emp_id = (self._frame_count % len(self._known_names)) + 1
        else:
            name = f"Person-{self._frame_count % 3 + 1}"
            emp_id = None

        x1, y1, x2, y2 = bbox
        # Create a fake face location in the upper portion of the bbox
        face_h = (y2 - y1) // 3
        face_location = (y1, x2 - 20, y1 + face_h, x1 + 20)

        return FaceMatch(
            employee_id=emp_id,
            employee_name=name,
            confidence=0.85,
            face_location=face_location,
            is_known=emp_id is not None
        )

    @property
    def known_faces_count(self) -> int:
        return len(self._known_names)


def create_face_recognizer(
    tolerance: float = 0.6,
    model: str = "hog",
    demo_mode: bool = False
) -> FaceRecognizer:
    """
    Factory function to create face recognizer

    Args:
        tolerance: Face matching tolerance
        model: "hog" (CPU) or "cnn" (GPU)
        demo_mode: If True, return demo recognizer

    Returns:
        FaceRecognizer instance
    """
    if demo_mode or not FACE_RECOGNITION_AVAILABLE:
        return DemoFaceRecognizer(tolerance, model)

    return FaceRecognizer(tolerance, model)
