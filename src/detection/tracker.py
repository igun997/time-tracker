from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist
import time

from .yolo_detector import Detection, PersonDetection  # PersonDetection is alias for backward compat


@dataclass
class TrackedObject:
    """Represents a tracked object with identity and history"""
    track_id: int
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    class_id: int = 0
    class_name: str = "person"
    employee_id: Optional[int] = None
    employee_name: Optional[str] = None
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    disappeared_frames: int = 0
    is_identified: bool = False

    @property
    def display_name(self) -> str:
        if self.employee_name:
            return self.employee_name
        if self.class_id == 0:  # person
            return f"Unknown-{self.track_id}"
        return f"{self.class_name}-{self.track_id}"

    @property
    def duration_seconds(self) -> float:
        return self.last_seen - self.first_seen

    @property
    def is_person(self) -> bool:
        return self.class_id == 0

    @property
    def is_vehicle(self) -> bool:
        return self.class_id in [1, 2, 3, 5, 6, 7, 8]


# Alias for backwards compatibility
TrackedPerson = TrackedObject


class CentroidTracker:
    """
    Simple centroid-based object tracker.
    Associates detections across frames based on centroid distance.
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 100.0
    ):
        """
        Args:
            max_disappeared: Frames before removing a track
            max_distance: Max centroid distance for matching
        """
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

        self._next_id = 0
        self._objects: OrderedDict[int, TrackedPerson] = OrderedDict()
        self._disappeared: Dict[int, int] = {}

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Update tracker with new detections

        Args:
            detections: List of Detection from YOLO

        Returns:
            List of currently tracked objects
        """
        current_time = time.time()

        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for object_id in list(self._disappeared.keys()):
                self._disappeared[object_id] += 1

                if self._disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            return list(self._objects.values())

        # Get input centroids and metadata
        input_centroids = np.array([d.center for d in detections])
        input_bboxes = [d.bbox for d in detections]
        input_class_ids = [d.class_id for d in detections]
        input_class_names = [d.class_name for d in detections]

        # If we have no existing objects, register all detections
        if len(self._objects) == 0:
            for i in range(len(detections)):
                self._register(
                    tuple(input_centroids[i]),
                    input_bboxes[i],
                    input_class_ids[i],
                    input_class_names[i],
                    current_time
                )
        else:
            # Match existing objects with new detections
            object_ids = list(self._objects.keys())
            object_centroids = np.array([
                self._objects[oid].centroid for oid in object_ids
            ])

            # Compute distance matrix
            D = dist.cdist(object_centroids, input_centroids)

            # Find matches using Hungarian algorithm (greedy approximation)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                self._objects[object_id].centroid = tuple(input_centroids[col])
                self._objects[object_id].bbox = input_bboxes[col]
                self._objects[object_id].last_seen = current_time
                self._disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing objects
            unused_rows = set(range(len(object_ids))) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self._disappeared[object_id] += 1

                if self._disappeared[object_id] > self.max_disappeared:
                    self._deregister(object_id)

            # Handle unmatched new detections
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self._register(
                    tuple(input_centroids[col]),
                    input_bboxes[col],
                    input_class_ids[col],
                    input_class_names[col],
                    current_time
                )

        return list(self._objects.values())

    def _register(
        self,
        centroid: Tuple[int, int],
        bbox: Tuple[int, int, int, int],
        class_id: int,
        class_name: str,
        timestamp: float
    ):
        """Register a new tracked object"""
        self._objects[self._next_id] = TrackedObject(
            track_id=self._next_id,
            centroid=centroid,
            bbox=bbox,
            class_id=class_id,
            class_name=class_name,
            first_seen=timestamp,
            last_seen=timestamp
        )
        self._disappeared[self._next_id] = 0
        self._next_id += 1

    def _deregister(self, object_id: int):
        """Remove a tracked object"""
        del self._objects[object_id]
        del self._disappeared[object_id]

    def update_identity(
        self,
        track_id: int,
        employee_id: Optional[int],
        employee_name: Optional[str]
    ):
        """Update the identity of a tracked person"""
        if track_id in self._objects:
            self._objects[track_id].employee_id = employee_id
            self._objects[track_id].employee_name = employee_name
            self._objects[track_id].is_identified = employee_id is not None

    def get_tracked_person(self, track_id: int) -> Optional[TrackedPerson]:
        """Get a specific tracked person"""
        return self._objects.get(track_id)

    def get_all_tracked(self) -> List[TrackedPerson]:
        """Get all currently tracked persons"""
        return list(self._objects.values())

    def get_disappeared_persons(self) -> List[TrackedPerson]:
        """Get persons that just disappeared (for session ending)"""
        # This would need to be called before they're removed
        disappeared = []
        for object_id, frames in self._disappeared.items():
            if frames == self.max_disappeared:
                if object_id in self._objects:
                    disappeared.append(self._objects[object_id])
        return disappeared

    def reset(self):
        """Reset the tracker"""
        self._next_id = 0
        self._objects.clear()
        self._disappeared.clear()

    @property
    def active_count(self) -> int:
        return len(self._objects)


class PersonTracker:
    """
    High-level person tracker that combines YOLO detection with centroid tracking
    and face recognition for identity association.
    """

    def __init__(
        self,
        max_disappeared: int = 30,
        max_distance: float = 100.0,
        identity_update_interval: int = 30  # frames
    ):
        self.tracker = CentroidTracker(max_disappeared, max_distance)
        self.identity_update_interval = identity_update_interval
        self._frame_count = 0
        self._pending_identifications: Dict[int, int] = {}  # track_id -> frame_count

    def update(
        self,
        detections: List[PersonDetection],
        frame: Optional[np.ndarray] = None,
        face_recognizer=None
    ) -> List[TrackedPerson]:
        """
        Update tracker with detections and optionally perform face recognition

        Args:
            detections: Person detections from YOLO
            frame: Original frame for face recognition
            face_recognizer: FaceRecognizer instance

        Returns:
            List of tracked persons with identities
        """
        self._frame_count += 1

        # Update centroid tracker
        tracked = self.tracker.update(detections)

        # Perform face recognition for unidentified persons
        if frame is not None and face_recognizer is not None:
            for person in tracked:
                # Skip if already identified or recently checked
                if person.is_identified:
                    continue

                if person.track_id in self._pending_identifications:
                    last_check = self._pending_identifications[person.track_id]
                    if self._frame_count - last_check < self.identity_update_interval:
                        continue

                # Attempt face recognition
                match = face_recognizer.recognize_face_in_region(frame, person.bbox)

                if match and match.is_known:
                    self.tracker.update_identity(
                        person.track_id,
                        match.employee_id,
                        match.employee_name
                    )

                self._pending_identifications[person.track_id] = self._frame_count

        return tracked

    def manual_identify(
        self,
        track_id: int,
        employee_id: int,
        employee_name: str
    ):
        """Manually set identity for a tracked person"""
        self.tracker.update_identity(track_id, employee_id, employee_name)

    def get_session_updates(self) -> Tuple[List[TrackedPerson], List[TrackedPerson]]:
        """
        Get persons who just appeared and disappeared

        Returns:
            (new_persons, departed_persons)
        """
        departed = self.tracker.get_disappeared_persons()
        # New persons would need frame-by-frame comparison
        # For simplicity, handle this in session manager
        return [], departed

    def reset(self):
        """Reset the tracker"""
        self.tracker.reset()
        self._frame_count = 0
        self._pending_identifications.clear()

    @property
    def active_persons(self) -> List[TrackedPerson]:
        return self.tracker.get_all_tracked()

    @property
    def active_count(self) -> int:
        return self.tracker.active_count
