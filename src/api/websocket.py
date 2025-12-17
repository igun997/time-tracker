import asyncio
import base64
import json
import time
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from config import settings
from src.video.capture import VideoCapture, VideoSourceFactory, FrameData
from src.detection.yolo_detector import create_detector, PersonDetection
from src.detection.face_recognizer import create_face_recognizer
from src.detection.tracker import PersonTracker, TrackedPerson
from src.tracking.session_manager import SessionManager, SessionEvent


@dataclass
class DetectionState:
    """State for active detection"""
    is_running: bool = False
    source_name: str = ""
    start_time: float = 0
    frames_processed: int = 0
    active_persons: int = 0
    identified_persons: int = 0
    demo_mode: bool = False


class ConnectionManager:
    """Manages WebSocket connections"""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.event_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, connection_type: str = "stream"):
        await websocket.accept()
        if connection_type == "events":
            self.event_connections.add(websocket)
        else:
            self.active_connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        self.event_connections.discard(websocket)

    async def broadcast_frame(self, frame_data: dict):
        """Broadcast frame to all stream connections"""
        message = json.dumps(frame_data)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)

    async def broadcast_event(self, event: dict):
        """Broadcast event to all event connections"""
        message = json.dumps(event)
        disconnected = []
        for connection in self.event_connections:
            try:
                await connection.send_text(message)
            except:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


class DetectionPipeline:
    """Main detection pipeline for video processing"""

    def __init__(self):
        self.state = DetectionState()
        self.connection_manager = ConnectionManager()
        self._video_capture: Optional[VideoCapture] = None
        self._detector = None
        self._face_recognizer = None
        self._tracker: Optional[PersonTracker] = None
        self._session_manager: Optional[SessionManager] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    async def start(
        self,
        source_type: str,
        source_name: str,
        source_url: Optional[str] = None,
        device_index: int = 0,
        demo_mode: bool = False,
        frame_skip: int = 2,
        confidence_threshold: float = 0.5,
        db_session_callback=None
    ) -> bool:
        """Start the detection pipeline"""
        if self.state.is_running:
            return False

        try:
            # Create video source
            video_source = VideoSourceFactory.create(
                source_type,
                source_name,
                source_url=source_url,
                device_index=device_index
            )

            self._video_capture = VideoCapture(
                video_source,
                frame_skip=frame_skip,
                max_fps=settings.max_fps
            )

            if not self._video_capture.start():
                return False

            # Create detector
            self._detector = create_detector(
                model_path=settings.yolo_model_path,
                confidence_threshold=confidence_threshold,
                device=settings.yolo_device,
                demo_mode=demo_mode
            )

            if not demo_mode:
                self._detector.load_model()

            # Create face recognizer
            self._face_recognizer = create_face_recognizer(
                tolerance=settings.face_recognition_tolerance,
                model=settings.face_recognition_model,
                demo_mode=demo_mode
            )

            # Create tracker
            self._tracker = PersonTracker(
                max_disappeared=30,
                max_distance=100.0
            )

            # Create session manager
            self._session_manager = SessionManager(
                source_name=source_name,
                timeout_seconds=settings.track_timeout_seconds,
                min_session_seconds=settings.min_session_seconds,
                on_session_end=db_session_callback
            )

            # Update state
            self.state = DetectionState(
                is_running=True,
                source_name=source_name,
                start_time=time.time(),
                demo_mode=demo_mode
            )

            # Start processing task
            self._stop_event.clear()
            self._task = asyncio.create_task(self._process_frames())

            return True

        except Exception as e:
            print(f"Error starting detection: {e}")
            self.stop()
            return False

    def stop(self):
        """Stop the detection pipeline"""
        self._stop_event.set()
        self.state.is_running = False

        if self._video_capture:
            self._video_capture.stop()
            self._video_capture = None

        if self._session_manager:
            events = self._session_manager.force_end_all()
            # Handle ended sessions

        self._detector = None
        self._face_recognizer = None
        self._tracker = None
        self._session_manager = None

    async def _process_frames(self):
        """Main frame processing loop"""
        try:
            while not self._stop_event.is_set() and self._video_capture.is_running:
                # Get frame
                frame_data = self._video_capture.source.read_frame()
                if frame_data is None:
                    await asyncio.sleep(0.01)
                    continue

                self.state.frames_processed += 1

                # Run detection
                detections = self._detector.detect(frame_data.frame, use_tracking=True)

                # Update tracker
                tracked_persons = self._tracker.update(
                    detections,
                    frame_data.frame,
                    self._face_recognizer
                )

                # Update sessions
                if self._session_manager:
                    events = self._session_manager.update(tracked_persons)
                    for event in events:
                        await self.connection_manager.broadcast_event(event.to_dict())

                # Update state
                self.state.active_persons = len(tracked_persons)
                self.state.identified_persons = sum(
                    1 for p in tracked_persons if p.is_identified
                )

                # Draw annotations and encode frame
                annotated_frame = self._draw_annotations(
                    frame_data.frame, tracked_persons
                )

                # Encode frame as JPEG
                _, buffer = cv2.imencode(
                    '.jpg', annotated_frame,
                    [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
                )
                frame_base64 = base64.b64encode(buffer).decode('utf-8')

                # Prepare response data
                response = {
                    "type": "frame",
                    "timestamp": frame_data.timestamp,
                    "frame_number": self.state.frames_processed,
                    "frame": frame_base64,
                    "persons": [
                        {
                            "track_id": p.track_id,
                            "employee_id": p.employee_id,
                            "name": p.display_name,
                            "bbox": list(p.bbox),
                            "duration_seconds": int(p.duration_seconds),
                            "is_identified": p.is_identified
                        }
                        for p in tracked_persons
                    ],
                    "stats": {
                        "active_persons": self.state.active_persons,
                        "identified": self.state.identified_persons,
                        "unknown": self.state.active_persons - self.state.identified_persons,
                        "uptime": int(time.time() - self.state.start_time),
                        "fps": self.state.frames_processed / max(1, time.time() - self.state.start_time)
                    }
                }

                # Broadcast to all connections
                await self.connection_manager.broadcast_frame(response)

                # Small delay to prevent CPU overload
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Error in frame processing: {e}")
        finally:
            self.state.is_running = False

    def _draw_annotations(
        self,
        frame: np.ndarray,
        tracked_persons: List[TrackedPerson]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()

        for person in tracked_persons:
            x1, y1, x2, y2 = person.bbox

            # Color based on identification status
            if person.is_identified:
                color = (0, 255, 0)  # Green for identified
            else:
                color = (0, 165, 255)  # Orange for unknown

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            label = person.display_name
            duration = int(person.duration_seconds)
            if duration > 0:
                mins, secs = divmod(duration, 60)
                label += f" ({mins}:{secs:02d})"

            # Draw label background
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                annotated,
                (x1, y1 - label_height - 10),
                (x1 + label_width + 10, y1),
                color, -1
            )

            # Draw label text
            cv2.putText(
                annotated, label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2
            )

        # Draw stats overlay
        stats_text = f"Persons: {len(tracked_persons)} | "
        stats_text += f"Identified: {sum(1 for p in tracked_persons if p.is_identified)}"

        cv2.putText(
            annotated, stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2
        )

        return annotated

    def get_status(self) -> dict:
        """Get current detection status"""
        return {
            "is_running": self.state.is_running,
            "source_name": self.state.source_name,
            "active_persons": self.state.active_persons,
            "identified_persons": self.state.identified_persons,
            "unknown_persons": self.state.active_persons - self.state.identified_persons,
            "uptime_seconds": int(time.time() - self.state.start_time) if self.state.is_running else 0,
            "frames_processed": self.state.frames_processed,
            "demo_mode": self.state.demo_mode
        }

    def manual_identify(self, track_id: int, employee_id: int, employee_name: str):
        """Manually identify a tracked person"""
        if self._tracker:
            self._tracker.manual_identify(track_id, employee_id, employee_name)
        if self._session_manager:
            self._session_manager.manual_identify(track_id, employee_id, employee_name)


# Global pipeline instance
detection_pipeline = DetectionPipeline()


async def websocket_stream_handler(websocket: WebSocket):
    """Handle WebSocket connection for video streaming"""
    await detection_pipeline.connection_manager.connect(websocket, "stream")
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Handle commands if needed
            try:
                message = json.loads(data)
                if message.get("type") == "identify":
                    detection_pipeline.manual_identify(
                        message["track_id"],
                        message["employee_id"],
                        message["employee_name"]
                    )
            except:
                pass
    except WebSocketDisconnect:
        detection_pipeline.connection_manager.disconnect(websocket)


async def websocket_events_handler(websocket: WebSocket):
    """Handle WebSocket connection for session events"""
    await detection_pipeline.connection_manager.connect(websocket, "events")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        detection_pipeline.connection_manager.disconnect(websocket)
