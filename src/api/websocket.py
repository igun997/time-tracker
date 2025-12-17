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

from config import settings, parse_detection_classes, get_class_name, COCO_CLASSES
from src.video.capture import VideoCapture, VideoSourceFactory, FrameData
from src.detection.yolo_detector import create_detector, Detection
from src.detection.face_recognizer import create_face_recognizer
from src.detection.tracker import PersonTracker, TrackedPerson
from src.tracking.session_manager import ObjectSessionManager, SessionManager, SessionEvent


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
    client_mode: bool = False  # True when receiving frames from client webcam
    detection_classes: List[int] = field(default_factory=lambda: [0])  # Classes being detected
    tracking_mode: str = "presence"  # "presence", "counter", or "both"


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
        detection_classes: Optional[str] = None,
        tracking_mode: str = "presence",
        db_session_callback=None,
        db_entry_callback=None,
        db_exit_callback=None
    ) -> bool:
        """Start the detection pipeline"""
        if self.state.is_running:
            return False

        try:
            # Parse detection classes
            classes_str = detection_classes or settings.yolo_classes
            class_ids = parse_detection_classes(classes_str)

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

            # Create detector with configured classes
            self._detector = create_detector(
                model_path=settings.yolo_model_path,
                confidence_threshold=confidence_threshold,
                device=settings.yolo_device,
                demo_mode=demo_mode,
                classes=class_ids
            )

            if not demo_mode:
                self._detector.load_model()

            # Create face recognizer (only useful for person detection)
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

            # Create session manager with tracking mode
            self._session_manager = ObjectSessionManager(
                source_name=source_name,
                mode=tracking_mode,
                timeout_seconds=settings.track_timeout_seconds,
                min_session_seconds=settings.min_session_seconds,
                on_session_end=db_session_callback,
                on_object_entry=db_entry_callback,
                on_object_exit=db_exit_callback
            )

            # Update state
            self.state = DetectionState(
                is_running=True,
                source_name=source_name,
                start_time=time.time(),
                demo_mode=demo_mode,
                detection_classes=class_ids,
                tracking_mode=tracking_mode
            )

            # Start processing task
            self._stop_event.clear()
            self._task = asyncio.create_task(self._process_frames())

            return True

        except Exception as e:
            print(f"Error starting detection: {e}")
            self.stop()
            return False

    async def start_client_mode(
        self,
        source_name: str = "Client Webcam",
        demo_mode: bool = False,
        confidence_threshold: float = 0.5,
        detection_classes: Optional[str] = None,
        tracking_mode: str = "presence",
        db_session_callback=None,
        db_entry_callback=None,
        db_exit_callback=None
    ) -> bool:
        """Start detection pipeline in client mode (receiving frames via WebSocket)"""
        if self.state.is_running:
            return False

        try:
            # Parse detection classes
            classes_str = detection_classes or settings.yolo_classes
            class_ids = parse_detection_classes(classes_str)

            # Create detector with configured classes
            self._detector = create_detector(
                model_path=settings.yolo_model_path,
                confidence_threshold=confidence_threshold,
                device=settings.yolo_device,
                demo_mode=demo_mode,
                classes=class_ids
            )

            if not demo_mode:
                self._detector.load_model()

            # Create face recognizer (only useful for person detection)
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

            # Create session manager with tracking mode
            self._session_manager = ObjectSessionManager(
                source_name=source_name,
                mode=tracking_mode,
                timeout_seconds=settings.track_timeout_seconds,
                min_session_seconds=settings.min_session_seconds,
                on_session_end=db_session_callback,
                on_object_entry=db_entry_callback,
                on_object_exit=db_exit_callback
            )

            # Update state
            self.state = DetectionState(
                is_running=True,
                source_name=source_name,
                start_time=time.time(),
                demo_mode=demo_mode,
                client_mode=True,
                detection_classes=class_ids,
                tracking_mode=tracking_mode
            )

            return True

        except Exception as e:
            print(f"Error starting client mode detection: {e}")
            self.stop()
            return False

    async def process_client_frame(self, frame_base64: str) -> Optional[dict]:
        """Process a frame received from client webcam"""
        if not self.state.is_running or not self.state.client_mode:
            return None

        try:
            # Decode base64 frame
            frame_data = base64.b64decode(frame_base64)
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                return None

            self.state.frames_processed += 1

            # Run detection
            detections = self._detector.detect(frame, use_tracking=True)

            # Update tracker
            tracked_persons = self._tracker.update(
                detections,
                frame,
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
            annotated_frame = self._draw_annotations(frame, tracked_persons)

            # Encode frame as JPEG
            _, buffer = cv2.imencode(
                '.jpg', annotated_frame,
                [cv2.IMWRITE_JPEG_QUALITY, settings.jpeg_quality]
            )
            result_base64 = base64.b64encode(buffer).decode('utf-8')

            # Count by class
            class_counts = {}
            for p in tracked_persons:
                class_counts[p.class_name] = class_counts.get(p.class_name, 0) + 1

            # Prepare response data (convert numpy types to native Python types for JSON)
            response = {
                "type": "frame",
                "timestamp": time.time(),
                "frame_number": self.state.frames_processed,
                "frame": result_base64,
                "detections": [
                    {
                        "track_id": int(p.track_id) if p.track_id is not None else None,
                        "class_id": int(p.class_id),
                        "class_name": str(p.class_name),
                        "employee_id": int(p.employee_id) if p.employee_id is not None else None,
                        "name": str(p.display_name),
                        "bbox": [int(x) for x in p.bbox],
                        "duration_seconds": int(p.duration_seconds),
                        "is_identified": bool(p.is_identified),
                        "is_person": bool(p.is_person),
                        "is_vehicle": bool(p.is_vehicle)
                    }
                    for p in tracked_persons
                ],
                "stats": {
                    "total_objects": len(tracked_persons),
                    "active_persons": sum(1 for p in tracked_persons if p.is_person),
                    "active_vehicles": sum(1 for p in tracked_persons if p.is_vehicle),
                    "identified": self.state.identified_persons,
                    "class_counts": class_counts,
                    "uptime": int(time.time() - self.state.start_time),
                    "fps": self.state.frames_processed / max(1, time.time() - self.state.start_time)
                }
            }

            return response

        except Exception as e:
            print(f"Error processing client frame: {e}")
            return None

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

                # Count by class
                class_counts = {}
                for p in tracked_persons:
                    class_counts[p.class_name] = class_counts.get(p.class_name, 0) + 1

                # Prepare response data (convert numpy types to native Python types for JSON)
                response = {
                    "type": "frame",
                    "timestamp": frame_data.timestamp,
                    "frame_number": self.state.frames_processed,
                    "frame": frame_base64,
                    "detections": [
                        {
                            "track_id": int(p.track_id) if p.track_id is not None else None,
                            "class_id": int(p.class_id),
                            "class_name": str(p.class_name),
                            "employee_id": int(p.employee_id) if p.employee_id is not None else None,
                            "name": str(p.display_name),
                            "bbox": [int(x) for x in p.bbox],
                            "duration_seconds": int(p.duration_seconds),
                            "is_identified": bool(p.is_identified),
                            "is_person": bool(p.is_person),
                            "is_vehicle": bool(p.is_vehicle)
                        }
                        for p in tracked_persons
                    ],
                    "stats": {
                        "total_objects": len(tracked_persons),
                        "active_persons": sum(1 for p in tracked_persons if p.is_person),
                        "active_vehicles": sum(1 for p in tracked_persons if p.is_vehicle),
                        "identified": self.state.identified_persons,
                        "class_counts": class_counts,
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

    def _get_class_color(self, obj) -> tuple:
        """Get color based on object class and identification status"""
        # Color palette for different object types (BGR format)
        CLASS_COLORS = {
            0: (0, 255, 0),      # person - green
            1: (255, 165, 0),    # bicycle - orange
            2: (255, 0, 0),      # car - blue
            3: (0, 255, 255),    # motorcycle - yellow
            5: (255, 0, 255),    # bus - magenta
            7: (0, 128, 255),    # truck - orange-red
            9: (0, 255, 128),    # traffic light - cyan-green
        }

        if obj.is_person:
            if obj.is_identified:
                return (0, 255, 0)    # Green for identified person
            else:
                return (0, 165, 255)  # Orange for unknown person
        else:
            return CLASS_COLORS.get(obj.class_id, (128, 128, 128))  # Default gray

    def _draw_annotations(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedPerson]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        annotated = frame.copy()

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj.bbox

            # Color based on class and identification status
            color = self._get_class_color(obj)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            label = obj.display_name
            duration = int(obj.duration_seconds)
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

        # Count by class
        class_counts = {}
        for obj in tracked_objects:
            class_counts[obj.class_name] = class_counts.get(obj.class_name, 0) + 1

        # Draw stats overlay
        stats_parts = []
        for class_name, count in sorted(class_counts.items()):
            stats_parts.append(f"{class_name}: {count}")
        stats_text = " | ".join(stats_parts) if stats_parts else "No detections"

        cv2.putText(
            annotated, stats_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2
        )

        return annotated

    def get_status(self) -> dict:
        """Get current detection status"""
        class_names = [get_class_name(c) for c in self.state.detection_classes]
        return {
            "is_running": self.state.is_running,
            "source_name": self.state.source_name,
            "active_persons": self.state.active_persons,
            "identified_persons": self.state.identified_persons,
            "unknown_persons": self.state.active_persons - self.state.identified_persons,
            "uptime_seconds": int(time.time() - self.state.start_time) if self.state.is_running else 0,
            "frames_processed": self.state.frames_processed,
            "demo_mode": self.state.demo_mode,
            "client_mode": self.state.client_mode,
            "detection_classes": class_names
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


async def websocket_client_cam_handler(
    websocket: WebSocket,
    demo_mode: bool = False,
    detection_classes: Optional[str] = None,
    tracking_mode: str = "presence"
):
    """Handle WebSocket connection for client webcam streaming

    Client sends frames, server processes and returns annotated frames.
    Protocol:
    - Client sends: {"type": "frame", "frame": "<base64 jpeg>"}
    - Server responds: {"type": "frame", "frame": "<annotated base64>", "detections": [...], "stats": {...}}
    - Client can send: {"type": "stop"} to stop detection
    """
    await websocket.accept()

    # Start detection in client mode
    success = await detection_pipeline.start_client_mode(
        source_name="Client Webcam",
        demo_mode=demo_mode,
        confidence_threshold=settings.yolo_confidence,
        detection_classes=detection_classes,
        tracking_mode=tracking_mode
    )

    if not success:
        await websocket.send_json({
            "type": "error",
            "message": "Failed to start detection. Is another detection already running?"
        })
        await websocket.close()
        return

    # Get detected classes for response
    class_names = [get_class_name(c) for c in detection_pipeline.state.detection_classes]

    await websocket.send_json({
        "type": "started",
        "message": "Detection started in client mode",
        "detection_classes": class_names,
        "tracking_mode": tracking_mode
    })

    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "frame":
                # Process the frame
                frame_base64 = message.get("frame", "")
                if frame_base64:
                    result = await detection_pipeline.process_client_frame(frame_base64)
                    if result:
                        await websocket.send_json(result)

            elif message.get("type") == "stop":
                detection_pipeline.stop()
                await websocket.send_json({
                    "type": "stopped",
                    "message": "Detection stopped"
                })
                break

            elif message.get("type") == "identify":
                detection_pipeline.manual_identify(
                    message["track_id"],
                    message["employee_id"],
                    message["employee_name"]
                )

    except WebSocketDisconnect:
        detection_pipeline.stop()
    except Exception as e:
        print(f"Client webcam error: {e}")
        detection_pipeline.stop()
