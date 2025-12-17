from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, date
import asyncio
import time

from src.detection.tracker import TrackedPerson


@dataclass
class ObjectActiveSession:
    """Represents an active presence session for any tracked object"""
    session_id: Optional[int]  # Database ID, None until persisted
    track_id: int
    class_id: int
    class_name: str
    employee_id: Optional[int]
    employee_name: Optional[str]
    source_name: str
    start_time: datetime
    last_seen: datetime
    label: Optional[str] = None
    is_persisted: bool = False

    @property
    def duration_seconds(self) -> int:
        return int((self.last_seen - self.start_time).total_seconds())

    @property
    def display_name(self) -> str:
        if self.employee_name:
            return self.employee_name
        if self.label:
            return self.label
        return f"{self.class_name.title()}-{self.track_id}"

    @property
    def is_person(self) -> bool:
        return self.class_id == 0

    @property
    def is_identified(self) -> bool:
        return self.employee_id is not None or self.label is not None


# Alias for backward compatibility
ActiveSession = ObjectActiveSession


@dataclass
class SessionEvent:
    """Event emitted when session state changes"""
    event_type: str  # "started", "updated", "ended", "entry", "exit"
    session: ObjectActiveSession
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "track_id": self.session.track_id,
            "class_id": self.session.class_id,
            "class_name": self.session.class_name,
            "employee_id": self.session.employee_id,
            "employee_name": self.session.display_name,
            "source_name": self.session.source_name,
            "start_time": self.session.start_time.isoformat(),
            "duration_seconds": self.session.duration_seconds,
            "timestamp": self.timestamp.isoformat()
        }


class ObjectSessionManager:
    """
    Enhanced session manager for tracking any detected objects.
    Supports two modes:
    - presence: Track how long objects stay in frame
    - counter: Count objects entering/exiting
    - both: Track both duration and counts
    """

    def __init__(
        self,
        source_name: str,
        mode: str = "presence",  # "presence", "counter", or "both"
        timeout_seconds: int = 30,
        min_session_seconds: int = 3,
        counter_hourly: bool = True,
        on_session_start: Optional[Callable[[ObjectActiveSession], Any]] = None,
        on_session_end: Optional[Callable[[ObjectActiveSession], Any]] = None,
        on_object_entry: Optional[Callable[[ObjectActiveSession], Any]] = None,
        on_object_exit: Optional[Callable[[ObjectActiveSession], Any]] = None
    ):
        """
        Args:
            source_name: Name of the video source/camera
            mode: Tracking mode - "presence", "counter", or "both"
            timeout_seconds: Seconds without detection before ending session
            min_session_seconds: Minimum session duration to record
            counter_hourly: Track hourly counts (vs daily only)
            on_session_start: Callback when session starts (presence mode)
            on_session_end: Callback when session ends (presence mode)
            on_object_entry: Callback when object first detected (counter mode)
            on_object_exit: Callback when object leaves (counter mode)
        """
        self.source_name = source_name
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.min_session_seconds = min_session_seconds
        self.counter_hourly = counter_hourly
        self.on_session_start = on_session_start
        self.on_session_end = on_session_end
        self.on_object_entry = on_object_entry
        self.on_object_exit = on_object_exit

        self._active_sessions: Dict[int, ObjectActiveSession] = {}  # track_id -> session
        self._counted_entries: Set[int] = set()  # track_ids that have been counted as entries
        self._last_update = time.time()

    def update(self, tracked_objects: List[TrackedPerson]) -> List[SessionEvent]:
        """
        Update sessions based on current tracked objects

        Args:
            tracked_objects: List of currently tracked objects

        Returns:
            List of session events that occurred
        """
        events = []

        if self.mode == "presence":
            events = self._update_presence_mode(tracked_objects)
        elif self.mode == "counter":
            events = self._update_counter_mode(tracked_objects)
        else:  # both
            events = self._update_presence_mode(tracked_objects)
            events.extend(self._update_counter_mode(tracked_objects))

        self._last_update = time.time()
        return events

    def _update_presence_mode(self, tracked_objects: List[TrackedPerson]) -> List[SessionEvent]:
        """Update sessions in presence mode - track duration"""
        current_time = datetime.utcnow()
        events = []
        current_track_ids = {obj.track_id for obj in tracked_objects}

        # Update existing sessions and start new ones
        for obj in tracked_objects:
            if obj.track_id in self._active_sessions:
                # Update existing session
                session = self._active_sessions[obj.track_id]
                session.last_seen = current_time

                # Update identity if it changed
                if obj.employee_id and session.employee_id != obj.employee_id:
                    session.employee_id = obj.employee_id
                    session.employee_name = obj.employee_name

            else:
                # Start new session
                session = ObjectActiveSession(
                    session_id=None,
                    track_id=obj.track_id,
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                    employee_id=obj.employee_id,
                    employee_name=obj.employee_name,
                    source_name=self.source_name,
                    start_time=current_time,
                    last_seen=current_time
                )
                self._active_sessions[obj.track_id] = session

                event = SessionEvent("started", session)
                events.append(event)

                if self.on_session_start:
                    self.on_session_start(session)

        # Check for ended sessions (objects no longer tracked)
        ended_track_ids = []
        for track_id, session in self._active_sessions.items():
            if track_id not in current_track_ids:
                # Check if timeout exceeded
                time_since_seen = (current_time - session.last_seen).total_seconds()
                if time_since_seen >= self.timeout_seconds:
                    ended_track_ids.append(track_id)

        # End sessions
        for track_id in ended_track_ids:
            session = self._active_sessions.pop(track_id)

            # Only emit event if session was long enough
            if session.duration_seconds >= self.min_session_seconds:
                event = SessionEvent("ended", session)
                events.append(event)

                if self.on_session_end:
                    self.on_session_end(session)

        return events

    def _update_counter_mode(self, tracked_objects: List[TrackedPerson]) -> List[SessionEvent]:
        """Update counts in counter mode - count entries/exits"""
        current_time = datetime.utcnow()
        events = []
        current_track_ids = {obj.track_id for obj in tracked_objects}

        # Count new entries
        for obj in tracked_objects:
            if obj.track_id not in self._counted_entries:
                self._counted_entries.add(obj.track_id)

                # Create a session object for the event
                session = ObjectActiveSession(
                    session_id=None,
                    track_id=obj.track_id,
                    class_id=obj.class_id,
                    class_name=obj.class_name,
                    employee_id=obj.employee_id,
                    employee_name=obj.employee_name,
                    source_name=self.source_name,
                    start_time=current_time,
                    last_seen=current_time
                )

                event = SessionEvent("entry", session)
                events.append(event)

                if self.on_object_entry:
                    self.on_object_entry(session)

        # Count exits (track_ids that were in counted_entries but no longer tracked)
        # We need to track last seen times to determine exits
        if not hasattr(self, '_counter_last_seen'):
            self._counter_last_seen: Dict[int, tuple] = {}  # track_id -> (last_seen, class_id, class_name)

        # Update last seen times
        for obj in tracked_objects:
            self._counter_last_seen[obj.track_id] = (current_time, obj.class_id, obj.class_name)

        # Check for exits
        exited_track_ids = []
        for track_id in list(self._counter_last_seen.keys()):
            if track_id not in current_track_ids:
                last_seen, class_id, class_name = self._counter_last_seen[track_id]
                time_since_seen = (current_time - last_seen).total_seconds()
                if time_since_seen >= self.timeout_seconds:
                    exited_track_ids.append(track_id)

        for track_id in exited_track_ids:
            last_seen, class_id, class_name = self._counter_last_seen.pop(track_id)
            self._counted_entries.discard(track_id)

            session = ObjectActiveSession(
                session_id=None,
                track_id=track_id,
                class_id=class_id,
                class_name=class_name,
                employee_id=None,
                employee_name=None,
                source_name=self.source_name,
                start_time=last_seen,
                last_seen=current_time
            )

            event = SessionEvent("exit", session)
            events.append(event)

            if self.on_object_exit:
                self.on_object_exit(session)

        return events

    def force_end_all(self) -> List[SessionEvent]:
        """Force end all active sessions (e.g., when stopping detection)"""
        events = []
        for track_id in list(self._active_sessions.keys()):
            session = self._active_sessions.pop(track_id)
            session.last_seen = datetime.utcnow()

            if session.duration_seconds >= self.min_session_seconds:
                event = SessionEvent("ended", session)
                events.append(event)

                if self.on_session_end:
                    self.on_session_end(session)

        # Clear counter state too
        self._counted_entries.clear()
        if hasattr(self, '_counter_last_seen'):
            self._counter_last_seen.clear()

        return events

    def get_active_sessions(self) -> List[ObjectActiveSession]:
        """Get all currently active sessions"""
        return list(self._active_sessions.values())

    def get_session_by_track_id(self, track_id: int) -> Optional[ObjectActiveSession]:
        """Get session for a specific track ID"""
        return self._active_sessions.get(track_id)

    def manual_identify(self, track_id: int, employee_id: int, employee_name: str):
        """Manually identify a person in an active session"""
        if track_id in self._active_sessions:
            session = self._active_sessions[track_id]
            session.employee_id = employee_id
            session.employee_name = employee_name

    def set_label(self, track_id: int, label: str):
        """Set a custom label for a tracked object"""
        if track_id in self._active_sessions:
            session = self._active_sessions[track_id]
            session.label = label

    def get_summary(self) -> dict:
        """Get summary of current session state"""
        sessions = self.get_active_sessions()

        # Group by class
        by_class: Dict[str, list] = {}
        for s in sessions:
            if s.class_name not in by_class:
                by_class[s.class_name] = []
            by_class[s.class_name].append(s)

        return {
            "source_name": self.source_name,
            "mode": self.mode,
            "total_active": len(sessions),
            "by_class": {
                class_name: len(class_sessions)
                for class_name, class_sessions in by_class.items()
            },
            "sessions": [
                {
                    "track_id": s.track_id,
                    "class_id": s.class_id,
                    "class_name": s.class_name,
                    "name": s.display_name,
                    "employee_id": s.employee_id,
                    "duration_seconds": s.duration_seconds,
                    "start_time": s.start_time.isoformat()
                }
                for s in sessions
            ]
        }

    @property
    def active_count(self) -> int:
        return len(self._active_sessions)


# Keep old class name for backward compatibility
SessionManager = ObjectSessionManager


class MultiSourceSessionManager:
    """Manages sessions across multiple video sources"""

    def __init__(
        self,
        mode: str = "presence",
        timeout_seconds: int = 30,
        min_session_seconds: int = 3
    ):
        self.mode = mode
        self.timeout_seconds = timeout_seconds
        self.min_session_seconds = min_session_seconds
        self._managers: Dict[str, ObjectSessionManager] = {}
        self._event_callbacks: List[Callable[[SessionEvent], Any]] = []

    def get_or_create_manager(self, source_name: str) -> ObjectSessionManager:
        """Get or create a session manager for a source"""
        if source_name not in self._managers:
            self._managers[source_name] = ObjectSessionManager(
                source_name=source_name,
                mode=self.mode,
                timeout_seconds=self.timeout_seconds,
                min_session_seconds=self.min_session_seconds,
                on_session_start=self._on_session_event,
                on_session_end=self._on_session_event
            )
        return self._managers[source_name]

    def _on_session_event(self, session: ObjectActiveSession):
        """Internal callback for session events"""
        for callback in self._event_callbacks:
            try:
                callback(session)
            except Exception as e:
                print(f"Error in session callback: {e}")

    def register_callback(self, callback: Callable[[ObjectActiveSession], Any]):
        """Register a callback for session events"""
        self._event_callbacks.append(callback)

    def get_all_active_sessions(self) -> Dict[str, List[ObjectActiveSession]]:
        """Get all active sessions grouped by source"""
        return {
            source: manager.get_active_sessions()
            for source, manager in self._managers.items()
        }

    def get_global_summary(self) -> dict:
        """Get summary across all sources"""
        all_sessions = []
        for manager in self._managers.values():
            all_sessions.extend(manager.get_active_sessions())

        # Group by class
        by_class: Dict[str, int] = {}
        for session in all_sessions:
            by_class[session.class_name] = by_class.get(session.class_name, 0) + 1

        # Aggregate by employee (for persons)
        employee_times: Dict[int, dict] = {}
        unknown_count = 0

        for session in all_sessions:
            if session.is_person:
                if session.employee_id:
                    if session.employee_id not in employee_times:
                        employee_times[session.employee_id] = {
                            "name": session.employee_name,
                            "total_seconds": 0,
                            "sources": set()
                        }
                    employee_times[session.employee_id]["total_seconds"] += session.duration_seconds
                    employee_times[session.employee_id]["sources"].add(session.source_name)
                else:
                    unknown_count += 1

        return {
            "total_active": len(all_sessions),
            "by_class": by_class,
            "identified_persons": len(employee_times),
            "unknown_persons": unknown_count,
            "sources_count": len(self._managers),
            "employees": [
                {
                    "employee_id": emp_id,
                    "name": data["name"],
                    "total_seconds": data["total_seconds"],
                    "sources": list(data["sources"])
                }
                for emp_id, data in employee_times.items()
            ]
        }

    def stop_all(self) -> List[SessionEvent]:
        """Stop all managers and end all sessions"""
        all_events = []
        for manager in self._managers.values():
            events = manager.force_end_all()
            all_events.extend(events)
        return all_events
