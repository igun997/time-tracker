from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, date
import asyncio
import time

from src.detection.tracker import TrackedPerson


@dataclass
class ActiveSession:
    """Represents an active presence session"""
    session_id: Optional[int]  # Database ID, None until persisted
    track_id: int
    employee_id: Optional[int]
    employee_name: Optional[str]
    source_name: str
    start_time: datetime
    last_seen: datetime
    is_persisted: bool = False

    @property
    def duration_seconds(self) -> int:
        return int((self.last_seen - self.start_time).total_seconds())

    @property
    def display_name(self) -> str:
        if self.employee_name:
            return self.employee_name
        return f"Unknown-{self.track_id}"


@dataclass
class SessionEvent:
    """Event emitted when session state changes"""
    event_type: str  # "started", "updated", "ended"
    session: ActiveSession
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "track_id": self.session.track_id,
            "employee_id": self.session.employee_id,
            "employee_name": self.session.display_name,
            "source_name": self.session.source_name,
            "start_time": self.session.start_time.isoformat(),
            "duration_seconds": self.session.duration_seconds,
            "timestamp": self.timestamp.isoformat()
        }


class SessionManager:
    """
    Manages presence sessions based on tracked persons.
    Handles session lifecycle: start, update, end.
    """

    def __init__(
        self,
        source_name: str,
        timeout_seconds: int = 30,
        min_session_seconds: int = 5,
        on_session_start: Optional[Callable[[ActiveSession], Any]] = None,
        on_session_end: Optional[Callable[[ActiveSession], Any]] = None
    ):
        """
        Args:
            source_name: Name of the video source/camera
            timeout_seconds: Seconds without detection before ending session
            min_session_seconds: Minimum session duration to record
            on_session_start: Callback when session starts
            on_session_end: Callback when session ends
        """
        self.source_name = source_name
        self.timeout_seconds = timeout_seconds
        self.min_session_seconds = min_session_seconds
        self.on_session_start = on_session_start
        self.on_session_end = on_session_end

        self._active_sessions: Dict[int, ActiveSession] = {}  # track_id -> session
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._last_update = time.time()

    def update(self, tracked_persons: List[TrackedPerson]) -> List[SessionEvent]:
        """
        Update sessions based on current tracked persons

        Args:
            tracked_persons: List of currently tracked persons

        Returns:
            List of session events that occurred
        """
        current_time = datetime.utcnow()
        events = []
        current_track_ids = {p.track_id for p in tracked_persons}

        # Update existing sessions and start new ones
        for person in tracked_persons:
            if person.track_id in self._active_sessions:
                # Update existing session
                session = self._active_sessions[person.track_id]
                session.last_seen = current_time

                # Update identity if it changed
                if person.employee_id and session.employee_id != person.employee_id:
                    session.employee_id = person.employee_id
                    session.employee_name = person.employee_name

            else:
                # Start new session
                session = ActiveSession(
                    session_id=None,
                    track_id=person.track_id,
                    employee_id=person.employee_id,
                    employee_name=person.employee_name,
                    source_name=self.source_name,
                    start_time=current_time,
                    last_seen=current_time
                )
                self._active_sessions[person.track_id] = session

                event = SessionEvent("started", session)
                events.append(event)

                if self.on_session_start:
                    self.on_session_start(session)

        # Check for ended sessions (persons no longer tracked)
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

        self._last_update = time.time()
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

        return events

    def get_active_sessions(self) -> List[ActiveSession]:
        """Get all currently active sessions"""
        return list(self._active_sessions.values())

    def get_session_by_track_id(self, track_id: int) -> Optional[ActiveSession]:
        """Get session for a specific track ID"""
        return self._active_sessions.get(track_id)

    def manual_identify(self, track_id: int, employee_id: int, employee_name: str):
        """Manually identify a person in an active session"""
        if track_id in self._active_sessions:
            session = self._active_sessions[track_id]
            session.employee_id = employee_id
            session.employee_name = employee_name

    def get_summary(self) -> dict:
        """Get summary of current session state"""
        sessions = self.get_active_sessions()
        identified = [s for s in sessions if s.employee_id is not None]
        unknown = [s for s in sessions if s.employee_id is None]

        return {
            "source_name": self.source_name,
            "total_active": len(sessions),
            "identified_count": len(identified),
            "unknown_count": len(unknown),
            "sessions": [
                {
                    "track_id": s.track_id,
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


class MultiSourceSessionManager:
    """Manages sessions across multiple video sources"""

    def __init__(
        self,
        timeout_seconds: int = 30,
        min_session_seconds: int = 5
    ):
        self.timeout_seconds = timeout_seconds
        self.min_session_seconds = min_session_seconds
        self._managers: Dict[str, SessionManager] = {}
        self._event_callbacks: List[Callable[[SessionEvent], Any]] = []

    def get_or_create_manager(self, source_name: str) -> SessionManager:
        """Get or create a session manager for a source"""
        if source_name not in self._managers:
            self._managers[source_name] = SessionManager(
                source_name=source_name,
                timeout_seconds=self.timeout_seconds,
                min_session_seconds=self.min_session_seconds,
                on_session_start=self._on_session_event,
                on_session_end=self._on_session_event
            )
        return self._managers[source_name]

    def _on_session_event(self, session: ActiveSession):
        """Internal callback for session events"""
        # Notify all registered callbacks
        for callback in self._event_callbacks:
            try:
                callback(session)
            except Exception as e:
                print(f"Error in session callback: {e}")

    def register_callback(self, callback: Callable[[ActiveSession], Any]):
        """Register a callback for session events"""
        self._event_callbacks.append(callback)

    def get_all_active_sessions(self) -> Dict[str, List[ActiveSession]]:
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

        # Aggregate by employee
        employee_times: Dict[int, dict] = {}
        unknown_count = 0

        for session in all_sessions:
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
            "identified_count": len(employee_times),
            "unknown_count": unknown_count,
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
