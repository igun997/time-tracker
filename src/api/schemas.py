from datetime import datetime, date
from typing import Optional, List
from pydantic import BaseModel, Field


# Employee schemas
class EmployeeCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    employee_id: Optional[str] = Field(None, max_length=50)


class EmployeeUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    employee_id: Optional[str] = Field(None, max_length=50)
    is_active: Optional[bool] = None


class EmployeeResponse(BaseModel):
    id: int
    name: str
    employee_id: Optional[str]
    created_at: datetime
    is_active: bool
    face_count: int = 0

    class Config:
        from_attributes = True


class EmployeeListResponse(BaseModel):
    employees: List[EmployeeResponse]
    total: int


# Video source schemas
class VideoSourceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    source_type: str = Field(..., pattern="^(webcam|rtsp|file)$")
    source_url: Optional[str] = None
    device_index: Optional[int] = Field(None, ge=0)


class VideoSourceResponse(BaseModel):
    id: int
    name: str
    source_type: str
    source_url: Optional[str]
    device_index: Optional[int]
    is_active: bool
    created_at: datetime

    class Config:
        from_attributes = True


class VideoSourceListResponse(BaseModel):
    sources: List[VideoSourceResponse]
    total: int


# Detection schemas
class DetectionConfig(BaseModel):
    source_id: int
    demo_mode: bool = False
    frame_skip: int = Field(2, ge=1, le=10)
    confidence_threshold: float = Field(0.5, ge=0.1, le=1.0)


class DetectionStatus(BaseModel):
    is_running: bool
    source_name: Optional[str]
    active_persons: int
    identified_persons: int
    unknown_persons: int
    uptime_seconds: int
    frames_processed: int


class PersonDetectionResponse(BaseModel):
    track_id: int
    employee_id: Optional[int]
    name: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    duration_seconds: int


class DetectionFrame(BaseModel):
    timestamp: float
    frame_number: int
    persons: List[PersonDetectionResponse]
    source_name: str


# Session schemas
class SessionResponse(BaseModel):
    id: int
    employee_id: Optional[int]
    employee_name: Optional[str]
    unknown_person_id: Optional[str]
    source_name: str
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: int
    is_active: bool


class SessionListResponse(BaseModel):
    sessions: List[SessionResponse]
    total: int


# Time tracking schemas
class TimeLogResponse(BaseModel):
    employee_id: int
    employee_name: str
    date: date
    total_seconds: int
    formatted_duration: str
    source_name: str


class DailyReportResponse(BaseModel):
    date: date
    entries: List[TimeLogResponse]
    total_seconds: int
    total_formatted: str
    employee_count: int


class EmployeeTimeHistory(BaseModel):
    employee_id: int
    employee_name: str
    start_date: date
    end_date: date
    total_seconds: int
    total_formatted: str
    days_present: int
    daily_entries: List[TimeLogResponse]


# Manual identification
class ManualIdentifyRequest(BaseModel):
    track_id: int
    employee_id: int


# SSE Event schemas
class SSEEvent(BaseModel):
    event_type: str
    data: dict
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Face upload response
class FaceUploadResponse(BaseModel):
    success: bool
    message: str
    encoding_id: Optional[int] = None


# API status
class APIStatus(BaseModel):
    status: str
    version: str
    detection_running: bool
    active_sources: int
    total_employees: int
    yolo_loaded: bool
    face_recognition_available: bool
