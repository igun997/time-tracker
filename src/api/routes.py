from datetime import date, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
from sqlalchemy.ext.asyncio import AsyncSession
import aiofiles
import os
import uuid

from config import settings
from src.database import crud
from src.database.models import get_async_session_maker
from .schemas import (
    EmployeeCreate, EmployeeUpdate, EmployeeResponse, EmployeeListResponse,
    VideoSourceCreate, VideoSourceResponse, VideoSourceListResponse,
    DailyReportResponse, EmployeeTimeHistory, TimeLogResponse,
    FaceUploadResponse, ManualIdentifyRequest
)

router = APIRouter()

# Dependency to get database session
async def get_db():
    from main import async_session_maker
    async with async_session_maker() as session:
        yield session


# ============== Employee Endpoints ==============

@router.post("/employees", response_model=EmployeeResponse)
async def create_employee(employee: EmployeeCreate, db: AsyncSession = Depends(get_db)):
    """Register a new employee"""
    # Check if employee_id already exists
    if employee.employee_id:
        existing = await crud.get_employee_by_employee_id(db, employee.employee_id)
        if existing:
            raise HTTPException(400, "Employee ID already exists")

    db_employee = await crud.create_employee(db, employee.name, employee.employee_id)
    return EmployeeResponse(
        id=db_employee.id,
        name=db_employee.name,
        employee_id=db_employee.employee_id,
        created_at=db_employee.created_at,
        is_active=db_employee.is_active,
        face_count=len(db_employee.face_encodings) if db_employee.face_encodings else 0
    )


@router.get("/employees", response_model=EmployeeListResponse)
async def list_employees(
    active_only: bool = Query(True),
    db: AsyncSession = Depends(get_db)
):
    """List all employees"""
    employees = await crud.get_all_employees(db, active_only)
    return EmployeeListResponse(
        employees=[
            EmployeeResponse(
                id=e.id,
                name=e.name,
                employee_id=e.employee_id,
                created_at=e.created_at,
                is_active=e.is_active,
                face_count=len(e.face_encodings) if e.face_encodings else 0
            )
            for e in employees
        ],
        total=len(employees)
    )


@router.get("/employees/{employee_id}", response_model=EmployeeResponse)
async def get_employee(employee_id: int, db: AsyncSession = Depends(get_db)):
    """Get employee details"""
    employee = await crud.get_employee(db, employee_id)
    if not employee:
        raise HTTPException(404, "Employee not found")
    return EmployeeResponse(
        id=employee.id,
        name=employee.name,
        employee_id=employee.employee_id,
        created_at=employee.created_at,
        is_active=employee.is_active,
        face_count=len(employee.face_encodings) if employee.face_encodings else 0
    )


@router.put("/employees/{employee_id}", response_model=EmployeeResponse)
async def update_employee(
    employee_id: int,
    update: EmployeeUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update employee details"""
    employee = await crud.update_employee(
        db, employee_id,
        name=update.name,
        employee_id=update.employee_id,
        is_active=update.is_active
    )
    if not employee:
        raise HTTPException(404, "Employee not found")
    return EmployeeResponse(
        id=employee.id,
        name=employee.name,
        employee_id=employee.employee_id,
        created_at=employee.created_at,
        is_active=employee.is_active,
        face_count=len(employee.face_encodings) if employee.face_encodings else 0
    )


@router.delete("/employees/{employee_id}")
async def delete_employee(employee_id: int, db: AsyncSession = Depends(get_db)):
    """Delete an employee"""
    success = await crud.delete_employee(db, employee_id)
    if not success:
        raise HTTPException(404, "Employee not found")
    return {"message": "Employee deleted"}


@router.post("/employees/{employee_id}/faces", response_model=FaceUploadResponse)
async def upload_face_photo(
    employee_id: int,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """Upload a face photo for an employee"""
    # Verify employee exists
    employee = await crud.get_employee(db, employee_id)
    if not employee:
        raise HTTPException(404, "Employee not found")

    # Validate file type
    allowed_types = ["image/jpeg", "image/png", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(400, "Invalid file type. Use JPEG or PNG.")

    # Save the file
    file_ext = file.filename.split(".")[-1] if file.filename else "jpg"
    filename = f"{employee_id}_{uuid.uuid4().hex[:8]}.{file_ext}"
    file_path = settings.uploads_dir / filename

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    # Process face encoding
    try:
        from src.detection.face_recognizer import create_face_recognizer
        import cv2
        import numpy as np

        recognizer = create_face_recognizer()

        # Load and process image
        img_array = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            os.remove(file_path)
            raise HTTPException(400, "Could not read image file")

        # Get face encoding
        encoding = recognizer.encode_face(image)

        if encoding is None:
            os.remove(file_path)
            raise HTTPException(400, "No face detected in image")

        # Save to database
        face_encoding = await crud.add_face_encoding(
            db, employee_id, encoding, str(file_path)
        )

        return FaceUploadResponse(
            success=True,
            message="Face registered successfully",
            encoding_id=face_encoding.id
        )

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(500, f"Error processing face: {str(e)}")


# ============== Video Source Endpoints ==============

@router.post("/sources", response_model=VideoSourceResponse)
async def create_video_source(
    source: VideoSourceCreate,
    db: AsyncSession = Depends(get_db)
):
    """Add a new video source"""
    db_source = await crud.create_video_source(
        db,
        name=source.name,
        source_type=source.source_type,
        source_url=source.source_url,
        device_index=source.device_index
    )
    return VideoSourceResponse(
        id=db_source.id,
        name=db_source.name,
        source_type=db_source.source_type,
        source_url=db_source.source_url,
        device_index=db_source.device_index,
        is_active=db_source.is_active,
        created_at=db_source.created_at
    )


@router.get("/sources", response_model=VideoSourceListResponse)
async def list_video_sources(
    active_only: bool = Query(True),
    db: AsyncSession = Depends(get_db)
):
    """List all video sources"""
    sources = await crud.get_all_video_sources(db, active_only)
    return VideoSourceListResponse(
        sources=[
            VideoSourceResponse(
                id=s.id,
                name=s.name,
                source_type=s.source_type,
                source_url=s.source_url,
                device_index=s.device_index,
                is_active=s.is_active,
                created_at=s.created_at
            )
            for s in sources
        ],
        total=len(sources)
    )


@router.delete("/sources/{source_id}")
async def delete_video_source(source_id: int, db: AsyncSession = Depends(get_db)):
    """Delete a video source"""
    success = await crud.delete_video_source(db, source_id)
    if not success:
        raise HTTPException(404, "Video source not found")
    return {"message": "Video source deleted"}


# ============== Report Endpoints ==============

@router.get("/reports/daily", response_model=DailyReportResponse)
async def get_daily_report(
    report_date: Optional[date] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Get daily time tracking report"""
    if report_date is None:
        report_date = date.today()

    report_data = await crud.get_daily_report(db, report_date)

    entries = [
        TimeLogResponse(
            employee_id=r["employee_id"],
            employee_name=r["name"],
            date=report_date,
            total_seconds=r["total_seconds"],
            formatted_duration=r["formatted_duration"],
            source_name=""
        )
        for r in report_data
    ]

    total_seconds = sum(e.total_seconds for e in entries)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    return DailyReportResponse(
        date=report_date,
        entries=entries,
        total_seconds=total_seconds,
        total_formatted=f"{hours:02d}:{minutes:02d}",
        employee_count=len(entries)
    )


@router.get("/reports/employee/{employee_id}", response_model=EmployeeTimeHistory)
async def get_employee_time_history(
    employee_id: int,
    start_date: Optional[date] = Query(None),
    end_date: Optional[date] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Get time history for a specific employee"""
    employee = await crud.get_employee(db, employee_id)
    if not employee:
        raise HTTPException(404, "Employee not found")

    if end_date is None:
        end_date = date.today()
    if start_date is None:
        start_date = end_date - timedelta(days=30)

    time_logs = await crud.get_employee_time_history(db, employee_id, start_date, end_date)

    entries = [
        TimeLogResponse(
            employee_id=employee_id,
            employee_name=employee.name,
            date=log.log_date,
            total_seconds=log.total_seconds,
            formatted_duration=log.formatted_duration,
            source_name=log.source_name
        )
        for log in time_logs
    ]

    total_seconds = sum(e.total_seconds for e in entries)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    days_present = len(set(e.date for e in entries))

    return EmployeeTimeHistory(
        employee_id=employee_id,
        employee_name=employee.name,
        start_date=start_date,
        end_date=end_date,
        total_seconds=total_seconds,
        total_formatted=f"{hours:02d}:{minutes:02d}",
        days_present=days_present,
        daily_entries=entries
    )
