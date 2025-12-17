from datetime import datetime, date
from typing import Optional, List
import numpy as np
from sqlalchemy import select, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import (
    Employee, FaceEncoding, Session, TimeLog, VideoSource,
    ObjectSession, ObjectCount, ObjectLabel
)


# Employee CRUD
async def create_employee(db: AsyncSession, name: str, employee_id: Optional[str] = None) -> Employee:
    employee = Employee(name=name, employee_id=employee_id)
    db.add(employee)
    await db.commit()
    # Refetch with face_encodings relationship eagerly loaded
    result = await db.execute(
        select(Employee).where(Employee.id == employee.id).options(selectinload(Employee.face_encodings))
    )
    return result.scalar_one()


async def get_employee(db: AsyncSession, employee_pk: int) -> Optional[Employee]:
    result = await db.execute(
        select(Employee).where(Employee.id == employee_pk).options(selectinload(Employee.face_encodings))
    )
    return result.scalar_one_or_none()


async def get_employee_by_employee_id(db: AsyncSession, employee_id: str) -> Optional[Employee]:
    result = await db.execute(
        select(Employee).where(Employee.employee_id == employee_id)
    )
    return result.scalar_one_or_none()


async def get_all_employees(db: AsyncSession, active_only: bool = True) -> List[Employee]:
    query = select(Employee).options(selectinload(Employee.face_encodings))
    if active_only:
        query = query.where(Employee.is_active == True)
    result = await db.execute(query.order_by(Employee.name))
    return list(result.scalars().all())


async def update_employee(db: AsyncSession, employee_pk: int, name: Optional[str] = None,
                          employee_id: Optional[str] = None, is_active: Optional[bool] = None) -> Optional[Employee]:
    employee = await get_employee(db, employee_pk)
    if not employee:
        return None
    if name is not None:
        employee.name = name
    if employee_id is not None:
        employee.employee_id = employee_id
    if is_active is not None:
        employee.is_active = is_active
    await db.commit()
    # Refetch with face_encodings relationship eagerly loaded
    return await get_employee(db, employee_pk)


async def delete_employee(db: AsyncSession, employee_pk: int) -> bool:
    employee = await get_employee(db, employee_pk)
    if not employee:
        return False
    await db.delete(employee)
    await db.commit()
    return True


# Face Encoding CRUD
async def add_face_encoding(db: AsyncSession, employee_pk: int, encoding: np.ndarray,
                            photo_path: Optional[str] = None) -> FaceEncoding:
    face_encoding = FaceEncoding(
        employee_id=employee_pk,
        encoding=encoding.tobytes(),
        photo_path=photo_path
    )
    db.add(face_encoding)
    await db.commit()
    await db.refresh(face_encoding)
    return face_encoding


async def get_all_face_encodings(db: AsyncSession) -> List[tuple]:
    """Returns list of (employee_id, employee_name, encoding_array)"""
    result = await db.execute(
        select(FaceEncoding, Employee.name)
        .join(Employee)
        .where(Employee.is_active == True)
    )
    encodings = []
    for face_encoding, name in result.all():
        encoding_array = np.frombuffer(face_encoding.encoding, dtype=np.float64)
        encodings.append((face_encoding.employee_id, name, encoding_array))
    return encodings


async def delete_face_encoding(db: AsyncSession, encoding_id: int) -> bool:
    result = await db.execute(select(FaceEncoding).where(FaceEncoding.id == encoding_id))
    face_encoding = result.scalar_one_or_none()
    if not face_encoding:
        return False
    await db.delete(face_encoding)
    await db.commit()
    return True


# Session CRUD
async def create_session(db: AsyncSession, source_name: str, employee_pk: Optional[int] = None,
                         unknown_person_id: Optional[str] = None) -> Session:
    session = Session(
        employee_id=employee_pk,
        unknown_person_id=unknown_person_id,
        source_name=source_name,
        start_time=datetime.utcnow(),
        session_date=date.today()
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def end_session(db: AsyncSession, session_id: int) -> Optional[Session]:
    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if not session:
        return None
    session.end_time = datetime.utcnow()
    await db.commit()
    await db.refresh(session)
    return session


async def get_active_sessions(db: AsyncSession, source_name: Optional[str] = None) -> List[Session]:
    query = select(Session).where(Session.end_time == None)
    if source_name:
        query = query.where(Session.source_name == source_name)
    result = await db.execute(query.options(selectinload(Session.employee)))
    return list(result.scalars().all())


async def get_sessions_by_date(db: AsyncSession, log_date: date,
                               employee_pk: Optional[int] = None) -> List[Session]:
    query = select(Session).where(Session.session_date == log_date)
    if employee_pk:
        query = query.where(Session.employee_id == employee_pk)
    result = await db.execute(query.options(selectinload(Session.employee)).order_by(Session.start_time))
    return list(result.scalars().all())


# Time Log CRUD
async def update_time_log(db: AsyncSession, employee_pk: int, source_name: str,
                          log_date: date, duration_seconds: int) -> TimeLog:
    result = await db.execute(
        select(TimeLog).where(
            and_(
                TimeLog.employee_id == employee_pk,
                TimeLog.log_date == log_date,
                TimeLog.source_name == source_name
            )
        )
    )
    time_log = result.scalar_one_or_none()

    if time_log:
        time_log.total_seconds += duration_seconds
    else:
        time_log = TimeLog(
            employee_id=employee_pk,
            log_date=log_date,
            total_seconds=duration_seconds,
            source_name=source_name
        )
        db.add(time_log)

    await db.commit()
    await db.refresh(time_log)
    return time_log


async def get_daily_report(db: AsyncSession, log_date: date) -> List[dict]:
    """Get aggregated time for all employees on a specific date"""
    result = await db.execute(
        select(
            Employee.id,
            Employee.name,
            Employee.employee_id,
            func.sum(TimeLog.total_seconds).label('total_seconds')
        )
        .join(TimeLog)
        .where(TimeLog.log_date == log_date)
        .group_by(Employee.id)
        .order_by(Employee.name)
    )

    report = []
    for row in result.all():
        total_seconds = row.total_seconds or 0
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        report.append({
            'employee_id': row.id,
            'name': row.name,
            'employee_code': row.employee_id,
            'total_seconds': total_seconds,
            'formatted_duration': f"{hours:02d}:{minutes:02d}"
        })
    return report


async def get_employee_time_history(db: AsyncSession, employee_pk: int,
                                    start_date: date, end_date: date) -> List[TimeLog]:
    result = await db.execute(
        select(TimeLog)
        .where(
            and_(
                TimeLog.employee_id == employee_pk,
                TimeLog.log_date >= start_date,
                TimeLog.log_date <= end_date
            )
        )
        .order_by(TimeLog.log_date.desc())
    )
    return list(result.scalars().all())


# Video Source CRUD
async def create_video_source(db: AsyncSession, name: str, source_type: str,
                              source_url: Optional[str] = None,
                              device_index: Optional[int] = None) -> VideoSource:
    source = VideoSource(
        name=name,
        source_type=source_type,
        source_url=source_url,
        device_index=device_index
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


async def get_all_video_sources(db: AsyncSession, active_only: bool = True) -> List[VideoSource]:
    query = select(VideoSource)
    if active_only:
        query = query.where(VideoSource.is_active == True)
    result = await db.execute(query.order_by(VideoSource.name))
    return list(result.scalars().all())


async def get_video_source(db: AsyncSession, source_id: int) -> Optional[VideoSource]:
    result = await db.execute(select(VideoSource).where(VideoSource.id == source_id))
    return result.scalar_one_or_none()


async def delete_video_source(db: AsyncSession, source_id: int) -> bool:
    source = await get_video_source(db, source_id)
    if not source:
        return False
    await db.delete(source)
    await db.commit()
    return True


# ============== Object Session CRUD ==============

async def create_object_session(
    db: AsyncSession,
    track_id: int,
    class_id: int,
    class_name: str,
    source_name: str,
    employee_id: Optional[int] = None,
    label: Optional[str] = None
) -> ObjectSession:
    """Create a new object session when an object is first detected"""
    session = ObjectSession(
        track_id=track_id,
        class_id=class_id,
        class_name=class_name,
        source_name=source_name,
        start_time=datetime.utcnow(),
        session_date=date.today(),
        employee_id=employee_id,
        label=label
    )
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


async def end_object_session(
    db: AsyncSession,
    session_id: int,
    duration_seconds: Optional[int] = None
) -> Optional[ObjectSession]:
    """End an object session when the object leaves"""
    result = await db.execute(
        select(ObjectSession).where(ObjectSession.id == session_id)
    )
    session = result.scalar_one_or_none()
    if not session:
        return None

    session.end_time = datetime.utcnow()
    if duration_seconds is not None:
        session.duration_seconds = duration_seconds
    else:
        session.duration_seconds = int((session.end_time - session.start_time).total_seconds())

    await db.commit()
    await db.refresh(session)
    return session


async def get_active_object_sessions(
    db: AsyncSession,
    source_name: Optional[str] = None
) -> List[ObjectSession]:
    """Get all active (not ended) object sessions"""
    query = select(ObjectSession).where(ObjectSession.end_time == None)
    if source_name:
        query = query.where(ObjectSession.source_name == source_name)
    result = await db.execute(query)
    return list(result.scalars().all())


async def get_object_sessions_by_date(
    db: AsyncSession,
    log_date: date,
    class_id: Optional[int] = None,
    source_name: Optional[str] = None
) -> List[ObjectSession]:
    """Get object sessions for a specific date"""
    query = select(ObjectSession).where(ObjectSession.session_date == log_date)
    if class_id is not None:
        query = query.where(ObjectSession.class_id == class_id)
    if source_name:
        query = query.where(ObjectSession.source_name == source_name)
    result = await db.execute(query.order_by(ObjectSession.start_time.desc()))
    return list(result.scalars().all())


async def get_object_duration_report(
    db: AsyncSession,
    log_date: date,
    source_name: Optional[str] = None
) -> List[dict]:
    """Get duration report grouped by class for a specific date"""
    query = select(
        ObjectSession.class_id,
        ObjectSession.class_name,
        func.count(ObjectSession.id).label('session_count'),
        func.sum(ObjectSession.duration_seconds).label('total_seconds'),
        func.avg(ObjectSession.duration_seconds).label('avg_seconds')
    ).where(
        and_(
            ObjectSession.session_date == log_date,
            ObjectSession.end_time != None  # Only completed sessions
        )
    )

    if source_name:
        query = query.where(ObjectSession.source_name == source_name)

    query = query.group_by(ObjectSession.class_id, ObjectSession.class_name)
    result = await db.execute(query)

    report = []
    for row in result.all():
        total_seconds = row.total_seconds or 0
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        report.append({
            'class_id': row.class_id,
            'class_name': row.class_name,
            'session_count': row.session_count,
            'total_seconds': total_seconds,
            'avg_seconds': int(row.avg_seconds or 0),
            'formatted_duration': f"{hours:02d}:{minutes:02d}"
        })
    return report


# ============== Object Count CRUD ==============

async def increment_object_count(
    db: AsyncSession,
    source_name: str,
    class_id: int,
    class_name: str,
    direction: str,  # "entry" or "exit"
    hourly: bool = True
) -> ObjectCount:
    """Increment entry or exit count for an object class"""
    current_hour = datetime.utcnow().hour if hourly else None
    current_date = date.today()

    # Find or create the count record
    result = await db.execute(
        select(ObjectCount).where(
            and_(
                ObjectCount.source_name == source_name,
                ObjectCount.class_id == class_id,
                ObjectCount.count_date == current_date,
                ObjectCount.hour == current_hour
            )
        )
    )
    count_record = result.scalar_one_or_none()

    if count_record:
        if direction == "entry":
            count_record.entry_count += 1
        else:
            count_record.exit_count += 1
    else:
        count_record = ObjectCount(
            source_name=source_name,
            class_id=class_id,
            class_name=class_name,
            count_date=current_date,
            hour=current_hour,
            entry_count=1 if direction == "entry" else 0,
            exit_count=1 if direction == "exit" else 0
        )
        db.add(count_record)

    await db.commit()
    await db.refresh(count_record)
    return count_record


async def get_hourly_counts(
    db: AsyncSession,
    count_date: date,
    source_name: Optional[str] = None,
    class_id: Optional[int] = None
) -> List[ObjectCount]:
    """Get hourly counts for a date"""
    query = select(ObjectCount).where(
        and_(
            ObjectCount.count_date == count_date,
            ObjectCount.hour != None
        )
    )
    if source_name:
        query = query.where(ObjectCount.source_name == source_name)
    if class_id is not None:
        query = query.where(ObjectCount.class_id == class_id)

    result = await db.execute(query.order_by(ObjectCount.hour))
    return list(result.scalars().all())


async def get_daily_counts(
    db: AsyncSession,
    count_date: date,
    source_name: Optional[str] = None
) -> List[dict]:
    """Get daily totals grouped by class"""
    query = select(
        ObjectCount.class_id,
        ObjectCount.class_name,
        func.sum(ObjectCount.entry_count).label('total_entries'),
        func.sum(ObjectCount.exit_count).label('total_exits')
    ).where(ObjectCount.count_date == count_date)

    if source_name:
        query = query.where(ObjectCount.source_name == source_name)

    query = query.group_by(ObjectCount.class_id, ObjectCount.class_name)
    result = await db.execute(query)

    return [
        {
            'class_id': row.class_id,
            'class_name': row.class_name,
            'total_entries': row.total_entries or 0,
            'total_exits': row.total_exits or 0,
            'total': (row.total_entries or 0) + (row.total_exits or 0)
        }
        for row in result.all()
    ]


# ============== Object Label CRUD ==============

async def create_object_label(
    db: AsyncSession,
    name: str,
    class_id: int,
    class_name: str
) -> ObjectLabel:
    """Create a label for an object type"""
    label = ObjectLabel(
        name=name,
        class_id=class_id,
        class_name=class_name
    )
    db.add(label)
    await db.commit()
    await db.refresh(label)
    return label


async def get_all_object_labels(db: AsyncSession) -> List[ObjectLabel]:
    """Get all object labels"""
    result = await db.execute(select(ObjectLabel).order_by(ObjectLabel.name))
    return list(result.scalars().all())


async def delete_object_label(db: AsyncSession, label_id: int) -> bool:
    """Delete an object label"""
    result = await db.execute(select(ObjectLabel).where(ObjectLabel.id == label_id))
    label = result.scalar_one_or_none()
    if not label:
        return False
    await db.delete(label)
    await db.commit()
    return True
