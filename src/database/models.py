from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, DateTime, Date, Boolean,
    ForeignKey, LargeBinary, Float, create_engine
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class Employee(Base):
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    employee_id = Column(String(50), nullable=True, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # Relationships
    face_encodings = relationship("FaceEncoding", back_populates="employee", cascade="all, delete-orphan")
    sessions = relationship("Session", back_populates="employee", cascade="all, delete-orphan")
    time_logs = relationship("TimeLog", back_populates="employee", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Employee(id={self.id}, name='{self.name}')>"


class FaceEncoding(Base):
    __tablename__ = "face_encodings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    encoding = Column(LargeBinary, nullable=False)  # Numpy array as bytes
    photo_path = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    employee = relationship("Employee", back_populates="face_encodings")

    def __repr__(self):
        return f"<FaceEncoding(id={self.id}, employee_id={self.employee_id})>"


class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=True)  # Null for unknown persons
    unknown_person_id = Column(String(50), nullable=True)  # "Unknown-1", "Unknown-2", etc.
    source_name = Column(String(100), nullable=False)
    start_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)  # Null if session is active
    session_date = Column(Date, nullable=False, default=date.today)

    # Relationships
    employee = relationship("Employee", back_populates="sessions")

    @property
    def duration_seconds(self) -> int:
        if self.end_time:
            return int((self.end_time - self.start_time).total_seconds())
        return int((datetime.utcnow() - self.start_time).total_seconds())

    @property
    def is_active(self) -> bool:
        return self.end_time is None

    def __repr__(self):
        return f"<Session(id={self.id}, employee_id={self.employee_id}, active={self.is_active})>"


class TimeLog(Base):
    __tablename__ = "time_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    log_date = Column(Date, nullable=False)
    total_seconds = Column(Integer, default=0)
    source_name = Column(String(100), nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    employee = relationship("Employee", back_populates="time_logs")

    @property
    def formatted_duration(self) -> str:
        hours = self.total_seconds // 3600
        minutes = (self.total_seconds % 3600) // 60
        seconds = self.total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def __repr__(self):
        return f"<TimeLog(employee_id={self.employee_id}, date={self.log_date}, duration={self.formatted_duration})>"


class VideoSource(Base):
    __tablename__ = "video_sources"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    source_type = Column(String(20), nullable=False)  # "rtsp", "webcam", "file"
    source_url = Column(String(500), nullable=True)  # URL for RTSP, path for file
    device_index = Column(Integer, nullable=True)  # For webcam
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<VideoSource(name='{self.name}', type='{self.source_type}')>"


# Database initialization
async def init_db(database_url: str):
    engine = create_async_engine(database_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return engine


def get_async_session_maker(engine):
    return sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
