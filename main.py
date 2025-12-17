import asyncio
from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, WebSocket, Request, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from src.database.models import init_db, get_async_session_maker
from src.database import crud
from src.api.routes import router as api_router
from src.api.websocket import (
    detection_pipeline,
    websocket_stream_handler,
    websocket_events_handler,
    websocket_client_cam_handler
)
from src.api.schemas import DetectionConfig, DetectionStatus, APIStatus

# Global database session maker
async_session_maker = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global async_session_maker

    # Startup
    print(f"Starting {settings.app_name}...")

    # Initialize database
    engine = await init_db(settings.database_url)
    async_session_maker = get_async_session_maker(engine)
    print("Database initialized")

    # Load face encodings into recognizer if detection is running
    # This will be done when detection starts

    yield

    # Shutdown
    print("Shutting down...")
    if detection_pipeline.state.is_running:
        detection_pipeline.stop()

    await engine.dispose()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Include API router
app.include_router(api_router, prefix="/api")


# ============== Root and UI Routes ==============

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve main UI"""
    return templates.TemplateResponse("index.html", {"request": request})


# ============== Detection Control Endpoints ==============

@app.post("/api/detection/start")
async def start_detection(config: DetectionConfig):
    """Start detection on a video source"""
    global async_session_maker

    if detection_pipeline.state.is_running:
        raise HTTPException(400, "Detection is already running")

    # Get video source from database
    async with async_session_maker() as db:
        source = await crud.get_video_source(db, config.source_id)
        if not source:
            raise HTTPException(404, "Video source not found")

        # Load known faces
        faces_data = await crud.get_all_face_encodings(db)

    # Callback for saving sessions to database (presence mode)
    async def save_object_session(session):
        async with async_session_maker() as db:
            # Save to object_sessions table (tracks all objects)
            await crud.create_object_session(
                db,
                track_id=session.track_id,
                class_id=session.class_id,
                class_name=session.class_name,
                source_name=session.source_name,
                employee_id=session.employee_id,
                label=session.label
            )
            # Also update employee time_logs if identified person
            if session.employee_id and session.is_person:
                await crud.update_time_log(
                    db,
                    session.employee_id,
                    session.source_name,
                    date.today(),
                    session.duration_seconds
                )

    # Callback for counting entries (counter mode)
    async def count_entry(session):
        async with async_session_maker() as db:
            await crud.increment_object_count(
                db,
                source_name=session.source_name,
                class_id=session.class_id,
                class_name=session.class_name,
                direction="entry",
                hourly=settings.counter_hourly
            )

    # Callback for counting exits (counter mode)
    async def count_exit(session):
        async with async_session_maker() as db:
            await crud.increment_object_count(
                db,
                source_name=session.source_name,
                class_id=session.class_id,
                class_name=session.class_name,
                direction="exit",
                hourly=settings.counter_hourly
            )

    # Start detection with tracking mode
    success = await detection_pipeline.start(
        source_type=source.source_type,
        source_name=source.name,
        source_url=source.source_url,
        device_index=source.device_index or 0,
        demo_mode=config.demo_mode,
        frame_skip=config.frame_skip,
        confidence_threshold=config.confidence_threshold,
        detection_classes=config.detection_classes,
        tracking_mode=config.tracking_mode,
        db_session_callback=lambda s: asyncio.create_task(save_object_session(s)),
        db_entry_callback=lambda s: asyncio.create_task(count_entry(s)),
        db_exit_callback=lambda s: asyncio.create_task(count_exit(s))
    )

    if not success:
        raise HTTPException(500, "Failed to start detection")

    # Load known faces into recognizer
    if detection_pipeline._face_recognizer and faces_data:
        detection_pipeline._face_recognizer.load_known_faces(faces_data)

    # Get the actual classes being detected
    from config import get_class_name
    detected_classes = [get_class_name(c) for c in detection_pipeline.state.detection_classes]

    return {
        "message": "Detection started",
        "source": source.name,
        "detection_classes": detected_classes,
        "tracking_mode": config.tracking_mode
    }


@app.post("/api/detection/stop")
async def stop_detection():
    """Stop detection"""
    if not detection_pipeline.state.is_running:
        raise HTTPException(400, "Detection is not running")

    detection_pipeline.stop()
    return {"message": "Detection stopped"}


@app.get("/api/detection/status", response_model=DetectionStatus)
async def get_detection_status():
    """Get current detection status"""
    status = detection_pipeline.get_status()
    return DetectionStatus(**status)


@app.post("/api/detection/identify")
async def manual_identify(track_id: int = Query(...), employee_id: int = Query(...)):
    """Manually identify a tracked person"""
    global async_session_maker

    if not detection_pipeline.state.is_running:
        raise HTTPException(400, "Detection is not running")

    # Get employee name
    async with async_session_maker() as db:
        employee = await crud.get_employee(db, employee_id)
        if not employee:
            raise HTTPException(404, "Employee not found")

    detection_pipeline.manual_identify(track_id, employee_id, employee.name)
    return {"message": f"Person {track_id} identified as {employee.name}"}


# ============== WebSocket Endpoints ==============

@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """WebSocket endpoint for video streaming"""
    await websocket_stream_handler(websocket)


@app.websocket("/ws/events")
async def websocket_events(websocket: WebSocket):
    """WebSocket endpoint for session events"""
    await websocket_events_handler(websocket)


@app.websocket("/ws/client-cam")
async def websocket_client_cam(
    websocket: WebSocket,
    demo: bool = False,
    classes: str = None,
    mode: str = "presence"
):
    """WebSocket endpoint for client webcam streaming

    Connect to this endpoint to stream your local webcam to the server for processing.
    The server will process frames and return annotated results.

    Query params:
        demo: Set to true for demo mode (no YOLO model required)
        classes: Detection classes (e.g., "person", "traffic", "person,car,motorcycle")
        mode: Tracking mode - "presence", "counter", or "both"

    Protocol:
        Send: {"type": "frame", "frame": "<base64 jpeg data>"}
        Receive: {"type": "frame", "frame": "<annotated base64>", "detections": [...], "stats": {...}}
        Send: {"type": "stop"} to end the session
    """
    await websocket_client_cam_handler(
        websocket,
        demo_mode=demo,
        detection_classes=classes,
        tracking_mode=mode
    )


# ============== API Status ==============

@app.get("/api/status", response_model=APIStatus)
async def get_api_status():
    """Get API status"""
    global async_session_maker

    async with async_session_maker() as db:
        employees = await crud.get_all_employees(db)
        sources = await crud.get_all_video_sources(db)

    # Check YOLO availability
    try:
        from ultralytics import YOLO
        yolo_available = True
    except ImportError:
        yolo_available = False

    # Check face_recognition availability
    try:
        import face_recognition
        face_recognition_available = True
    except ImportError:
        face_recognition_available = False

    return APIStatus(
        status="ok",
        version="1.0.0",
        detection_running=detection_pipeline.state.is_running,
        active_sources=len(sources),
        total_employees=len(employees),
        yolo_loaded=detection_pipeline._detector is not None and detection_pipeline._detector.is_loaded if detection_pipeline._detector else False,
        face_recognition_available=face_recognition_available
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
