# YOLO Time Tracker

Real-time person detection and time tracking system using YOLOv8 for detection and face recognition for employee identification. Track how long employees are present in a monitored area with automatic identification.

## Features

- **Real-time Person Detection** - YOLOv8-based person detection with configurable confidence threshold
- **Face Recognition** - Automatic employee identification using face recognition
- **Time Tracking** - Track presence duration per employee with session management
- **Multiple Video Sources** - Support for webcam, RTSP streams, and video files
- **Live Streaming** - WebSocket-based real-time video preview with annotations
- **REST API** - Full API for all operations
- **Web Dashboard** - Modern web UI for monitoring and management
- **Demo Mode** - Test the system without YOLO model (for development)

## Requirements

- Python 3.9+
- OpenCV
- CUDA-capable GPU (optional, for better performance)

## Installation

### 1. Clone the Repository

```bash
git clone git@github.com:igun997/time-tracker.git
cd time-tracker
```

### 2. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO Model

Download the YOLOv8 model and place it in the `models/` directory:

```bash
# Option 1: Download directly
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/

# Option 2: Using Python
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
mv yolov8n.pt models/
```

**Available models** (choose based on your hardware):
| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| yolov8n.pt | 6 MB | Fastest | Good |
| yolov8s.pt | 22 MB | Fast | Better |
| yolov8m.pt | 52 MB | Medium | Great |
| yolov8l.pt | 87 MB | Slow | Best |

### 5. Install Face Recognition Dependencies

For face recognition to work, you need dlib:

```bash
# Ubuntu/Debian
sudo apt-get install cmake libopenblas-dev liblapack-dev

# Then install face_recognition
pip install face-recognition
```

## Usage

### Start the Server

```bash
python main.py
```

The server will start at `http://localhost:8000`

### Access the Web UI

Open your browser and navigate to:
```
http://localhost:8000
```

### Quick Start Guide

1. **Add a Video Source**
   - Click "Add Video Source" in the UI
   - Choose type: Webcam (device 0), RTSP URL, or video file path
   - Give it a name (e.g., "Office Camera")

2. **Register Employees**
   - Click "Register Employee"
   - Enter employee name and optional ID
   - Upload face photos for recognition

3. **Start Detection**
   - Select your video source from the dropdown
   - Click "Start Detection"
   - Watch the live feed with person detection

4. **View Reports**
   - Go to "Reports" tab
   - Select a date to view time tracking data

## Configuration

Configuration can be done via environment variables or `.env` file:

```env
# .env file
DEBUG=true

# YOLO Settings
YOLO_MODEL_PATH=models/yolov8n.pt
YOLO_CONFIDENCE=0.5
YOLO_DEVICE=auto  # auto, cpu, cuda, cuda:0

# Face Recognition
FACE_RECOGNITION_TOLERANCE=0.6  # Lower = stricter matching
FACE_RECOGNITION_MODEL=hog     # hog (CPU) or cnn (GPU)

# Tracking
TRACK_TIMEOUT_SECONDS=30       # Seconds before marking person as "exited"
MIN_SESSION_SECONDS=5          # Minimum session duration to record

# Video Processing
FRAME_SKIP=2                   # Process every Nth frame
MAX_FPS=30
JPEG_QUALITY=80
```

## API Reference

### Employees

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/employees` | List all employees |
| POST | `/api/employees` | Create employee |
| GET | `/api/employees/{id}` | Get employee details |
| PUT | `/api/employees/{id}` | Update employee |
| DELETE | `/api/employees/{id}` | Delete employee |
| POST | `/api/employees/{id}/faces` | Upload face photo |

### Video Sources

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/sources` | List all sources |
| POST | `/api/sources` | Add video source |
| DELETE | `/api/sources/{id}` | Delete source |

### Detection

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/detection/start` | Start detection |
| POST | `/api/detection/stop` | Stop detection |
| GET | `/api/detection/status` | Get detection status |
| POST | `/api/detection/identify` | Manually identify person |

### Reports

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/reports/daily` | Get daily time report |
| GET | `/api/reports/employee/{id}` | Get employee time history |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `ws://host/ws/stream` | Video stream with detections |
| `ws://host/ws/events` | Session events (start/end) |

## API Examples

### Create Employee

```bash
curl -X POST http://localhost:8000/api/employees \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "employee_id": "EMP001"}'
```

### Upload Face Photo

```bash
curl -X POST http://localhost:8000/api/employees/1/faces \
  -F "file=@john_face.jpg"
```

### Add Webcam Source

```bash
curl -X POST http://localhost:8000/api/sources \
  -H "Content-Type: application/json" \
  -d '{"name": "Office Webcam", "source_type": "webcam", "device_index": 0}'
```

### Add RTSP Source

```bash
curl -X POST http://localhost:8000/api/sources \
  -H "Content-Type: application/json" \
  -d '{"name": "IP Camera", "source_type": "rtsp", "source_url": "rtsp://192.168.1.100:554/stream"}'
```

### Start Detection

```bash
curl -X POST http://localhost:8000/api/detection/start \
  -H "Content-Type: application/json" \
  -d '{"source_id": 1, "demo_mode": false}'
```

### Get Daily Report

```bash
curl "http://localhost:8000/api/reports/daily?report_date=2025-12-17"
```

## Project Structure

```
time-tracker/
├── main.py                 # FastAPI application entry
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── models/                 # YOLO model files
│   └── yolov8n.pt
├── src/
│   ├── api/
│   │   ├── routes.py       # REST API endpoints
│   │   ├── schemas.py      # Pydantic models
│   │   └── websocket.py    # WebSocket & detection pipeline
│   ├── database/
│   │   ├── models.py       # SQLAlchemy models
│   │   └── crud.py         # Database operations
│   ├── detection/
│   │   ├── yolo_detector.py    # YOLO wrapper
│   │   ├── face_recognizer.py  # Face recognition
│   │   └── tracker.py          # Object tracking
│   ├── tracking/
│   │   ├── session_manager.py  # Session management
│   │   └── time_calculator.py  # Time calculations
│   └── video/
│       └── capture.py      # Video source handling
├── static/
│   ├── css/style.css
│   └── js/app.js
├── templates/
│   └── index.html
├── uploads/                # Face photo uploads
└── data/
    └── timetracker.db      # SQLite database
```

## Demo Mode

For testing without a YOLO model or camera:

1. Check "Demo Mode" in the UI before starting detection
2. Or via API:
   ```bash
   curl -X POST http://localhost:8000/api/detection/start \
     -H "Content-Type: application/json" \
     -d '{"source_id": 1, "demo_mode": true}'
   ```

Demo mode simulates person detection with mock data.

## Deployment

### Production with Uvicorn

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

Note: Use only 1 worker since the detection pipeline uses shared state.

### With GPU (NVIDIA T4 or similar)

1. Install CUDA drivers
2. Install PyTorch with CUDA support:
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```
3. Set device in config:
   ```env
   YOLO_DEVICE=cuda
   FACE_RECOGNITION_MODEL=cnn
   ```

### Docker (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Face recognition not working

- Ensure `dlib` is properly installed
- Check that uploaded photos clearly show faces
- Try adjusting `FACE_RECOGNITION_TOLERANCE` (lower = stricter)

### YOLO model not loading

- Verify model file exists in `models/yolov8n.pt`
- Check file permissions
- Try downloading the model again

### WebSocket connection failing

- Ensure you're using the correct protocol (ws:// or wss://)
- Check if detection is running before connecting

### Low FPS

- Increase `FRAME_SKIP` value
- Use a smaller YOLO model (yolov8n)
- Enable GPU acceleration

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request
