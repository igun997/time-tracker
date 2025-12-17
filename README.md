# YOLO Object Tracker

Sistem pelacakan objek real-time menggunakan YOLOv8 dengan pengenalan wajah untuk identifikasi karyawan.

## Fitur Utama

- **Deteksi Objek Real-time** - YOLOv8 dengan kelas yang dapat dikonfigurasi (orang, kendaraan, lalu lintas)
- **Mode Pelacakan Ganda**:
  - `presence` - Lacak durasi objek dalam frame
  - `counter` - Hitung objek masuk/keluar
  - `both` - Keduanya sekaligus
- **Pengenalan Wajah** - Identifikasi karyawan otomatis
- **Sumber Video** - Webcam, RTSP, HLS (.m3u8), file video
- **Mode Webcam Client** - Stream webcam lokal ke server remote
- **Dashboard Web** - UI modern dengan live streaming

## Instalasi

```bash
# Clone repo
git clone git@github.com:igun997/time-tracker.git
cd time-tracker

# Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download model YOLO
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/

# Install face recognition (opsional)
sudo apt-get install cmake libopenblas-dev liblapack-dev
pip install face-recognition
```

## Penggunaan

```bash
python main.py
# Buka http://localhost:8000
```

### Panduan Cepat

1. **Tambah Sumber Video** - Webcam, RTSP URL, atau file video
2. **Daftar Karyawan** - Nama + upload foto wajah
3. **Mulai Deteksi** - Pilih sumber, kelas deteksi, mode pelacakan
4. **Lihat Laporan** - Tab Reports, Object Sessions, Object Counts

## Konfigurasi (.env)

```env
# YOLO
YOLO_MODEL_PATH=models/yolov8n.pt
YOLO_CONFIDENCE=0.5
YOLO_DEVICE=auto

# Face Recognition
FACE_RECOGNITION_TOLERANCE=0.6
FACE_RECOGNITION_MODEL=hog

# Tracking
TRACKING_MODE=presence    # presence, counter, both
TRACK_TIMEOUT_SECONDS=30
MIN_SESSION_SECONDS=5

# Video
FRAME_SKIP=2
MAX_FPS=30
```

## Kelas Deteksi

| Preset | Kelas |
|--------|-------|
| `person` | Orang saja |
| `vehicles` | Mobil, motor, bus, truk |
| `traffic` | Orang + kendaraan + rambu |

## API Endpoints

### Karyawan
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET/POST | `/api/employees` | List/Tambah karyawan |
| GET/PUT/DELETE | `/api/employees/{id}` | Detail/Update/Hapus |
| POST | `/api/employees/{id}/faces` | Upload foto wajah |

### Sumber Video
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET/POST | `/api/sources` | List/Tambah sumber |
| DELETE | `/api/sources/{id}` | Hapus sumber |

### Deteksi
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| POST | `/api/detection/start` | Mulai deteksi |
| POST | `/api/detection/stop` | Stop deteksi |
| GET | `/api/detection/status` | Status deteksi |

### Laporan
| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/api/reports/daily` | Laporan harian karyawan |
| GET | `/api/reports/employee/{id}` | Riwayat waktu karyawan |
| GET | `/api/reports/objects/sessions` | Sesi objek (presence mode) |
| GET | `/api/reports/objects/counts` | Hitungan objek (counter mode) |
| GET | `/api/reports/objects/daily` | Laporan durasi per kelas |

### WebSocket
| Endpoint | Deskripsi |
|----------|-----------|
| `/ws/stream` | Stream video dengan deteksi |
| `/ws/events` | Event sesi (mulai/selesai) |
| `/ws/client-cam` | Stream webcam client |

## Contoh API

```bash
# Tambah karyawan
curl -X POST http://localhost:8000/api/employees \
  -H "Content-Type: application/json" \
  -d '{"name": "John Doe", "employee_id": "EMP001"}'

# Tambah webcam
curl -X POST http://localhost:8000/api/sources \
  -H "Content-Type: application/json" \
  -d '{"name": "Webcam", "source_type": "webcam", "device_index": 0}'

# Mulai deteksi (presence mode)
curl -X POST http://localhost:8000/api/detection/start \
  -H "Content-Type: application/json" \
  -d '{"source_id": 1, "tracking_mode": "presence"}'

# Mulai deteksi (counter mode)
curl -X POST http://localhost:8000/api/detection/start \
  -H "Content-Type: application/json" \
  -d '{"source_id": 1, "tracking_mode": "counter", "detection_classes": "person,car"}'
```

## Struktur Proyek

```
time-tracker/
├── main.py                 # Entry point FastAPI
├── config.py               # Konfigurasi
├── models/                 # File model YOLO
├── src/
│   ├── api/                # Routes, schemas, websocket
│   ├── database/           # Models, CRUD
│   ├── detection/          # YOLO, face recognition, tracker
│   ├── tracking/           # Session manager
│   └── video/              # Video capture
├── static/                 # CSS, JS
├── templates/              # HTML
└── data/                   # Database SQLite
```

## Deployment

```bash
# Production
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1

# Dengan GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# Set YOLO_DEVICE=cuda di .env
```

## Troubleshooting

| Masalah | Solusi |
|---------|--------|
| Face recognition error | Install dlib, cek foto wajah jelas |
| YOLO tidak load | Cek file models/yolov8n.pt |
| Webcam client error | Butuh HTTPS untuk remote (localhost OK) |
| FPS rendah | Naikkan FRAME_SKIP, gunakan yolov8n |

## Lisensi

MIT License
