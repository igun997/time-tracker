import cv2
import threading
import time
from abc import ABC, abstractmethod
from typing import Optional, Generator
from dataclasses import dataclass
from enum import Enum
import numpy as np


class SourceType(Enum):
    WEBCAM = "webcam"
    RTSP = "rtsp"
    FILE = "file"


@dataclass
class FrameData:
    frame: np.ndarray
    timestamp: float
    frame_number: int
    source_name: str


class VideoSource(ABC):
    """Abstract base class for video sources"""

    def __init__(self, name: str):
        self.name = name
        self._cap: Optional[cv2.VideoCapture] = None
        self._is_running = False
        self._frame_count = 0
        self._lock = threading.Lock()

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to video source"""
        pass

    def read_frame(self) -> Optional[FrameData]:
        """Read a single frame from the source"""
        if not self._cap or not self._cap.isOpened():
            return None

        with self._lock:
            ret, frame = self._cap.read()
            if not ret:
                return None

            self._frame_count += 1
            return FrameData(
                frame=frame,
                timestamp=time.time(),
                frame_number=self._frame_count,
                source_name=self.name
            )

    def release(self):
        """Release the video source"""
        self._is_running = False
        if self._cap:
            self._cap.release()
            self._cap = None

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def fps(self) -> float:
        if self._cap:
            return self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        return 30.0

    @property
    def frame_size(self) -> tuple:
        if self._cap:
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (640, 480)


class WebcamSource(VideoSource):
    """Video source from USB/built-in webcam"""

    def __init__(self, name: str, device_index: int = 0):
        super().__init__(name)
        self.device_index = device_index
        self.source_type = SourceType.WEBCAM

    def connect(self) -> bool:
        try:
            self._cap = cv2.VideoCapture(self.device_index)
            if not self._cap.isOpened():
                return False
            # Set some properties for better performance
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._is_running = True
            return True
        except Exception as e:
            print(f"Error connecting to webcam {self.device_index}: {e}")
            return False


class RTSPSource(VideoSource):
    """Video source from RTSP stream (IP cameras)"""

    def __init__(self, name: str, rtsp_url: str):
        super().__init__(name)
        self.rtsp_url = rtsp_url
        self.source_type = SourceType.RTSP

    def connect(self) -> bool:
        try:
            # Use FFMPEG backend for better RTSP support
            self._cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            if not self._cap.isOpened():
                # Try with default backend
                self._cap = cv2.VideoCapture(self.rtsp_url)

            if not self._cap.isOpened():
                return False

            # Set buffer size to reduce latency
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self._is_running = True
            return True
        except Exception as e:
            print(f"Error connecting to RTSP stream {self.rtsp_url}: {e}")
            return False


class FileSource(VideoSource):
    """Video source from video file"""

    def __init__(self, name: str, file_path: str, loop: bool = False):
        super().__init__(name)
        self.file_path = file_path
        self.loop = loop
        self.source_type = SourceType.FILE
        self._total_frames = 0

    def connect(self) -> bool:
        try:
            self._cap = cv2.VideoCapture(self.file_path)
            if not self._cap.isOpened():
                return False
            self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self._is_running = True
            return True
        except Exception as e:
            print(f"Error opening video file {self.file_path}: {e}")
            return False

    def read_frame(self) -> Optional[FrameData]:
        """Read frame with optional looping for video files"""
        frame_data = super().read_frame()

        if frame_data is None and self.loop and self._cap:
            # Reset to beginning if looping
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._frame_count = 0
            frame_data = super().read_frame()

        return frame_data

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def current_position(self) -> float:
        """Returns current position as percentage (0-100)"""
        if self._cap and self._total_frames > 0:
            current = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
            return (current / self._total_frames) * 100
        return 0


class VideoSourceFactory:
    """Factory for creating video sources"""

    @staticmethod
    def create(source_type: str, name: str, **kwargs) -> VideoSource:
        if source_type == "webcam":
            device_index = kwargs.get("device_index", 0)
            return WebcamSource(name, device_index)
        elif source_type == "rtsp":
            rtsp_url = kwargs.get("source_url")
            if not rtsp_url:
                raise ValueError("RTSP source requires 'source_url'")
            return RTSPSource(name, rtsp_url)
        elif source_type == "file":
            file_path = kwargs.get("source_url")
            if not file_path:
                raise ValueError("File source requires 'source_url' (file path)")
            loop = kwargs.get("loop", False)
            return FileSource(name, file_path, loop)
        else:
            raise ValueError(f"Unknown source type: {source_type}")


class VideoCapture:
    """High-level video capture manager with frame skipping"""

    def __init__(self, source: VideoSource, frame_skip: int = 1, max_fps: int = 30):
        self.source = source
        self.frame_skip = max(1, frame_skip)
        self.max_fps = max_fps
        self._frame_interval = 1.0 / max_fps
        self._last_frame_time = 0

    def start(self) -> bool:
        return self.source.connect()

    def stop(self):
        self.source.release()

    def get_frames(self) -> Generator[FrameData, None, None]:
        """Generator that yields frames with frame skipping and rate limiting"""
        frame_count = 0

        while self.source.is_opened:
            frame_data = self.source.read_frame()
            if frame_data is None:
                break

            frame_count += 1

            # Skip frames based on frame_skip setting
            if frame_count % self.frame_skip != 0:
                continue

            # Rate limiting
            current_time = time.time()
            elapsed = current_time - self._last_frame_time
            if elapsed < self._frame_interval:
                time.sleep(self._frame_interval - elapsed)

            self._last_frame_time = time.time()
            yield frame_data

    @property
    def is_running(self) -> bool:
        return self.source.is_opened
