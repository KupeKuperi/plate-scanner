"""
Central configuration for the ANPR system.
To switch input from webcam to a CCTV RTSP stream, change VIDEO_SOURCE only.
"""
import os

# ---------------------------------------------------------------------------
# Input source
# ---------------------------------------------------------------------------
# Webcam:  use the integer index (0 = first/built-in camera)
# RTSP:    "rtsp://username:password@192.168.1.100:554/stream1"
VIDEO_SOURCE = 0

# ---------------------------------------------------------------------------
# OCR performance  ← main CPU knobs
# ---------------------------------------------------------------------------
# Maximum times per second EasyOCR is allowed to run.
# 2 FPS is plenty for a car-wash entrance; lower = cooler CPU.
OCR_TARGET_FPS = 2

# Resize frames to this width before sending to EasyOCR.
# Smaller = faster OCR, less CPU heat. 640 is a good default.
MAX_FRAME_WIDTH = 640

# Minimum EasyOCR confidence to accept a detection.
MIN_CONFIDENCE = 0.40

# ---------------------------------------------------------------------------
# Motion detection gate
# ---------------------------------------------------------------------------
# When True, OCR only runs when the camera sees movement above the threshold.
# This is the biggest CPU-saver: idle scene = zero OCR work.
MOTION_DETECTION  = True
MOTION_THRESHOLD  = 3000   # non-zero pixels in the diff frame

# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------
COOLDOWN_SECONDS = 15 * 60   # 15 minutes

# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
OVERLAY_TTL = 3.0   # seconds to keep a bounding-box on screen after detection

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
STORAGE_DIR  = "data"
STORAGE_FILE = os.path.join(STORAGE_DIR, "plates.json")

# ---------------------------------------------------------------------------
# OCR language
# ---------------------------------------------------------------------------
OCR_LANGUAGES = ["en"]
OCR_USE_GPU   = False   # set True only if CUDA is installed
