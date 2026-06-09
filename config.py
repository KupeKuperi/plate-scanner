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
# Frame processing
# ---------------------------------------------------------------------------
# Process every Nth frame — balances CPU load vs. detection latency.
# Lower = more responsive but heavier; higher = lighter but slower to react.
FRAME_SKIP = 5

# Resize frames wider than this before OCR (pixels). Speeds up EasyOCR.
MAX_FRAME_WIDTH = 1280

# Minimum EasyOCR confidence to accept a text detection (0.0 – 1.0)
MIN_CONFIDENCE = 0.40

# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------
# How long (seconds) to suppress re-counting the same plate after it is saved.
# Default: 15 minutes = 900 seconds
COOLDOWN_SECONDS = 15 * 60

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
STORAGE_DIR  = "data"
STORAGE_FILE = os.path.join(STORAGE_DIR, "plates.json")

# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------
# 'en' covers the Latin alphabet used on Georgian plates.
# Add additional language codes here if needed.
OCR_LANGUAGES = ["en"]

# Whether to use GPU for EasyOCR. Set True only if you have CUDA installed.
OCR_USE_GPU = False

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
WINDOW_TITLE = "ANPR System  |  press 'q' to quit"
