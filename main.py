"""
ANPR System — entry point.

Architecture (fixes camera lag):
  - Main thread  : reads frames from the camera and displays them at full FPS.
                   It never waits for OCR.
  - OCR thread   : runs EasyOCR on whatever the latest available frame is.
                   When it finishes, it immediately picks up the next frame.

Detection overlays stay on screen for OVERLAY_TTL seconds so you can see
results even between OCR cycles.
"""
import logging
import sys
import threading
import time

import cv2
import numpy as np

from config import COOLDOWN_SECONDS, OVERLAY_TTL, VIDEO_SOURCE, WINDOW_TITLE
from detector import PlateDetector
from storage import get_all, record_plate

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
_GREEN  = (0, 220, 0)
_YELLOW = (0, 200, 220)
_FONT   = cv2.FONT_HERSHEY_SIMPLEX


def _draw_detection(frame: np.ndarray, plate: str, bbox: list, skipped: bool) -> None:
    colour = _YELLOW if skipped else _GREEN
    pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=2)
    label_x = pts[0][0]
    label_y = max(pts[0][1] - 10, 20)
    cv2.putText(frame, plate, (label_x, label_y), _FONT, 0.85, colour, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# OCR background thread
# ---------------------------------------------------------------------------
def _ocr_worker(
    detector: PlateDetector,
    pending_frame,       # list[ndarray | None]  — single-slot latest frame
    pending_lock,
    overlays,            # list of (plate, conf, bbox, skipped, expire_time)
    overlays_lock,
    cooldown_tracker,    # dict[str, float]
    stop_event,
) -> None:
    while not stop_event.is_set():
        # Grab the latest frame (if any)
        with pending_lock:
            frame = pending_frame[0]
            pending_frame[0] = None

        if frame is None:
            time.sleep(0.01)
            continue

        detections = detector.detect(frame)
        if not detections:
            continue

        now = time.time()
        new_overlays = []

        for plate, confidence, bbox in detections:
            elapsed = now - cooldown_tracker.get(plate, 0.0)
            skipped = elapsed < COOLDOWN_SECONDS

            if skipped:
                remaining_min = int((COOLDOWN_SECONDS - elapsed) / 60)
                remaining_sec = int((COOLDOWN_SECONDS - elapsed) % 60)
                logger.info(
                    f"[SKIP]   {plate:<14}  conf={confidence:.2f}  "
                    f"cooldown: {remaining_min:02d}:{remaining_sec:02d} remaining"
                )
            else:
                count = record_plate(plate)
                cooldown_tracker[plate] = now
                logger.info(
                    f"[SAVED]  {plate:<14}  conf={confidence:.2f}  "
                    f"total scans: {count}"
                )

            new_overlays.append((plate, confidence, bbox, skipped, now + OVERLAY_TTL))

        with overlays_lock:
            overlays[:] = new_overlays


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    detector = PlateDetector()

    cooldown_tracker: dict = {}

    # Shared state between main thread and OCR thread
    pending_frame  = [None]
    pending_lock   = threading.Lock()
    overlays       = []
    overlays_lock  = threading.Lock()
    stop_event     = threading.Event()

    ocr_thread = threading.Thread(
        target=_ocr_worker,
        args=(detector, pending_frame, pending_lock,
              overlays, overlays_lock, cooldown_tracker, stop_event),
        daemon=True,
        name="OCR-thread",
    )

    source_label = f"webcam (index {VIDEO_SOURCE})" if isinstance(VIDEO_SOURCE, int) else VIDEO_SOURCE
    logger.info(f"Opening video source: {source_label}")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error(
            "Could not open video source. "
            "Check that your webcam is connected, or verify the RTSP URL in config.py."
        )
        sys.exit(1)

    # Reduce the camera's internal buffer to 1 frame — key fix for lag.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    ocr_thread.start()
    logger.info("Stream opened. ANPR running — press 'q' in the preview window to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame — stream may have ended or dropped.")
                time.sleep(0.1)
                continue

            # Always feed the latest frame to the OCR thread (overwrites any unprocessed one)
            with pending_lock:
                pending_frame[0] = frame.copy()

            # Draw overlays that haven't expired yet
            now = time.time()
            with overlays_lock:
                overlays[:] = [o for o in overlays if o[4] > now]
                active = list(overlays)

            for plate, conf, bbox, skipped, _ in active:
                _draw_detection(frame, plate, bbox, skipped)

            cv2.imshow(WINDOW_TITLE, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("'q' pressed — shutting down.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted (Ctrl+C).")

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        _print_summary()


def _print_summary() -> None:
    all_plates = get_all()
    logger.info("")
    logger.info("=" * 45)
    logger.info("  SESSION SUMMARY")
    logger.info("=" * 45)
    if all_plates:
        for plate, count in sorted(all_plates.items()):
            logger.info(f"  {plate:<14}  {count:>4} scan(s)")
    else:
        logger.info("  No plates were recorded.")
    logger.info("=" * 45)


if __name__ == "__main__":
    main()
