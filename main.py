"""
ANPR System — entry point.

Flow:
  1. Open webcam (or RTSP stream — see config.py).
  2. Every FRAME_SKIP frames, run the plate detector.
  3. For each detected plate:
       - If outside the cooldown window → save to JSON, log [SAVED].
       - If inside  the cooldown window → log [SKIP] with remaining time.
  4. Draw bounding boxes and plate text on the live preview window.
  5. On exit, print a summary of all recorded plates.
"""
import logging
import sys
import time

import cv2
import numpy as np

from config import COOLDOWN_SECONDS, FRAME_SKIP, VIDEO_SOURCE, WINDOW_TITLE
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
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    detector = PlateDetector()

    # cooldown_tracker maps  plate_text → unix timestamp of last record
    cooldown_tracker: dict[str, float] = {}

    source_label = f"webcam (index {VIDEO_SOURCE})" if isinstance(VIDEO_SOURCE, int) else VIDEO_SOURCE
    logger.info(f"Opening video source: {source_label}")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logger.error(
            "Could not open video source. "
            "Check that your webcam is connected, or verify the RTSP URL in config.py."
        )
        sys.exit(1)

    logger.info("Stream opened. ANPR running — press 'q' in the preview window to quit.\n")

    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read frame — stream may have ended or dropped.")
                time.sleep(0.1)
                continue

            frame_index += 1

            if frame_index % FRAME_SKIP == 0:
                detections = detector.detect(frame)

                for plate, confidence, bbox in detections:
                    now     = time.time()
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

                    _draw_detection(frame, plate, bbox, skipped)

            cv2.imshow(WINDOW_TITLE, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("'q' pressed — shutting down.")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted (Ctrl+C).")

    finally:
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
