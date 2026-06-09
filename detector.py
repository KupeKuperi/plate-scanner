"""
Plate detection and OCR module.

Responsibilities:
  1. Pre-process each frame (resize, greyscale, enhance contrast).
  2. Run EasyOCR on the pre-processed frame.
  3. Normalise raw OCR strings and validate them against Georgian plate patterns.
"""
import logging
import re

import cv2
import easyocr
import numpy as np

from config import MAX_FRAME_WIDTH, MIN_CONFIDENCE, OCR_LANGUAGES, OCR_USE_GPU

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Georgian licence-plate regex patterns
# ---------------------------------------------------------------------------
# AA-111-AA  →  2 letters  + dash + 3 digits + dash + 2 letters
# AAA-111    →  3 letters  + dash + 3 digits
_PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{2}-\d{3}-[A-Z]{2}$"),
    re.compile(r"^[A-Z]{3}-\d{3}$"),
]


def _preprocess(frame: np.ndarray) -> np.ndarray:
    """Resize and enhance contrast to improve OCR accuracy."""
    h, w = frame.shape[:2]
    if w > MAX_FRAME_WIDTH:
        scale = MAX_FRAME_WIDTH / w
        frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return frame


def _normalize(raw: str) -> str:
    """
    Coerce a raw OCR string toward expected Georgian plate format.

    Steps:
      - Uppercase and strip whitespace.
      - Replace spaces / underscores with dashes.
      - Remove every character that is not A-Z, 0-9, or a dash.
    """
    text = raw.upper().strip()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"[^A-Z0-9\-]", "", text)
    return text


def _is_valid_plate(text: str) -> bool:
    return any(pattern.match(text) for pattern in _PLATE_PATTERNS)


class PlateDetector:
    """
    Wraps EasyOCR and exposes a single :meth:`detect` method that returns
    validated Georgian licence-plate detections from a BGR frame.
    """

    def __init__(self) -> None:
        logger.info(
            "Initialising EasyOCR reader — the first run downloads model weights "
            "(~100 MB). This may take a minute…"
        )
        self._reader = easyocr.Reader(OCR_LANGUAGES, gpu=OCR_USE_GPU)
        logger.info("EasyOCR ready.")

    def detect(self, frame: np.ndarray) -> list:
        """
        Analyse *frame* and return a list of validated detections.

        Each element is a tuple:  (plate_text, confidence, bounding_box)
          - plate_text   : normalised string, e.g. "AA-111-BB"
          - confidence   : float in [0, 1]
          - bounding_box : list of four [x, y] corner points from EasyOCR
        """
        processed = _preprocess(frame)
        raw_results = self._reader.readtext(processed)

        detections = []
        for bbox, text, confidence in raw_results:
            if confidence < MIN_CONFIDENCE:
                continue
            normalised = _normalize(text)
            if _is_valid_plate(normalised):
                logger.debug(
                    f"  OCR hit: '{text}' → '{normalised}'  conf={confidence:.2f}"
                )
                detections.append((normalised, confidence, bbox))

        return detections
