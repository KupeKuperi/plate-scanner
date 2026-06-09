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


# ---------------------------------------------------------------------------
# Position-aware character correction
# ---------------------------------------------------------------------------
# OCR often confuses visually similar characters.
# Georgian plates have a known structure, so we know which positions must be
# letters and which must be digits, and can fix common mix-ups accordingly.

_DIGIT_TO_LETTER = {"0": "O", "1": "I", "5": "S", "8": "B", "6": "G"}
_LETTER_TO_DIGIT = {"O": "0", "I": "1", "L": "1", "S": "5", "B": "8", "G": "6"}


def _fix_plate_chars(s: str) -> str:
    """
    Given a dash-free string of the right length, swap characters that are
    wrong for their position (letter slot got a digit, or vice versa).
    Returns the corrected string with dashes re-inserted, or '' on failure.
    """
    if len(s) == 7:
        # AA-DDD-AA  →  positions 0,1,5,6 = letters; 2,3,4 = digits
        letter_pos = {0, 1, 5, 6}
        digit_pos  = {2, 3, 4}
    elif len(s) == 6:
        # AAA-DDD  →  positions 0,1,2 = letters; 3,4,5 = digits
        letter_pos = {0, 1, 2}
        digit_pos  = {3, 4, 5}
    else:
        return ""

    fixed = []
    for i, ch in enumerate(s):
        if i in letter_pos:
            fixed.append(_DIGIT_TO_LETTER.get(ch, ch))
        else:
            fixed.append(_LETTER_TO_DIGIT.get(ch, ch))

    c = "".join(fixed)
    if len(c) == 7:
        return f"{c[0:2]}-{c[2:5]}-{c[5:7]}"
    return f"{c[0:3]}-{c[3:6]}"


def _normalize(raw: str) -> str:
    """Basic cleanup: uppercase, replace spaces with dashes, strip junk chars."""
    text = raw.upper().strip()
    text = re.sub(r"[\s_]+", "-", text)
    text = re.sub(r"[^A-Z0-9\-]", "", text)
    return text


def _best_candidate(raw: str) -> str:
    """
    Return the best plate candidate from a raw OCR string, or '' if none found.
    Tries three strategies in order:
      1. Direct normalisation (OCR was accurate, dashes present)
      2. Insert dashes by position (dashes were missed by OCR)
      3. Insert dashes + fix character-level OCR errors
    """
    normalised = _normalize(raw)

    # Strategy 1: already looks like a plate
    if _is_valid_plate(normalised):
        return normalised

    # Strip dashes and try inserting them + correcting chars
    stripped = normalised.replace("-", "")

    # Strategy 2: dashes missing but characters are otherwise correct
    if len(stripped) in (6, 7):
        if len(stripped) == 7:
            with_dashes = f"{stripped[0:2]}-{stripped[2:5]}-{stripped[5:7]}"
        else:
            with_dashes = f"{stripped[0:3]}-{stripped[3:6]}"
        if _is_valid_plate(with_dashes):
            return with_dashes

    # Strategy 3: dashes missing AND character errors present
    if len(stripped) in (6, 7):
        corrected = _fix_plate_chars(stripped)
        if corrected and _is_valid_plate(corrected):
            return corrected

    return ""


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
            candidate = _best_candidate(text)
            if candidate:
                logger.debug(f"  OCR hit: '{text}' → '{candidate}'  conf={confidence:.2f}")
                detections.append((candidate, confidence, bbox))

        return detections
