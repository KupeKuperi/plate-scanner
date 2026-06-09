"""
Persistent storage for plate scan counts.
Format: { "AA-111-BB": 5, "XYZ-123": 2, ... }
"""
import json
import logging
import os
from config import STORAGE_DIR, STORAGE_FILE

logger = logging.getLogger(__name__)


def _load() -> dict:
    if not os.path.exists(STORAGE_FILE):
        return {}
    try:
        with open(STORAGE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not read storage file ({exc}). Starting fresh.")
        return {}


def _save(data: dict) -> None:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(STORAGE_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def get_all() -> dict:
    """Return a copy of the full plates dictionary."""
    return _load()


def record_plate(plate: str) -> int:
    """
    Increment the scan counter for *plate* and persist it.
    Returns the new total count.
    """
    data = _load()
    new_count = data.get(plate, 0) + 1
    data[plate] = new_count
    _save(data)
    return new_count
