"""
Persistent storage for plate scan data.

JSON format (new):
  { "AA-123-BB": {"count": 5, "first_seen": "...", "last_seen": "..."}, ... }

Old format (plain int) is transparently migrated on first write.
"""
import json
import logging
import os
from datetime import datetime

from config import STORAGE_DIR, STORAGE_FILE

logger = logging.getLogger(__name__)


def _load() -> dict:
    if not os.path.exists(STORAGE_FILE):
        return {}
    try:
        with open(STORAGE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"Could not read storage ({exc}). Starting fresh.")
        return {}


def _save(data: dict) -> None:
    os.makedirs(STORAGE_DIR, exist_ok=True)
    with open(STORAGE_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def get_all() -> dict:
    """Return the full plates dictionary (raw, as stored)."""
    return _load()


def record_plate(plate: str) -> tuple:
    """
    Increment the scan counter for *plate* and persist it.
    Returns (new_count: int, last_seen: str).
    """
    data    = _load()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    entry = data.get(plate, {})
    if isinstance(entry, int):          # migrate old plain-int format
        entry = {"count": entry}

    entry["count"]      = entry.get("count", 0) + 1
    entry["last_seen"]  = now_str
    entry.setdefault("first_seen", now_str)

    data[plate] = entry
    _save(data)
    return entry["count"], now_str
