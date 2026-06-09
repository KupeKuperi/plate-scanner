"""
ANPR System — CustomTkinter dashboard.

Thread model:
  Main thread  — reads camera frames, runs motion detection, updates the UI
                 at ~33 FPS.  Never blocks on OCR.
  OCR thread   — sits idle until the main thread drops a frame into the
                 single-slot pending buffer, then calls EasyOCR and puts
                 results on a queue for the main thread to process.

CPU-saving measures:
  1. OCR rate cap  : at most OCR_TARGET_FPS calls per second (default 2).
  2. Motion gate   : OCR only fires when the scene changes above a pixel
                     threshold — idle camera = near-zero CPU.
  3. Frame resize  : EasyOCR receives frames capped at MAX_FRAME_WIDTH.
  4. Buffer=1      : CAP_PROP_BUFFERSIZE=1 prevents queued-frame lag.
"""

import logging
import queue
import sys
import threading
import time
from datetime import datetime

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image

from config import (
    COOLDOWN_SECONDS, MOTION_DETECTION, MOTION_THRESHOLD,
    OCR_TARGET_FPS, OVERLAY_TTL, VIDEO_SOURCE,
)
from detector import PlateDetector
from storage import get_all, record_plate

# ── Appearance ─────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Colour palette
_BG     = "#1a1a2e"
_PANEL  = "#16213e"
_BORDER = "#0f3460"
_TEXT   = "#e0e0e0"
_ACCENT = "#4a9eff"
_GREEN  = "#00b894"
_YELLOW = "#fdcb6e"
_RED    = "#e17055"
_GRAY   = "#888888"


# ── Main application ────────────────────────────────────────────────────────

class ANPRApp(ctk.CTk):

    def __init__(self) -> None:
        super().__init__()
        self.title("ANPR  ·  Plate Recognition System")
        self.geometry("1220x700")
        self.minsize(960, 580)
        self.configure(fg_color=_BG)

        # Runtime state
        self._cap:          cv2.VideoCapture | None = None
        self._stop:         threading.Event   = threading.Event()
        self._result_q:     queue.Queue       = queue.Queue()
        self._ocr_slot                        = [None]   # single pending frame
        self._ocr_lock:     threading.Lock    = threading.Lock()
        self._overlays:     list              = []  # (plate, bbox, skipped, expire_ts)
        self._cooldown:     dict              = {}  # plate → last_record_ts
        self._prev_gray:    np.ndarray | None = None
        self._last_ocr_ts:  float             = 0.0
        self._ocr_interval: float             = 1.0 / OCR_TARGET_FPS
        self._history:      list              = []  # sorted scan history
        self._row_frames:   list              = []  # table row widgets
        self._last_img                        = None  # keep CTkImage alive

        self._build_ui()
        self._load_history()
        self._open_camera()
        self._start_ocr_thread()
        self._ui_loop()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Top bar
        topbar = ctk.CTkFrame(self, fg_color=_PANEL, corner_radius=0, height=52)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        ctk.CTkLabel(
            topbar, text="  ◈  ANPR System",
            font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"),
            text_color=_ACCENT,
        ).pack(side="left", padx=20)

        self._status_lbl = ctk.CTkLabel(
            topbar, text="● Connecting…",
            font=ctk.CTkFont(size=12), text_color=_YELLOW,
        )
        self._status_lbl.pack(side="right", padx=20)

        # Body — two columns
        body = ctk.CTkFrame(self, fg_color=_BG)
        body.pack(fill="both", expand=True, padx=14, pady=12)
        body.columnconfigure(0, weight=58)
        body.columnconfigure(1, weight=42)
        body.rowconfigure(0, weight=1)

        self._build_camera_panel(body)
        self._build_sidebar(body)

    def _build_camera_panel(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(parent, fg_color=_PANEL, corner_radius=12)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 7))
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel, text="Live Camera Feed",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=_GRAY,
        ).grid(row=0, column=0, padx=14, pady=(12, 0), sticky="w")

        self._cam_lbl = ctk.CTkLabel(
            panel, text="", fg_color="#000000", corner_radius=8,
        )
        self._cam_lbl.grid(row=1, column=0, padx=12, pady=(6, 6), sticky="nsew")

        # Info strip below camera
        strip = ctk.CTkFrame(panel, fg_color=_BG, corner_radius=8, height=32)
        strip.grid(row=2, column=0, padx=12, pady=(0, 12), sticky="ew")
        strip.pack_propagate(False)

        self._motion_lbl = ctk.CTkLabel(
            strip, text="Motion: —",
            font=ctk.CTkFont(size=11), text_color=_GRAY,
        )
        self._motion_lbl.pack(side="left", padx=12)

        self._ocr_lbl = ctk.CTkLabel(
            strip, text="OCR: Idle",
            font=ctk.CTkFont(size=11), text_color=_GRAY,
        )
        self._ocr_lbl.pack(side="right", padx=12)

    def _build_sidebar(self, parent: ctk.CTkFrame) -> None:
        panel = ctk.CTkFrame(parent, fg_color=_PANEL, corner_radius=12)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.rowconfigure(2, weight=1)
        panel.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel, text="Recent Scans",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=_GRAY,
        ).grid(row=0, column=0, padx=14, pady=(12, 4), sticky="w")

        # Column header
        hdr = ctk.CTkFrame(panel, fg_color=_BORDER, corner_radius=6, height=28)
        hdr.grid(row=1, column=0, padx=12, pady=(0, 4), sticky="ew")
        hdr.pack_propagate(False)
        for label, width in [("Plate", 115), ("Visits", 55), ("Last Seen", 135)]:
            ctk.CTkLabel(
                hdr, text=label, width=width, anchor="w",
                font=ctk.CTkFont(size=10, weight="bold"), text_color=_GRAY,
            ).pack(side="left", padx=8)

        # Scrollable rows
        self._tbl = ctk.CTkScrollableFrame(
            panel, fg_color=_PANEL, corner_radius=0,
        )
        self._tbl.grid(row=2, column=0, padx=12, pady=(0, 6), sticky="nsew")

        # Stats footer
        foot = ctk.CTkFrame(panel, fg_color=_BG, corner_radius=8, height=32)
        foot.grid(row=3, column=0, padx=12, pady=(0, 12), sticky="ew")
        foot.pack_propagate(False)

        self._unique_lbl = ctk.CTkLabel(
            foot, text="Unique: 0",
            font=ctk.CTkFont(size=11), text_color=_GRAY,
        )
        self._unique_lbl.pack(side="left", padx=12)

        self._scans_lbl = ctk.CTkLabel(
            foot, text="Total scans: 0",
            font=ctk.CTkFont(size=11), text_color=_GRAY,
        )
        self._scans_lbl.pack(side="right", padx=12)

    # ── Camera & OCR ────────────────────────────────────────────────────────

    def _open_camera(self) -> None:
        self._cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not self._cap.isOpened():
            self._status_lbl.configure(text="● Camera Error", text_color=_RED)
            logger.error("Could not open video source.")
            return
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._status_lbl.configure(text="● Connected  /  Scanning", text_color=_GREEN)
        logger.info("Camera opened.")

    def _start_ocr_thread(self) -> None:
        detector = PlateDetector()
        threading.Thread(
            target=self._ocr_worker, args=(detector,),
            daemon=True, name="OCR-thread",
        ).start()

    def _ocr_worker(self, detector: PlateDetector) -> None:
        while not self._stop.is_set():
            with self._ocr_lock:
                frame = self._ocr_slot[0]
                self._ocr_slot[0] = None

            if frame is None:
                time.sleep(0.02)
                continue

            self._result_q.put(("ocr_busy", True))
            detections = detector.detect(frame)
            now = time.time()

            for plate, confidence, bbox in detections:
                elapsed = now - self._cooldown.get(plate, 0.0)
                skipped = elapsed < COOLDOWN_SECONDS

                if not skipped:
                    count, last_seen = record_plate(plate)
                    self._cooldown[plate] = now
                    logger.info(f"[SAVED]  {plate:<14}  conf={confidence:.2f}  scans: {count}")
                else:
                    remaining = int(COOLDOWN_SECONDS - elapsed)
                    logger.info(f"[SKIP]   {plate:<14}  conf={confidence:.2f}  cooldown: {remaining}s")
                    raw  = get_all().get(plate, {})
                    count     = raw.get("count", 1)     if isinstance(raw, dict) else int(raw)
                    last_seen = raw.get("last_seen", "") if isinstance(raw, dict) else ""

                self._result_q.put(("hit", plate, count, last_seen, bbox, skipped))

            self._result_q.put(("ocr_busy", False))

    # ── Main UI loop ─────────────────────────────────────────────────────────

    def _ui_loop(self) -> None:
        # Read + display one camera frame
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()
            if ret:
                self._handle_frame(frame)

        # Drain OCR result queue (always runs in main thread → safe to touch widgets)
        try:
            while True:
                self._handle_result(self._result_q.get_nowait())
        except queue.Empty:
            pass

        if not self._stop.is_set():
            self.after(30, self._ui_loop)   # ~33 display FPS

    def _handle_frame(self, frame: np.ndarray) -> None:
        now = time.time()

        # ── Motion detection gate ──────────────────────
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        has_motion = not MOTION_DETECTION   # if disabled, always True

        if MOTION_DETECTION:
            if self._prev_gray is not None:
                diff       = cv2.absdiff(self._prev_gray, gray)
                _, thr     = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                has_motion = cv2.countNonZero(thr) > MOTION_THRESHOLD
            else:
                has_motion = True   # first frame always processes

        self._prev_gray = gray
        self._motion_lbl.configure(
            text  = f"Motion: {'Active' if has_motion else 'Still'}",
            text_color = _YELLOW if has_motion else _GRAY,
        )

        # ── Submit to OCR thread ───────────────────────
        if has_motion and (now - self._last_ocr_ts) >= self._ocr_interval:
            with self._ocr_lock:
                self._ocr_slot[0] = frame.copy()
            self._last_ocr_ts = now

        # ── Draw active overlays ───────────────────────
        self._overlays = [(p, b, sk, e) for p, b, sk, e in self._overlays if e > now]
        for plate, bbox, skipped, _ in self._overlays:
            _draw_box(frame, plate, bbox, skipped)

        # ── Resize to label dimensions & display ───────
        lw = max(self._cam_lbl.winfo_width(),  200)
        lh = max(self._cam_lbl.winfo_height(), 150)
        h, w = frame.shape[:2]
        scale = min(lw / w, lh / h)
        dw, dh = int(w * scale), int(h * scale)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((dw, dh), Image.BILINEAR)
        img = ctk.CTkImage(light_image=pil, dark_image=pil, size=(dw, dh))
        self._cam_lbl.configure(image=img)
        self._last_img = img    # prevent premature GC

    def _handle_result(self, item: tuple) -> None:
        kind = item[0]

        if kind == "ocr_busy":
            busy = item[1]
            self._ocr_lbl.configure(
                text       = "OCR: Processing…" if busy else "OCR: Idle",
                text_color = _YELLOW if busy else _GRAY,
            )

        elif kind == "hit":
            _, plate, count, last_seen, bbox, skipped = item
            # Update overlay
            self._overlays = [(p, b, sk, e) for p, b, sk, e in self._overlays if p != plate]
            self._overlays.append((plate, bbox, skipped, time.time() + OVERLAY_TTL))
            # Update table
            self._upsert_history(plate, count, last_seen)
            self._rebuild_table()
            self._update_stats()

    # ── Scan history table ───────────────────────────────────────────────────

    def _load_history(self) -> None:
        """Populate the table with whatever is already in plates.json."""
        for plate, raw in get_all().items():
            if isinstance(raw, dict):
                count, last_seen = raw.get("count", 1), raw.get("last_seen", "")
            else:
                count, last_seen = int(raw), ""
            self._upsert_history(plate, count, last_seen)
        self._rebuild_table()
        self._update_stats()

    def _upsert_history(self, plate: str, count: int, last_seen: str) -> None:
        try:
            ts = datetime.strptime(last_seen, "%Y-%m-%d %H:%M:%S")
        except Exception:
            ts = datetime.min
        self._history = [h for h in self._history if h["plate"] != plate]
        self._history.append({"plate": plate, "count": count, "last_seen": last_seen, "ts": ts})
        self._history.sort(key=lambda h: h["ts"], reverse=True)

    def _rebuild_table(self) -> None:
        for f in self._row_frames:
            f.destroy()
        self._row_frames.clear()

        for entry in self._history[:60]:
            plate     = entry["plate"]
            count     = entry["count"]
            raw_ts    = entry.get("last_seen", "")
            time_str  = raw_ts.split(" ")[1] if " " in raw_ts else (raw_ts or "—")

            row = ctk.CTkFrame(self._tbl, fg_color=_BORDER, corner_radius=6)
            row.pack(fill="x", pady=2)
            self._row_frames.append(row)

            ctk.CTkLabel(
                row, text=plate, width=115, anchor="w",
                font=ctk.CTkFont(size=12, weight="bold"), text_color=_GREEN,
            ).pack(side="left", padx=8, pady=6)

            ctk.CTkLabel(
                row, text=f"{count}×", width=55, anchor="w",
                font=ctk.CTkFont(size=12), text_color=_TEXT,
            ).pack(side="left", padx=4)

            ctk.CTkLabel(
                row, text=time_str, width=135, anchor="w",
                font=ctk.CTkFont(size=11), text_color=_GRAY,
            ).pack(side="left", padx=4)

    def _update_stats(self) -> None:
        unique = len(self._history)
        total  = sum(h["count"] for h in self._history)
        self._unique_lbl.configure(text=f"Unique: {unique}")
        self._scans_lbl.configure(text=f"Total scans: {total}")

    # ── Cleanup ──────────────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._stop.set()
        if self._cap:
            self._cap.release()
        self.destroy()


# ── Drawing helper ───────────────────────────────────────────────────────────

def _draw_box(frame: np.ndarray, plate: str, bbox: list, skipped: bool) -> None:
    colour = (0, 200, 220) if skipped else (0, 220, 0)   # BGR
    pts = np.array([[int(p[0]), int(p[1])] for p in bbox], dtype=np.int32)
    cv2.polylines(frame, [pts], True, colour, 2)
    cv2.putText(
        frame, plate,
        (pts[0][0], max(pts[0][1] - 8, 18)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2, cv2.LINE_AA,
    )


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    app = ANPRApp()
    app.mainloop()


if __name__ == "__main__":
    main()
