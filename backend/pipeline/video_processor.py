"""
Video processor — main orchestrator for basketball analytics pipeline.

Runs as a FastAPI BackgroundTask. Processes video end-to-end:
  detector → tracker → court → pose → jersey_ocr → action →
  event_engine → stats_calculator → MongoDB / Redis / WebSocket

Threading model
  Frame grabber thread : cv2.VideoCapture → queue.Queue(maxsize=5)
  Pipeline processing  : ThreadPoolExecutor via asyncio.run_in_executor
  Async I/O            : MongoDB writes, Redis publish, WebSocket broadcast
"""

import asyncio
import bisect
import colorsys
import concurrent.futures
import csv
try:
    import orjson as _json_lib   # Rust-backed JSON: ~3-5× faster than stdlib
    def _json_dumps(obj, **kw) -> str:
        # OPT_NON_STR_KEYS: player_stats uses int track_id as dict keys;
        # without this orjson raises TypeError and silently kills Redis writes.
        return _json_lib.dumps(
            obj,
            option=_json_lib.OPT_NON_STR_KEYS,
            default=str,
        ).decode()
except ImportError:
    import json as _json_lib     # fallback: stdlib json
    def _json_dumps(obj, **kw) -> str:
        return _json_lib.dumps(obj, default=str, **kw)
import json  # still needed for json.loads in a few places
import logging
import os
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Pipeline imports (graceful — each module is optional) ─────────────────────

def _rel_or_abs(mod: str, attr: str):
    """Import attr from .mod (relative) or mod (absolute), or return None."""
    for attempt in (
        lambda: getattr(__import__(f"{__package__}.{mod}", fromlist=[attr]), attr)
        if __package__ else None,
        lambda: getattr(__import__(mod), attr),
    ):
        try:
            r = attempt()
            if r is not None:
                return r
        except Exception:
            pass
    return None


_create_detector   = _rel_or_abs("detector",          "create_detector")
_create_tracker    = _rel_or_abs("tracker",            "create_tracker")
_create_court      = _rel_or_abs("court",              "create_court_mapper")
_create_pose       = _rel_or_abs("pose",               "create_pose_estimator")
_create_jersey     = _rel_or_abs("jersey_ocr",         "create_jersey_ocr")
_create_action     = _rel_or_abs("action",             "create_action_classifier")
_create_events     = _rel_or_abs("event_engine",       "create_event_engine")
_create_stats      = _rel_or_abs("stats_calculator",   "create_stats_calculator")

# ── Device detection ──────────────────────────────────────────────────────────

def _detect_device() -> str:
    """Auto-detect CUDA GPU; fall back to CPU. Called once at VideoProcessor init."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("GPU detected: %s — using CUDA", name)
            return "cuda"
    except ImportError:
        pass
    logger.info("No CUDA GPU detected — using CPU")
    return "cpu"


def _fmt_video_ts(ts_ms: int) -> str:
    """Format video timestamp (ms) as MM:SS.mmm — used in scoring logs."""
    total_s, ms = divmod(int(ts_ms), 1000)
    m, s = divmod(total_s, 60)
    return f"{m:02d}:{s:02d}.{ms:03d}"


def _query_gpu_metrics() -> dict:
    """
    Return GPU utilisation % and VRAM usage from nvidia-smi.
    Cached at module level; refreshed every ~2 s to keep subprocess overhead low.
    Returns {"gpu": 0, "vramUsed": 0, "vramTotal": 0} on failure (CPU-only server).
    """
    now = time.monotonic()
    if now - _query_gpu_metrics._last_t < 2.0:
        return _query_gpu_metrics._cached

    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            timeout=1, stderr=subprocess.DEVNULL,
        ).decode().strip().splitlines()
        parts = [p.strip() for p in out[0].split(",")] if out else []
        result = {
            "gpu":       int(parts[0]) if len(parts) > 0 else 0,
            "vramUsed":  int(parts[1]) if len(parts) > 1 else 0,
            "vramTotal": int(parts[2]) if len(parts) > 2 else 0,
        }
    except Exception:
        result = {"gpu": 0, "vramUsed": 0, "vramTotal": 0}

    _query_gpu_metrics._cached = result
    _query_gpu_metrics._last_t = now
    return result

_query_gpu_metrics._cached = {"gpu": 0, "vramUsed": 0, "vramTotal": 0}
_query_gpu_metrics._last_t  = 0.0


def _log_system_metrics() -> None:
    """Log GPU util, VRAM, and CPU usage in one line. Called every FPS_LOG_INTERVAL frames."""
    gm = _query_gpu_metrics()

    cpu_pct: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    try:
        import psutil
        cpu_pct      = psutil.cpu_percent(interval=None)
        vm           = psutil.virtual_memory()
        ram_used_gb  = vm.used  / 1024 ** 3
        ram_total_gb = vm.total / 1024 ** 3
    except Exception:
        pass

    parts = []
    if gm["gpu"] > 0 or gm["vramTotal"] > 0:
        vram_str = (
            f"VRAM {gm['vramUsed']/1024:.1f}/{gm['vramTotal']/1024:.1f}GB"
            if gm["vramTotal"] > 0 else ""
        )
        parts.append(f"GPU {gm['gpu']}%  {vram_str}".strip())
    if cpu_pct > 0:
        parts.append(f"CPU {cpu_pct:.1f}%")
    if ram_total_gb > 0:
        parts.append(f"RAM {ram_used_gb:.1f}/{ram_total_gb:.1f}GB")

    if parts:
        logger.info("[SYSTEM] %s", "  |  ".join(parts))


def _fmt_wall_clock() -> str:
    """Current wall-clock time as HH:MM:SS — used in scoring logs for live context."""
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")


def _roster_entry(roster: dict, jersey_num: int, team: str = "") -> dict:
    """
    Look up a player's roster entry by jersey number + team.

    Tries "{jersey}_{team}" first (team-qualified, no collision when both teams
    share the same number), then falls back to plain "{jersey}" for backward
    compatibility with old roster format or numbers unique to one team.
    """
    if team:
        entry = roster.get(f"{jersey_num}_{team}")
        if isinstance(entry, dict):
            return entry
    entry = roster.get(str(jersey_num))
    return entry if isinstance(entry, dict) else {}


# ── Constants ─────────────────────────────────────────────────────────────────

MODELS_DIR           = os.getenv("MODELS_PATH",
                        os.path.join(os.path.dirname(__file__), "..", "models"))
OUTPUT_DIR           = os.getenv("OUTPUT_PATH",
                        os.path.join(os.path.dirname(__file__), "..", "output"))
RENDER_OUTPUT_VIDEO  = os.getenv("RENDER_OUTPUT_VIDEO", "1") == "1"
TARGET_FPS           = 25
STATS_UPDATE_INTERVAL = 30    # frames between WebSocket pushes
BALL_TRAJ_MAXLEN     = 60    # court positions kept for ball trajectory
FRAME_QUEUE_MAXSIZE  = 8     # slightly larger buffer for GPU burst
FRAME_BUFFER_SIZE    = 16    # temporal window for action classifier
QUARTER_DURATION_S   = 600   # FIBA 10-minute quarters
FPS_LOG_INTERVAL     = 100   # frames between FPS log lines
FPS_DEFAULT          = 30

# Model frequency — how often each heavy model runs (in processed-frame units)
# Stride=2: process 1-in-2 source frames (GPU recommended)
# Court keypoints: camera moves slowly → run every 30 processed frames
# Action: SlowFast needs temporal clip, result is stable → run every 20 frames
# OCR: jersey number is stable once confirmed → run every 30 frames
PROCESS_STRIDE        = int(os.getenv("PROCESS_STRIDE",   "2"))   # 2=every 2nd frame → 15fps from 30fps source
COURT_EVERY_N_FRAMES  = int(os.getenv("COURT_EVERY_N",    "5"))   # court keypoint YOLOv8-pose
POSE_EVERY_N_FRAMES   = int(os.getenv("POSE_EVERY_N",     "2"))   # player pose estimation stride
OCR_EVERY_N_FRAMES    = int(os.getenv("OCR_EVERY_N",     "15"))   # fallback default; overridden at runtime
ACTION_EVERY_N_FRAMES = int(os.getenv("ACTION_EVERY_N",  "10"))   # action classifier interval

# ── Action label normalisation ────────────────────────────────────────────────
# Backend action.py returns Title-case ("Shoot", "Dribble", "Rebound_attempt").
# Frontend ActionLabel type uses UPPER ("SHOOT", "DRIBBLE", "REBOUND").
# Normalise here once so every output path (WS + snapshot) is consistent.
_ACTION_WS_MAP: dict[str, str] = {
    "shoot": "SHOOT", "pass": "PASS", "dribble": "DRIBBLE",
    "catch": "CATCH", "steal": "STEAL", "block": "BLOCK",
    "rebound_attempt": "REBOUND", "rebound": "REBOUND",
    "jump": "JUMP", "stand": "STAND", "jump_action": "JUMP",
    "walk": "WALK", "run": "RUN",
}

def _norm_action(action_str: str) -> str:
    return _ACTION_WS_MAP.get((action_str or "stand").lower(), (action_str or "STAND").upper())


def _open_video_cap(video_path: str):
    """
    Open a video file preferring NVDEC hardware decode when available.

    NVDEC (h264_cuvid) runs on a dedicated hardware block on NVIDIA GPUs —
    completely separate from CUDA cores — and decodes H.264 at ~0.5ms/frame
    vs ~5ms/frame for CPU software decode (10× faster).

    Returns a cap object with a `._nvdec` attribute flag so _frame_grabber
    knows which code path to use.
    """
    try:
        import ffmpegcv
        import torch
        if torch.cuda.is_available():
            nvdec_cap = ffmpegcv.VideoCapture(
                video_path, codec="h264_cuvid", pix_fmt="bgr24"
            )
            if nvdec_cap.isOpened():
                # ffmpegcv uses .fps/.width/.height/.count instead of .get().
                # Patch in a cv2-compatible .get() so the rest of the pipeline
                # works without changes.
                _fps    = getattr(nvdec_cap, "fps",    30.0) or 30.0
                _width  = getattr(nvdec_cap, "width",  1920) or 1920
                _height = getattr(nvdec_cap, "height", 1080) or 1080
                _count  = getattr(nvdec_cap, "count",  0)   or 0
                def _get(prop_id, _f=_fps, _w=_width, _h=_height, _c=_count):
                    return {
                        cv2.CAP_PROP_FPS:          _f,
                        cv2.CAP_PROP_FRAME_WIDTH:  _w,
                        cv2.CAP_PROP_FRAME_HEIGHT: _h,
                        cv2.CAP_PROP_FRAME_COUNT:  _c,
                    }.get(prop_id, 0.0)
                nvdec_cap.get   = _get
                nvdec_cap._nvdec = True
                logger.info("NVDEC hardware decode enabled (ffmpegcv h264_cuvid)")
                return nvdec_cap
    except Exception as _e:
        logger.debug("NVDEC unavailable (%s) — using CPU decode", _e)

    cap = cv2.VideoCapture(video_path)
    cap._nvdec = False
    return cap


def _hex_to_hsv(hex_color: str) -> list:
    """Convert #RRGGBB to OpenCV HSV [H:0-180, S:0-255, V:0-255]."""
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
    hh, s, v = colorsys.rgb_to_hsv(r, g, b)
    return [hh * 180.0, s * 255.0, v * 255.0]


def _hsv_dist(a: list, b: list) -> float:
    """Hue-aware HSV distance — mirrors JerseyOCR._hsv_distance."""
    dh = abs(a[0] - b[0])
    dh = min(dh, 180.0 - dh)
    hue_weight = 2.0 * min(a[1], b[1]) / 255.0
    return float(hue_weight * dh + abs(a[1] - b[1]) + abs(a[2] - b[2]) * 0.25)


# ── Main class ────────────────────────────────────────────────────────────────

class VideoProcessor:
    """
    Orchestrates the full CV pipeline for one video / match.

    Usage (FastAPI):
        processor = create_video_processor(models_path, mongo_db, redis)
        background_tasks.add_task(
            processor.process_video, video_path, match_id, roster, ws_manager
        )
    """

    def __init__(
        self,
        models_path:   Optional[str] = None,
        mongo_db                     = None,
        redis_client                 = None,
        device:        Optional[str] = None,
    ):
        self._models_path = models_path or MODELS_DIR
        self._mongo_db    = mongo_db
        self._redis       = redis_client
        self._device      = device or _detect_device()
        self._target_fps  = TARGET_FPS

        # Pipeline components — populated by _init_pipeline()
        self._detector      = None
        self._tracker       = None
        self._court         = None
        self._pose          = None
        self._jersey        = None
        self._action        = None
        self._events_engine = None
        self._stats         = None

        # Runtime state
        self._status:              str   = "idle"
        self._frame_count:         int   = 0
        self._total_frames:        int   = 0
        self._fps_actual:          float = 0.0
        self._frames_dropped:      int   = 0
        self._current_quarter:     int   = 1
        self._quarter_start_frame: int   = 0
        self._start_time:          float = 0.0
        self._source_fps:          float = float(FPS_DEFAULT)
        self._ocr_interval:        int   = OCR_EVERY_N_FRAMES  # overridden after FPS known
        self._frame_width:         int   = 0
        self._frame_height:        int   = 0
        self._fps_skip:            int   = 1  # >1 when source fps > MAX_FPS_PROCESS
        self._roster:              dict  = {}

        # Court result cache — avoid running YOLOv8-pose every frame (camera is slow)
        self._last_court_result:  dict = {"is_calibrated": False}
        # Pose cache — reuse last result on skipped frames (POSE_EVERY_N_FRAMES)
        self._last_pose_result:   dict = {"poses": []}
        # Sticky courtPos cache: {track_id → [x,y]} — last known valid position per player
        self._last_court_pos:     dict = {}
        # Sticky team cache: {track_id → "A"|"B"} — persists even when OCR not running
        self._last_known_team:    dict = {}
        # Roster-anchored team color calibration: accumulate torso HSV samples from
        # roster-confirmed players, then call set_team_reference once we have enough.
        self._team_color_samples: dict = {"A": [], "B": []}
        self._team_color_ready:   set  = set()   # tracks which teams are calibrated
        self._kmeans_calibrated:  bool = False   # True after first calibrate_teams() call
        self._kmeans_disambiguated: bool = False # True after first roster-confirmed jersey
        self._jersey_hint_a: Optional[list] = None  # user-supplied Team A HSV hint
        self._jersey_hint_b: Optional[list] = None  # user-supplied Team B HSV hint

        # Per-frame bbox store — written to JSON at finalize for frontend time-lookup
        self._frame_store:        list = []   # [{ts, p:[{i,j,t,b}], bl}]
        self._video_path:         str  = ""   # set in process_video, used in _finalize
        self._frames_json_path:   str  = ""   # output path written in _finalize
        self._frames_jsonl_path:  str  = ""   # incremental JSONL for progressive streaming
        self._jsonl_lock = threading.Lock()   # guards concurrent append vs. finalize rewrite
        self._output_video_path:  str  = ""   # rendered video with overlay
        self._output_csv_path:    str  = ""   # player stats CSV

        # Threading
        self._frame_queue  = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        # Parallel detection: _detection_worker runs GPU detect for frame N while
        # the main loop runs CPU ByteTrack for frame N-1. maxsize=3 caps the
        # pre-detected frame buffer so GPU never races too far ahead of CPU.
        self._detect_queue = queue.Queue(maxsize=3)
        self._stop_signal  = threading.Event()
        self._executor     = concurrent.futures.ThreadPoolExecutor(max_workers=2,
                                thread_name_prefix="pipeline")

        # Data buffers
        self._frame_buffer:    deque = deque(maxlen=FRAME_BUFFER_SIZE)
        self._ball_trajectory: deque = deque(maxlen=BALL_TRAJ_MAXLEN)

        # Speed cache: track_id → avg_speed_kmh, updated every WS cycle (~5 frames)
        # Used by _snapshot_frame so the per-frame JSON has speed without calling get_live_stats() every frame.
        self._speed_cache:     dict  = {}   # {track_id: float}

        # Score cache: [pts_A, pts_B] from StatsCalculator, updated every WS cycle.
        # _snapshot_frame uses this so "sc" in replay frameData matches the STATS panel.
        # EventEngine.score can diverge (skips events with team="") — StatsCalculator
        # has roster fallback so it always matches the displayed pts.
        self._pts_cache:       list  = [0, 0]
        # Score at the START of the current quarter — used to derive per-quarter
        # running score: q_sc = _pts_cache - _q_pts_start.
        self._q_pts_start:     list  = [0, 0]
        # Scoring events log: (ts_ms, track_id, pts, quarter) — collected during
        # processing, used at _finalize() to retroactively patch sc/q_sc using
        # final team assignments (team may be unknown at event time due to OCR lag).
        self._scoring_events:  list  = []

        # Sticky action cache: last non-empty action per player so that the
        # 19 frames between ACTION_EVERY_N_FRAMES snapshots still show a label.
        self._last_action_cache: dict = {}  # {track_id: str}

        # Tracks which track_ids have already had a jersey_confirmed WS message sent.
        # Reset per quarter so late-confirming players always get a notification.
        self._confirmed_jersey_sent: set = set()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def _init_pipeline(self, models_path: str) -> None:
        """
        Initialise and load all pipeline components.
        Missing model files → warning + None; processing continues without that step.
        """
        mp  = models_path
        dev = self._device

        def _try(label, fn, *args, **kwargs):
            try:
                obj = fn(*args, **kwargs)
                logger.info("%-20s loaded", label)
                return obj
            except Exception as e:
                logger.warning("%-20s SKIP: %s", label, e)
                return None

        if _create_detector:
            det = _try("Detector", _create_detector, mp)
            if det:
                try:
                    det.load_model()
                    self._detector = det
                except Exception as e:
                    logger.warning("Detector.load_model: %s", e)

        if _create_tracker:
            self._tracker = _try("Tracker", _create_tracker,
                                 device=dev, frame_rate=self._target_fps)

        if _create_court:
            court = _try("CourtMapper", _create_court, mp, device=dev)
            if court:
                # create_court_mapper already calls load_model() internally.
                # Calling it again here caused a silent double-load failure that left
                # self._court = None, disabling 2D court overlay entirely.
                self._court = court

        if _create_pose:
            pose = _try("PoseEstimator", _create_pose, mp, device=dev)
            if pose:
                try:
                    pose.load_model()
                    self._pose = pose
                except Exception as e:
                    logger.warning("PoseEstimator.load_model: %s", e)

        if _create_jersey:
            self._jersey = _try("JerseyOCR", _create_jersey, mp, device=dev)

        if _create_action:
            self._action = _try("ActionClassifier", _create_action, mp, device=dev)

        if _create_events:
            self._events_engine = _try("EventEngine", _create_events)

        if _create_stats:
            self._stats = _try("StatsCalculator", _create_stats, self._target_fps)
        else:
            # Always need stats — try fallback direct instantiation
            try:
                from .stats_calculator import StatsCalculator
                self._stats = StatsCalculator(fps=self._target_fps)
            except ImportError:
                try:
                    from stats_calculator import StatsCalculator
                    self._stats = StatsCalculator(fps=self._target_fps)
                except ImportError:
                    logger.warning("StatsCalculator unavailable")

        # Warmup: one dummy inference per model to trigger CUDA JIT compilation
        # before the first real video frame arrives. Adds ~2-5 s at startup
        # (background task — invisible to the user) but eliminates the JIT lag
        # spike on frames 1-3 that would otherwise delay K-Means calibration.
        if self._detector:
            self._detector.warmup()
        if self._jersey:
            self._jersey.warmup()
        if self._pose:
            self._pose.warmup()
        if self._court:
            self._court.warmup()
        if self._action:
            self._action.warmup()

        logger.info("Pipeline ready (device=%s)", dev)

    # ── Public async entry point ──────────────────────────────────────────────

    async def process_video(
        self,
        video_path:      str,
        match_id:        str,
        roster:          Optional[dict] = None,
        ws_manager                      = None,
        start_quarter:   int            = 1,
        lock_quarter:    bool           = False,
        jersey_color_a:  str            = "",
        jersey_color_b:  str            = "",
    ) -> None:
        """
        Process video end-to-end as an async coroutine (FastAPI BackgroundTask).
        Emits WebSocket updates every STATS_UPDATE_INTERVAL frames.

        start_quarter: 1–4.  Use >1 when uploading a single-quarter clip so that
        stats, events, and the scoreboard are attributed to the correct quarter.
        lock_quarter: True when the user explicitly selected a quarter at upload —
        disables time-based auto-advance so the whole video stays in that quarter.
        """
        self._roster          = roster or {}
        self._lock_quarter    = lock_quarter
        self._status          = "processing"
        self._start_time      = time.perf_counter()
        self._stop_signal.clear()
        self._video_path      = str(video_path)
        self._frame_store     = []
        self._pts_cache           = [0, 0]
        self._q_pts_start         = [0, 0]
        self._scoring_events      = []
        self._kmeans_calibrated   = False
        self._kmeans_disambiguated = False
        self._start_quarter   = max(1, min(4, start_quarter))   # kept for _finalize merge
        self._current_quarter = self._start_quarter

        # Progressive streaming: one JSON line per processed frame, read by /stream endpoint.
        # Overwritten at finalize with retroactively patched sc/q_sc.
        self._frames_jsonl_path = (
            os.path.splitext(str(video_path))[0]
            + f"_q{self._start_quarter}_frames.jsonl"
        )
        try:
            open(self._frames_jsonl_path, 'w').close()   # truncate / create fresh
        except Exception as _e:
            logger.warning("Cannot create JSONL stream file: %s", _e)
            self._frames_jsonl_path = ""

        if self._stats is None:
            self._init_pipeline(self._models_path)

        # Store user-supplied jersey colors as K-Means orientation hints.
        # K-Means ALWAYS runs to detect the ACTUAL jersey colors from the video pixels.
        # User hints are used only AFTER K-Means to determine which detected cluster
        # corresponds to Team A vs Team B (semantic labeling, not pixel reference).
        # This handles the common case where users enter brand colors (#00BCD4 teal)
        # instead of actual jersey colors (#FFFFFF white) — K-Means detects white/red
        # correctly from video; hints guide the A↔B orientation decision.
        # OCR-based confirmation still runs afterward to catch any hint errors.
        if jersey_color_a:
            try:
                self._jersey_hint_a = _hex_to_hsv(jersey_color_a)
            except Exception as _e:
                logger.warning("Failed to parse jersey_color_a '%s': %s", jersey_color_a, _e)
        if jersey_color_b:
            try:
                self._jersey_hint_b = _hex_to_hsv(jersey_color_b)
            except Exception as _e:
                logger.warning("Failed to parse jersey_color_b '%s': %s", jersey_color_b, _e)
        if jersey_color_a or jersey_color_b:
            logger.info(
                "Jersey color hints stored (A=%s  B=%s) — "
                "K-Means runs from video pixels, hints guide cluster orientation",
                jersey_color_a or "none", jersey_color_b or "none",
            )

        # Try NVDEC hardware decode first (GPU dedicated engine, ~10× faster than CPU).
        # Falls back to cv2 software decode when ffmpegcv is not installed or GPU unavailable.
        cap = _open_video_cap(str(video_path))
        if not cap.isOpened():
            self._status = "error"
            logger.error("Cannot open video: %s", video_path)
            return

        raw_fps        = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
        self._frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1920
        self._frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

        # Cap source FPS at MAX_FPS_PROCESS (default 25) so high-fps cameras
        # (60fps, 120fps) don't double/quadruple processing load.
        # cap.set(CAP_PROP_FPS) does nothing on file sources — instead we skip
        # extra frames in _frame_grabber by increasing the effective stride.
        _MAX_SRC_FPS = float(os.getenv("MAX_FPS_PROCESS", "30"))
        if raw_fps > _MAX_SRC_FPS + 1:
            self._fps_skip = max(1, round(raw_fps / _MAX_SRC_FPS))
            self._source_fps = raw_fps / self._fps_skip
            logger.info(
                "Source %.0ffps > cap %.0ffps → fps_skip=%d, effective src_fps=%.1f",
                raw_fps, _MAX_SRC_FPS, self._fps_skip, self._source_fps,
            )
        else:
            self._fps_skip   = 1
            self._source_fps = raw_fps

        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // self._fps_skip

        # Re-initialise ByteTrack with the actual processed fps so its Kalman filter
        # predicts position correctly. Processed fps = source_fps / PROCESS_STRIDE.
        # Without this, ByteTrack is calibrated for TARGET_FPS=25 but receives frames
        # at ~7.5fps → velocity model overshoots → IoU matching fails on fast players.
        if self._tracker:
            _eff_fps = max(1, round(self._source_fps / PROCESS_STRIDE))
            self._tracker.frame_rate = _eff_fps
            if self._frame_width and self._frame_height:
                self._tracker.initialize((self._frame_height, self._frame_width, 3))
                logger.info("Tracker re-initialised at effective fps=%d", _eff_fps)

        # FPS-aware OCR interval: target per-player OCR every ~8 source frames (~0.27s at 30fps).
        # jersey.process() is called every _ocr_interval processed frames; each player
        # fires OCR on every process() call (OCR_SAMPLE_EVERY=1).
        # Divisor 8 gives _raw=3 at 30fps; aligned to PROCESS_STRIDE=2 → _ocr_interval=2
        # → OCR every 2×2=4 source frames = 0.13s at 30fps.
        # VOTE_THRESHOLD=2 → confirmation in 2×0.13s ≈ 0.27s (fast-moving players).
        # Aligning to PROCESS_STRIDE prevents the lcm-doubling bug where an odd
        # interval combined with PROCESS_STRIDE makes the modulo never fire.
        _raw = max(PROCESS_STRIDE, int(self._source_fps / 8))
        self._ocr_interval = (_raw // PROCESS_STRIDE) * PROCESS_STRIDE or PROCESS_STRIDE

        logger.info("Video: %s  frames=%d  src_fps=%.1f  ocr_interval=%d",
                    Path(video_path).name, self._total_frames, self._source_fps,
                    self._ocr_interval)

        grab_thread = threading.Thread(
            target=self._frame_grabber, args=(cap,),
            daemon=True, name="frame-grabber",
        )
        grab_thread.start()

        # Detection thread: runs GPU inference for frame N while the main loop
        # runs CPU ByteTrack for frame N-1. frame_id/proc_id/ts_ms are computed
        # inside the worker and arrive pre-packaged via _detect_queue.
        detect_thread = threading.Thread(
            target=self._detection_worker,
            daemon=True, name="detection",
        )
        detect_thread.start()

        loop     = asyncio.get_event_loop()
        frame_id = 0   # updated from detect_queue each iteration
        proc_id  = 0

        try:
            while not self._stop_signal.is_set():
                # Non-blocking poll so the event loop stays responsive
                try:
                    item = self._detect_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.002)
                    continue

                if item is None:   # sentinel — video exhausted
                    break

                frame, frame_id, proc_id, ts_ms, detections = item

                # Heavy sync pipeline (ByteTrack + OCR + stats) runs off the
                # event loop. Detection is already done — GPU was running while
                # the previous frame's CPU work was executing.
                frame_data = await loop.run_in_executor(
                    self._executor, self._process_frame,
                    frame, proc_id, ts_ms, detections
                )

                # Store compact bbox snapshot — used by frontend for time-based replay lookup
                self._frame_store.append(self._snapshot_frame(frame_data, ts_ms))
                self._append_frame_to_jsonl(self._frame_store[-1])

                # Persist events
                scoring_types = {"MADE_FG", "MADE_3", "SHOT_MADE", "MADE_FT", "FT_MADE"}
                has_score_event = False
                for evt in frame_data.get("events", []):
                    if evt:
                        await self._save_event_mongo(evt, match_id)
                        etype = evt.get("type", "").upper()
                        if etype in scoring_types:
                            has_score_event = True
                            # Log for retroactive sc patch at finalize (team may be "" now)
                            pts = 1 if etype in ("MADE_FT", "FT_MADE") else (3 if etype == "MADE_3" else 2)
                            self._scoring_events.append((
                                ts_ms,
                                evt.get("track_id"),
                                pts,
                                self._current_quarter,
                            ))

                # WebSocket update: every 5 processed frames OR immediately on scoring
                do_broadcast = (proc_id % 5 == 0) or has_score_event
                if do_broadcast and self._stats:
                    stats = self._stats.get_live_stats()
                    # Refresh speed + score caches so _snapshot_frame embeds correct values
                    for tid, pstat in stats.get("player_stats", {}).items():
                        spd = pstat.get("mpi", {}).get("avg_speed_kmh", 0.0)
                        if spd > 0:
                            self._speed_cache[tid] = round(spd, 1)
                    ts_now = stats.get("team_stats", {})
                    self._pts_cache = [
                        ts_now.get("A", {}).get("pts", 0),
                        ts_now.get("B", {}).get("pts", 0),
                    ]
                    msg = self._build_ws_message(frame_data, stats)
                    if has_score_event:
                        score_a = stats.get("team_stats", {}).get("A", {}).get("pts", 0)
                        score_b = stats.get("team_stats", {}).get("B", {}).get("pts", 0)
                        logger.info(
                            "SCORE_UPDATE  A=%d  B=%d  video=%s  wall=%s",
                            score_a, score_b,
                            _fmt_video_ts(ts_ms),
                            _fmt_wall_clock(),
                        )
                    await self._push_redis(match_id, msg)
                    await self._set_redis_stats(match_id, stats)
                    if ws_manager:
                        try:
                            await ws_manager.broadcast(match_id, msg)
                        except Exception as e:
                            logger.debug("WS broadcast error: %s", e)

                        # ── jersey_confirmed: notify frontend to patch past events ──
                        if self._tracker:
                            for _tid, _jn in self._tracker.track_jersey_map.items():
                                if _tid in self._confirmed_jersey_sent:
                                    continue
                                self._confirmed_jersey_sent.add(_tid)
                                _known_team = self._last_known_team.get(_tid, "")
                                _entry = _roster_entry(self._roster, _jn, _known_team)
                                _name  = _entry.get("name", f"#{_jn}")
                                _team  = _known_team or _entry.get("team", "")
                                _conf_msg = {
                                    "type":        "jersey_confirmed",
                                    "trackId":     _tid,
                                    "jerseyNumber": _jn,
                                    "playerName":  _name,
                                    "team":        _team,
                                }
                                logger.info(
                                    "jersey_confirmed  tid=%d  jersey=%s  name=%s  team=%s",
                                    _tid, _jn, _name, _team,
                                )
                                try:
                                    await ws_manager.broadcast(match_id, _conf_msg)
                                except Exception as e:
                                    logger.debug("WS jersey_confirmed error: %s", e)

                # frame_id / proc_id come from _detect_queue — already correct
                self._frame_count = frame_id
                elapsed = time.perf_counter() - self._start_time
                self._fps_actual = frame_id / elapsed if elapsed > 0 else 0.0

                if proc_id % (max(1, FPS_LOG_INTERVAL // PROCESS_STRIDE)) == 0:
                    logger.info(
                        "Frame %d/%d (proc %d)  FPS=%.1f  dropped=%d  Q%d  stride=%d",
                        frame_id, self._total_frames, proc_id,
                        self._fps_actual, self._frames_dropped,
                        self._current_quarter, PROCESS_STRIDE,
                    )
                    _log_system_metrics()

                if self._detect_quarter_change(frame_data):
                    self._handle_quarter_change()

                # Offline analysis — process at GPU max speed, no artificial throttle

        except asyncio.CancelledError:
            self._status = "cancelled"
            raise
        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            self._status = "error"
        else:
            self._status = "done"
        finally:
            self._stop_signal.set()
            grab_thread.join(timeout=5.0)
            detect_thread.join(timeout=5.0)
            await self._finalize(match_id)
            logger.info("Processing complete: match=%s  frames=%d  status=%s",
                        match_id, frame_id, self._status)

    # ── Frame grabber (background thread) ────────────────────────────────────

    def _frame_grabber(self, cap: cv2.VideoCapture) -> None:
        """
        Read frames from the video source.

        Two paths:
        A) ffmpegcv NVDEC (preferred on CUDA) — hardware H.264 decode via GPU
           NVDEC engine. Decodes every frame at ~0.5ms each vs ~5ms CPU.
           No grab() optimisation needed; hardware decode is already faster
           than software grab+skip.
        B) cv2.VideoCapture (fallback) — uses cap.grab() (no H.264 decode,
           ~0.05ms) for stride-skipped frames to avoid full software decode
           on every source frame.
        """
        nvdec_cap = getattr(cap, "_nvdec", False)

        if nvdec_cap:
            # ── Path A: ffmpegcv NVDEC ───────────────────────────────────────
            # Apply fps_skip × PROCESS_STRIDE to match cv2 path throughput.
            # cv2 path grabs (fps_skip×PROCESS_STRIDE - 1) then reads 1, so
            # it processes 1 out of every fps_skip×PROCESS_STRIDE source frames.
            # NVDEC reads all frames; we queue 1 out of every _nvdec_stride frames.
            _nvdec_stride = self._fps_skip * PROCESS_STRIDE
            frame_idx = 0
            try:
                while not self._stop_signal.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % _nvdec_stride == (_nvdec_stride - 1):
                        try:
                            self._frame_queue.put(frame, block=True, timeout=10.0)
                        except queue.Full:
                            self._frames_dropped += 1
                    frame_idx += 1
            except Exception as e:
                logger.warning("NVDEC frame grabber error: %s", e)
            finally:
                cap.release()
                try:
                    self._frame_queue.put(None, timeout=2.0)
                except queue.Full:
                    pass
        else:
            # ── Path B: cv2 software decode with grab() optimisation ─────────
            _grabs_before_read = self._fps_skip * PROCESS_STRIDE - 1
            try:
                while not self._stop_signal.is_set():
                    for _ in range(_grabs_before_read):
                        if not cap.grab():
                            self._frame_queue.put(None, block=True, timeout=2.0)
                            return

                    ret, frame = cap.read()
                    if not ret:
                        break
                    try:
                        self._frame_queue.put(frame, block=True, timeout=10.0)
                    except queue.Full:
                        self._frames_dropped += 1
            except Exception as e:
                logger.warning("Frame grabber error: %s", e)
            finally:
                cap.release()
                try:
                    self._frame_queue.put(None, timeout=2.0)
                except queue.Full:
                    pass

    # ── GPU detection thread ──────────────────────────────────────────────────

    def _detection_worker(self) -> None:
        """
        Dedicated GPU detection thread.

        Reads raw frames from _frame_queue, runs YOLO inference, then puts
        (frame, frame_id, proc_id, ts_ms, detections) into _detect_queue.

        Running detection here lets GPU work on frame N while the main loop
        runs CPU ByteTrack for frame N-1 — eliminating the sequential GPU→CPU
        bubble that currently stalls the pipeline between every frame.
        """
        frame_id = 0
        proc_id  = 0
        try:
            while not self._stop_signal.is_set():
                try:
                    frame = self._frame_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                if frame is None:   # sentinel from grabber — video exhausted
                    # frame_id now equals the actual total source frames decoded.
                    # NVDEC container metadata often under-reports; update so that
                    # get_progress() reports 100% correctly when processing finishes.
                    if frame_id > self._total_frames:
                        self._total_frames = frame_id
                    try:
                        self._detect_queue.put(None, block=True, timeout=5.0)
                    except queue.Full:
                        pass
                    return

                ts_ms = int(frame_id / max(self._source_fps, 1.0) * 1000)

                detections: dict = {}
                if self._detector:
                    try:
                        detections = self._detector.detect(frame)
                    except Exception as e:
                        logger.debug("Detection worker frame %d: %s", frame_id, e)

                try:
                    self._detect_queue.put(
                        (frame, frame_id, proc_id, ts_ms, detections),
                        block=True, timeout=5.0,
                    )
                except queue.Full:
                    self._frames_dropped += 1
                    logger.debug("detect_queue full — dropped frame %d", frame_id)

                frame_id += PROCESS_STRIDE
                proc_id  += 1

        except Exception as e:
            logger.warning("Detection worker crashed: %s", e)
        finally:
            try:
                self._detect_queue.put(None, timeout=2.0)
            except queue.Full:
                pass

    # ── Per-frame pipeline ────────────────────────────────────────────────────

    def _process_frame(
        self,
        frame:                 np.ndarray,
        frame_id:              int,
        timestamp_ms:          int,
        precomputed_detections: Optional[dict] = None,
    ) -> dict:
        """
        Run all pipeline steps synchronously (called via run_in_executor).
        Each step is wrapped in try/except — partial data is returned on error.
        """
        if frame is None:
            return {}

        frame_data: dict = {
            "frame_id":    frame_id,
            "timestamp_ms": timestamp_ms,
            "quarter":     self._current_quarter,
            "detections":  {},
            "tracking":    {"tracked_players": [], "tracked_ball": None,
                            "tracked_referees": []},
            "court":       {"is_calibrated": False},
            "pose":        {"poses": []},
            "jersey":      {"jersey_results": []},
            "actions":     {"actions": []},
            "events":      [],
        }

        detections      = {}
        tracked_players = []
        ball_info       = None

        _t0 = time.perf_counter()

        # ── 1. Detection ──────────────────────────────────────────────────
        # Normally pre-computed by _detection_worker (GPU ran in parallel while
        # the previous frame's ByteTrack ran on CPU). Fall back to inline if
        # the worker is unavailable (e.g. detector not loaded).
        if precomputed_detections is not None:
            detections = precomputed_detections
        elif self._detector:
            try:
                detections = self._detector.detect(frame)
            except Exception as e:
                logger.debug("Detector frame %d: %s", frame_id, e)
        frame_data["detections"] = detections
        _t1 = time.perf_counter()

        # ── 2. Tracking ───────────────────────────────────────────────────
        if self._tracker:
            try:
                tracking = self._tracker.update(frame, detections)
                frame_data["tracking"] = tracking
                tracked_players = tracking.get("tracked_players", [])
                ball_info       = tracking.get("tracked_ball")
                # When ByteTrack re-assigns a track_id, inherit team from old ID
                # so bbox colour doesn't flicker during the 3-vote accumulation window.
                if self._jersey and self._tracker._newly_inherited:
                    for _new_tid, _old_tid in self._tracker._newly_inherited.items():
                        if _old_tid in self._last_known_team:
                            self._last_known_team[_new_tid] = self._last_known_team[_old_tid]
                        self._jersey.inherit_team(_new_tid, _old_tid)
            except Exception as e:
                logger.debug("Tracker frame %d: %s", frame_id, e)
        _t2 = time.perf_counter()

        # ── 3. Court mapping ──────────────────────────────────────────────
        # (keypoint-pose YOLO, every COURT_EVERY_N_FRAMES; cached H on skipped)
        # YOLOv8-pose keypoint inference runs only every COURT_EVERY_N_FRAMES;
        # pixel_to_court() uses the cached H matrix on skipped frames.
        if self._court:
            try:
                if frame_id % COURT_EVERY_N_FRAMES == 0:
                    court = self._court.process_frame(frame)
                    self._last_court_result = court
                else:
                    court = self._last_court_result
                frame_data["court"] = court
                court_ran = (frame_id % COURT_EVERY_N_FRAMES == 0)
                if court.get("is_calibrated"):
                    for player in tracked_players:
                        pc = player.get("center")
                        if pc:
                            player["court_pos"] = self._court.pixel_to_court(pc)
                    if ball_info:
                        bc = ball_info.get("center")
                        if bc:
                            ball_info["court_pos"] = self._court.pixel_to_court(bc)
                elif court_ran:
                    # Court detection ran this frame but failed — clear stale sticky
                    # positions so players don't appear frozen at old coordinates.
                    for player in tracked_players:
                        self._last_court_pos.pop(player["track_id"], None)
            except Exception as e:
                logger.warning("Court frame %d: %s", frame_id, e, exc_info=True)
        _t3 = time.perf_counter()

        # ── 4. Pose estimation ────────────────────────────────────────────
        if self._pose and tracked_players and frame_id % (POSE_EVERY_N_FRAMES * PROCESS_STRIDE) == 0:
            try:
                pose = self._pose.estimate(frame, tracked_players)
                self._last_pose_result = pose
                frame_data["pose"] = pose
            except Exception as e:
                logger.debug("Pose frame %d: %s", frame_id, e)
        elif self._pose and tracked_players:
            frame_data["pose"] = getattr(self, "_last_pose_result", {"poses": []})

        if self._pose and tracked_players and frame_data.get("pose"):
            pose = frame_data["pose"]
            try:
                # Refine court_pos using ankle keypoints (indices 15/16 = COCO).
                # Ankles are at floor level → homography projection is accurate.
                # Bbox center is at mid-body height → 0.3–0.8 m displacement in
                # court coords, which can flip 2PT/3PT near the arc.
                if self._court and frame_data["court"].get("is_calibrated"):
                    for pe in pose.get("poses", []):
                        kps = pe.get("keypoints", [])
                        foot_px = []
                        for idx in (15, 16):   # left_ankle, right_ankle
                            if idx < len(kps) and kps[idx].get("confidence", 0.0) > 0.35:
                                foot_px.append([kps[idx]["x"], kps[idx]["y"]])
                        if not foot_px:
                            continue
                        fx = sum(p[0] for p in foot_px) / len(foot_px)
                        fy = sum(p[1] for p in foot_px) / len(foot_px)
                        cp = self._court.pixel_to_court([fx, fy])
                        if cp is not None:
                            for pl in tracked_players:
                                if pl["track_id"] == pe["track_id"]:
                                    pl["court_pos"] = cp
                                    break
            except Exception as e:
                logger.debug("Pose frame %d: %s", frame_id, e)
        _t4 = time.perf_counter()

        # ── 5. Jersey OCR — run every self._ocr_interval (FPS-aware, stride-aligned) ──
        if self._jersey and tracked_players and frame_id % self._ocr_interval == 0:
            try:
                jersey = self._jersey.process(
                    frame, tracked_players, self._tracker,
                    roster=self._roster or None,
                )
                frame_data["jersey"] = jersey
            except Exception as e:
                logger.error("Jersey OCR frame %d: %s", frame_id, e)
        _t5 = time.perf_counter()

        # ── Annotate tracked_players with team ───────────────────────────────────
        # Priority: (1) roster lookup via confirmed jersey number, (2) last confirmed
        # sticky, (3) jersey color classification from OCR (works without a number).
        jersey_color_map: dict = {}
        for jr in frame_data.get("jersey", {}).get("jersey_results", []):
            t = jr.get("team")
            if t and jr.get("track_id") is not None:
                jersey_color_map[jr["track_id"]] = t

        # Early K-Means calibration: run as soon as ≥5 players visible, no jersey
        # confirmation needed. Clusters are arbitrary (A/B may be swapped) — a single
        # confirmed jersey from the roster disambiguates which cluster is which team.
        if (self._jersey and not self._kmeans_calibrated
                and len(tracked_players) >= 5):
            colors = self._jersey.calibrate_teams(frame, tracked_players)
            # Only mark done when BOTH clusters were valid (≥2 members each).
            # calibrate_teams() returns unchanged dict when it skips due to
            # unbalanced clusters (referee/official in frame) — retry next frame.
            if "A" in colors and "B" in colors:
                self._kmeans_calibrated = True
                logger.info("Early K-Means calibration done (%d players visible)", len(tracked_players))

                # If user supplied both color hints, orient clusters immediately so
                # every player gets a sensible team guess from the very first frame —
                # no need to wait for OCR to confirm a jersey number.
                # Assignment: pick the permutation that minimises total HSV distance
                # between user hints and K-Means centers (Hungarian-style, 2×2 case).
                # OCR-based verification (line below) still runs afterward and can swap
                # again if the hints turn out to be brand colors rather than jersey colors.
                if not self._kmeans_disambiguated:
                    c_a, c_b = colors["A"], colors["B"]
                    hint_a, hint_b = self._jersey_hint_a, self._jersey_hint_b

                    if hint_a is not None and hint_b is not None:
                        # Both hints available: use Hungarian 2×2 — pick the
                        # permutation with minimum total HSV distance to K-Means centers.
                        d_keep = _hsv_dist(hint_a, c_a) + _hsv_dist(hint_b, c_b)
                        d_swap = _hsv_dist(hint_a, c_b) + _hsv_dist(hint_b, c_a)
                        need_swap = d_swap < d_keep
                        _d_winner, _d_loser = (d_swap, d_keep) if need_swap else (d_keep, d_swap)
                        confident = _d_winner > 0 and _d_loser / _d_winner >= 2.0
                        logger.info(
                            "K-Means hint (both): keep=%.1f  swap=%.1f  %s  confident=%s",
                            d_keep, d_swap,
                            "→ SWAP" if need_swap else "→ KEEP",
                            confident,
                        )
                    elif hint_a is not None:
                        # Only Team A hint: the cluster closer to hint_a becomes Team A.
                        da = _hsv_dist(hint_a, c_a)
                        db = _hsv_dist(hint_a, c_b)
                        need_swap = db < da          # hint_a matches cluster B → swap
                        _d_winner, _d_loser = (db, da) if need_swap else (da, db)
                        confident = _d_winner > 0 and _d_loser / _d_winner >= 2.0
                        logger.info(
                            "K-Means hint (A only): da=%.1f  db=%.1f  %s  confident=%s",
                            da, db,
                            "→ SWAP" if need_swap else "→ KEEP",
                            confident,
                        )
                    elif hint_b is not None:
                        # Only Team B hint: the cluster closer to hint_b becomes Team B.
                        da = _hsv_dist(hint_b, c_a)
                        db = _hsv_dist(hint_b, c_b)
                        need_swap = da < db          # hint_b matches cluster A → swap
                        _d_winner, _d_loser = (da, db) if need_swap else (db, da)
                        confident = _d_winner > 0 and _d_loser / _d_winner >= 2.0
                        logger.info(
                            "K-Means hint (B only): da=%.1f  db=%.1f  %s  confident=%s",
                            da, db,
                            "→ SWAP" if need_swap else "→ KEEP",
                            confident,
                        )
                    else:
                        need_swap = False
                        confident = False

                    if need_swap and (hint_a is not None or hint_b is not None):
                        colors["A"], colors["B"] = colors["B"], colors["A"]
                        self._jersey._track_team.clear()
                        self._jersey._track_team_votes.clear()

                    # When orientation is confident (winner ≥2× loser), lock now so
                    # a poor-quality unique-jersey crop cannot accidentally flip back.
                    # Brand colors (not actual jersey colors) produce large similar
                    # distances → ratio ≈ 1.0 → not confident → OCR fallback still runs.
                    if confident and (hint_a is not None or hint_b is not None):
                        self._kmeans_disambiguated = True
                        self._team_color_ready.update(("A", "B"))
                        logger.info(
                            "K-Means orientation locked by confident hint "
                            "(ratio=%.1f ≥ 2.0)", _d_loser / _d_winner,
                        )

        if self._tracker:
            for player in tracked_players:
                tid = player["track_id"]
                jersey_num = self._tracker.get_jersey_number(tid)
                if jersey_num is not None and self._roster:
                    # ── Team from COLOR is the authority ─────────────────────────
                    # Jersey numbers appear on BOTH teams (e.g. #8 in Team A AND B),
                    # so the number alone is ambiguous for team determination.
                    # K-Means jersey-color classification is the primary signal.
                    # Jersey number is used ONLY to look up the player's NAME within
                    # the already-color-determined team — it NEVER overrides the team.
                    # Priority after K-Means calibrated: current K-Means result wins over
                    # stale pre-calibration cache (cache may be wrong when same jersey
                    # number exists on both teams and roster fallback guessed wrong team).
                    # Before K-Means: sticky cache → color map → nothing.
                    if self._kmeans_calibrated and jersey_color_map.get(tid):
                        color_team = jersey_color_map[tid]
                    else:
                        color_team = (self._last_known_team.get(tid, "")
                                      or jersey_color_map.get(tid, ""))

                    # ── K-Means orientation via unique jerseys (no user hints) ───
                    # Fires only until _kmeans_disambiguated is set.
                    # A unique jersey (worn by exactly one team) lets us verify which
                    # K-Means cluster is Team A by comparing color vs. roster.
                    if (self._jersey and self._kmeans_calibrated
                            and not self._kmeans_disambiguated):
                        has_a = bool(self._roster.get(f"{jersey_num}_A"))
                        has_b = bool(self._roster.get(f"{jersey_num}_B"))
                        if has_a != has_b:   # XOR = unique to one team
                            expected = "A" if has_a else "B"
                            try:
                                x1, y1, x2, y2 = [int(v) for v in player["bbox"]]
                                crop = frame[max(0, y1):min(frame.shape[0], y2),
                                             max(0, x1):min(frame.shape[1], x2)]
                                if crop.size > 0:
                                    self._jersey._track_team.pop(tid, None)
                                    raw_color = self._jersey._classify_team(crop, tid)
                                    if raw_color is not None:
                                        if raw_color != expected:
                                            tc = self._jersey._team_colors
                                            tc["A"], tc["B"] = tc["B"], tc["A"]
                                            self._jersey._track_team.clear()
                                            self._jersey._track_team_votes.clear()
                                            logger.info(
                                                "Team-color A↔B swapped: jersey #%d unique to "
                                                "team %s but color said %s → clusters swapped",
                                                jersey_num, expected, raw_color,
                                            )
                                        else:
                                            logger.info(
                                                "Team-color verified: jersey #%d unique to team %s ✓",
                                                jersey_num, expected,
                                            )
                                        self._kmeans_disambiguated = True
                                        self._team_color_ready.update(("A", "B"))
                                        color_team = expected
                            except Exception as e:
                                logger.debug("K-Means disambiguation error: %s", e)

                    # ── Name lookup in color-team's roster ────────────────────────
                    # Look up the player name using the color-determined team.
                    # This handles shared numbers correctly (#8 Team A → Syahmi,
                    # #8 Team B → Dylan) without ever changing the team assignment.
                    if color_team:
                        entry = _roster_entry(self._roster, jersey_num, color_team)
                        player["team"] = color_team          # color always wins
                        self._last_known_team[tid] = color_team
                        if self._stats:
                            self._stats.confirm_player_team(tid, color_team)
                    else:
                        # Color not yet determined (K-Means not calibrated yet).
                        # Fall back to unqualified roster lookup as a temporary hint;
                        # this will be corrected once K-Means fires.
                        entry = _roster_entry(self._roster, jersey_num, "")
                        if entry.get("team"):
                            color_team = entry["team"]
                            player["team"] = color_team
                            self._last_known_team[tid] = color_team
                            if self._stats:
                                self._stats.confirm_player_team(tid, color_team)
                    continue
                if tid in self._last_known_team:
                    player["team"] = self._last_known_team[tid]
                elif tid in jersey_color_map:
                    player["team"] = jersey_color_map[tid]
                    self._last_known_team[tid] = jersey_color_map[tid]
                    if self._stats:
                        self._stats.confirm_player_team(tid, jersey_color_map[tid])

        # ── 6. Action classification — run every ACTION_EVERY_N_FRAMES ───────────
        if self._action and tracked_players and frame_id % ACTION_EVERY_N_FRAMES == 0:
            try:
                actions = self._action.classify(
                    frame, tracked_players,
                    pose_results=frame_data.get("pose"),
                    ball_info=ball_info,
                    frame_buffer=self._frame_buffer,
                )
                frame_data["actions"] = actions
            except Exception as e:
                logger.debug("Action frame %d: %s", frame_id, e)
        _t6 = time.perf_counter()

        # ── 7. Event engine ───────────────────────────────────────────────
        # Ensure event engine always has hoop pixel references for scoring.
        # When best-object-basketball.pt doesn't detect the hoop this frame
        # (camera angle, occlusion) AND court is calibrated, synthesise hoop
        # pixel positions from FIBA coordinates via court_to_pixel() so the
        # pixel-space FG zone check never silently blocks scoring.
        if (
            self._court
            and not frame_data.get("detections", {}).get("hoops")
            and frame_data.get("court", {}).get("is_calibrated")
        ):
            FIBA_HOOPS_M = [[1.575, 7.5], [26.425, 7.5]]
            synthetic_hoops = []
            for court_pos in FIBA_HOOPS_M:
                try:
                    px = self._court.court_to_pixel(court_pos)
                    if px:
                        synthetic_hoops.append({"center": list(px), "confidence": 0.0})
                except Exception:
                    pass
            if synthetic_hoops:
                frame_data.setdefault("detections", {})["hoops"] = synthetic_hoops

        # When court is NOT calibrated and exactly 1 hoop was detected this frame,
        # add a pixel-space mirror so the opposite basket is also covered.
        # Full-court cameras often only reliably detect the near hoop; the far hoop
        # (where the other team attacks) stays invisible → near-zero SHOT_ATTEMPs
        # for the away team. Mirroring (x → frame_width - x) is a good approximation
        # when the camera captures both ends roughly symmetrically.
        if (
            self._frame_width > 0
            and not frame_data.get("court", {}).get("is_calibrated")
        ):
            cur_hoops = frame_data.get("detections", {}).get("hoops", [])
            if len(cur_hoops) == 1:
                h = cur_hoops[0]
                cx = cy = None
                if h.get("center") and len(h["center"]) >= 2:
                    cx, cy = h["center"][0], h["center"][1]
                elif h.get("bbox") and len(h["bbox"]) == 4:
                    b = h["bbox"]
                    cx = (b[0] + b[2]) / 2
                    cy = (b[1] + b[3]) / 2
                if cx is not None and cy is not None:
                    frame_data["detections"]["hoops"].append(
                        {"center": [self._frame_width - cx, cy], "confidence": 0.0}
                    )

        if self._events_engine:
            try:
                events = self._events_engine.process(frame_data)
                frame_data["events"] = events or []
            except Exception as e:
                logger.warning("EventEngine frame %d: %s", frame_id, e, exc_info=True)
        _t7 = time.perf_counter()

        # ── 8. Stats update ───────────────────────────────────────────────
        if self._stats:
            tracking_snapshot: dict = {}
            for player in tracked_players:
                cp = player.get("court_pos")
                if cp:
                    tracking_snapshot[player["track_id"]] = (
                        cp[0], cp[1], timestamp_ms
                    )
                # No fallback pixel-to-meter when court is uncalibrated — the linear
                # approximation assumes the full 28×15 m court fills the entire frame,
                # which is rarely true and produces systematically wrong speeds.

            # Build stats-compatible roster: {track_id: {name, jersey_number, team}}
            # using tracker jersey map + raw upload roster (jersey_str → name)
            stats_roster: dict = {}
            if self._tracker and self._roster:
                for tid, jersey in self._tracker.track_jersey_map.items():
                    known_team = self._last_known_team.get(tid, "")
                    entry = _roster_entry(self._roster, jersey, known_team)
                    name = entry.get("name", f"#{jersey}")
                    # Color-determined known_team is authoritative; roster's team
                    # field is only a fallback for players not yet color-classified.
                    team = known_team or entry.get("team", "")
                    stats_roster[tid] = {
                        "name":         name,
                        "jersey_number": jersey,
                        "team":          team,
                    }

            try:
                self._stats.update(
                    events=frame_data["events"],
                    tracking_snapshot=tracking_snapshot or None,
                    pose_snapshot=frame_data.get("pose"),
                    roster=stats_roster or None,
                )
            except Exception as e:
                logger.warning("Stats update frame %d: %s", frame_id, e, exc_info=True)

        self._frame_buffer.append(frame)

        # ── Per-step timing log (every 25 analyzed frames) ────────────────
        # Logs: det=detect trk=track court=court pose=pose
        #       ocr=jersey_ocr act=action evt=events sts=stats TOTAL ms
        _t8 = time.perf_counter()
        if frame_id % 25 == 0:
            ms = lambda a, b: round((b - a) * 1000)
            logger.info(
                "TIMING f%-4d  det=%3d  trk=%3d  court=%3d  pose=%3d  "
                "ocr=%3d  act=%3d  evt=%3d  sts=%3d  TOTAL=%3d ms  players=%d",
                frame_id,
                ms(_t0, _t1), ms(_t1, _t2), ms(_t2, _t3), ms(_t3, _t4),
                ms(_t4, _t5), ms(_t5, _t6), ms(_t6, _t7), ms(_t7, _t8),
                ms(_t0, _t8), len(tracked_players),
            )

        return frame_data

    # ── Per-frame bbox snapshot (for frontend time-based lookup) ─────────────

    def _snapshot_frame(self, frame_data: dict, timestamp_ms: int) -> dict:
        """
        Compact per-processed-frame snapshot stored to disk at finalize.

        Frontend fetches the full list and does binary search by ts to find the
        nearest entry for video.currentTime → frame-accurate bbox rendering
        regardless of video scrubbing, pause, or replay.

        Fields per player: i=trackId, j=jersey, t=team, b=norm_bbox, a=action, s=speedKmh
        """
        tracking = frame_data.get("tracking", {})

        # Build action lookup from this frame's action results (already in frame_data)
        actions_data = frame_data.get("actions", {})
        action_map: dict[int, str] = {
            a["track_id"]: _norm_action(a.get("action", ""))
            for a in actions_data.get("actions", [])
        }

        # Update sticky cache whenever a fresh classification is available
        for _tid, _label in action_map.items():
            if _label:
                self._last_action_cache[_tid] = _label

        players_out = []
        for player in tracking.get("tracked_players", []):
            tid    = player["track_id"]
            jersey = self._tracker.get_jersey_number(tid) if self._tracker else None
            team   = player.get("team", "") or ""
            raw    = player.get("bbox", [])
            if len(raw) == 4 and self._frame_width > 0:
                w, h = self._frame_width, self._frame_height
                norm = [
                    round(raw[0] / w, 4), round(raw[1] / h, 4),
                    round(raw[2] / w, 4), round(raw[3] / h, 4),
                ]
            else:
                norm = list(raw)
            # Prefer fresh action; fall back to last known so labels persist
            # across the 19 non-action frames between each classification cycle
            action_label = action_map.get(tid) or self._last_action_cache.get(tid, "")
            court_pos = player.get("court_pos")
            players_out.append({
                "i": tid, "j": jersey, "t": team, "b": norm,
                "a": action_label,
                "s": self._speed_cache.get(tid, 0.0),
                "cp": court_pos,   # [x_m, y_m] or None — for 2D court overlay sync
            })

        ball_raw = tracking.get("tracked_ball")
        ball_out = None
        if ball_raw:
            bb = ball_raw.get("bbox", [])
            if len(bb) == 4 and self._frame_width > 0:
                w, h = self._frame_width, self._frame_height
                ball_out = {
                    "b": [
                        round(bb[0] / w, 4), round(bb[1] / h, 4),
                        round(bb[2] / w, 4), round(bb[3] / h, 4),
                    ],
                    "cp": ball_raw.get("court_pos"),
                }

        return {
            "ts":   timestamp_ms,
            "p":    players_out,
            "bl":   ball_out,
            "sc":   list(self._pts_cache),
            "q_sc": [
                max(0, self._pts_cache[0] - self._q_pts_start[0]),
                max(0, self._pts_cache[1] - self._q_pts_start[1]),
            ],
            "q":    self._current_quarter,
        }

    # ── Incremental JSONL writer (progressive streaming) ─────────────────────

    def _append_frame_to_jsonl(self, entry: dict) -> None:
        """Append one frame snapshot as a JSON line for progressive streaming."""
        if not self._frames_jsonl_path:
            return
        try:
            line = _json_dumps(entry) + '\n'
            with self._jsonl_lock:
                with open(self._frames_jsonl_path, 'a') as f:
                    f.write(line)
        except Exception as e:
            logger.debug("JSONL append error (non-fatal): %s", e)

    # ── WebSocket message builder ─────────────────────────────────────────────

    def _build_ws_message(self, frame_data: dict, stats: dict) -> dict:
        """
        Build the WebSocket frame_update message (Master Prompt format).
        Pure function — safe to call without any live services.
        """
        tracking = frame_data.get("tracking", {})
        pose     = frame_data.get("pose",     {})
        actions  = frame_data.get("actions",  {})

        ball_raw = tracking.get("tracked_ball")

        # Index pose and action data by track_id for O(1) lookup
        pose_map   = {p["track_id"]: p for p in pose.get("poses", [])}
        action_map = {a["track_id"]: a for a in actions.get("actions", [])}

        player_stats_all = stats.get("player_stats", {})

        # ── Players (only jersey-confirmed) ───────────────────────────────
        # A player's bbox is shown only after jersey_no.pt + OCR confirm the
        # number (VOTE_THRESHOLD frames). Until then the player is tracked
        # internally but not broadcast — matches the gradual-discovery UX.
        players = []
        for player in tracking.get("tracked_players", []):
            tid          = player["track_id"]
            jersey_number = (self._tracker.get_jersey_number(tid)
                             if self._tracker else None)
            # Don't gate on jersey confirmation — send all tracked players.
            # jerseyNumber will be null in WS until OCR confirms it; bbox overlay
            # and MPI distance/speed accumulate regardless.

            pstat      = player_stats_all.get(tid, {})
            mpi        = pstat.get("mpi", {})
            pose_entry = pose_map.get(tid, {})
            act_entry  = action_map.get(tid, {})

            keypoints = [
                [kp.get("x", 0), kp.get("y", 0), kp.get("confidence", 0.0)]
                for kp in pose_entry.get("keypoints", [])
            ]

            # Normalize pixel bbox → [0, 1] for CourtOverlay canvas rendering
            raw_bbox = player.get("bbox", [])
            if len(raw_bbox) == 4 and self._frame_width > 0 and self._frame_height > 0:
                bx1, by1, bx2, by2 = raw_bbox
                norm_bbox = [
                    bx1 / self._frame_width,
                    by1 / self._frame_height,
                    bx2 / self._frame_width,
                    by2 / self._frame_height,
                ]
            else:
                norm_bbox = raw_bbox

            # Sticky courtPos: use current if available, else keep last known position
            cur_court_pos = player.get("court_pos")
            if cur_court_pos is not None:
                self._last_court_pos[tid] = cur_court_pos
            else:
                cur_court_pos = self._last_court_pos.get(tid)

            # Sticky action: prefer current frame result, fall back to last known
            raw_action = act_entry.get("action", "")
            if raw_action:
                self._last_action_cache[tid] = _norm_action(raw_action)
            ws_action = self._last_action_cache.get(tid, "Stand")

            players.append({
                "trackId":      tid,
                "jerseyNumber": jersey_number,
                "name":         pstat.get("name", f"Player_{tid}"),
                "team":         player.get("team", "") or pstat.get("team", ""),
                "bbox":         norm_bbox,
                "courtPos":     cur_court_pos,
                "action":       ws_action,
                "speedKmh":     mpi.get("avg_speed_kmh", 0.0),
                "keypoints":    keypoints,
            })

        # ── Ball ───────────────────────────────────────────────────────────
        ball_court_pos = None
        if ball_raw:
            ball_court_pos = ball_raw.get("court_pos")
            if ball_court_pos is None and ball_raw.get("center"):
                # Attempt pixel→court if court is calibrated
                court_info = frame_data.get("court", {})
                if court_info.get("is_calibrated") and self._court:
                    try:
                        ball_court_pos = self._court.pixel_to_court(
                            ball_raw["center"]
                        )
                    except Exception:
                        pass
            if ball_court_pos:
                self._ball_trajectory.append(ball_court_pos)

        # ── Score & possession ─────────────────────────────────────────────
        ts    = stats.get("team_stats", {})
        score = {
            "teamA": ts.get("A", {}).get("pts", 0),
            "teamB": ts.get("B", {}).get("pts", 0),
        }
        poss_raw  = stats.get("possession_pct", {"A": 50.0, "B": 50.0})
        possession = {
            "teamA": poss_raw.get("A", 50.0),
            "teamB": poss_raw.get("B", 50.0),
        }

        events       = frame_data.get("events", [])
        latest_event = self._map_event_for_ws(events[-1]) if events else None

        # K-Means detected jersey colors — only populated after calibration fires.
        # HSV [0-180, 0-255, 0-255] → null before first calibration.
        team_colors_raw = (self._jersey._team_colors
                           if self._jersey and self._kmeans_calibrated else {})
        team_colors_ws  = {k: [round(v, 1) for v in vals]
                           for k, vals in team_colors_raw.items()} if team_colors_raw else None

        return {
            "type":         "frame_update",
            "timestamp":    frame_data.get("timestamp_ms", 0),
            "wallClockMs":  int(time.time() * 1000),  # absolute wall time — useful for live mode
            "quarter":      frame_data.get("quarter", self._current_quarter),
            "gameClock":    self._format_game_clock(),
            "score":        score,
            "possession":   possession,
            "players":      players,
            "ball": {
                "bbox":       ball_raw.get("bbox", []) if ball_raw else [],
                "courtPos":   ball_court_pos,
                "trajectory": list(self._ball_trajectory),
            },
            "event":        latest_event,
            "fps":          round(self._fps_actual, 1),
            "gpuMetrics":   _query_gpu_metrics(),
            "teamColors":   team_colors_ws,   # {"A": [H,S,V], "B": [H,S,V]} or null
        }

    # ── Event format mapper ───────────────────────────────────────────────────

    _EVENT_TYPE_MAP = {
        "MADE_FG": "FGM", "MADE_3": "FGM", "SHOT_MADE": "FGM",
        "MISSED_FG": "FGA", "MISSED_3": "FGA", "SHOT_MISSED": "FGA",
        "REBOUND": "REB", "OREB": "REB", "DREB": "REB",
        "AST": "AST", "ASSIST": "AST",
        "STL": "STL", "STEAL": "STL",
        "BLK": "BLK", "BLOCK": "BLK",
        "TOV": "TOV", "TURNOVER": "TOV",
        "FOUL": "FOUL", "PERSONAL_FOUL": "FOUL",
    }

    def _map_event_for_ws(self, event: Optional[dict]) -> Optional[dict]:
        """Convert backend event dict → frontend GameEvent format."""
        if not event:
            return None
        etype      = str(event.get("type", "")).upper()
        event_type = self._EVENT_TYPE_MAP.get(etype)
        if not event_type:
            return None

        track_id   = event.get("track_id", 0)
        jersey_num = (self._tracker.get_jersey_number(track_id)
                      if self._tracker else None)

        points = None
        if event_type == "FGM":
            points = 3 if event.get("is_three") else 2

        # Use the event's own video timestamp so the log time matches
        # the video player position (elapsed MM:SS), not a game-clock countdown.
        ts_ms     = event.get("timestamp_ms", 0)
        elapsed_s = ts_ms / 1000.0
        evt_clock = f"{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}"

        return {
            "type":       "event",
            "eventType":  event_type,
            "trackId":    track_id,
            "playerId":   jersey_num if jersey_num is not None else track_id,
            "playerName": (f"#{jersey_num}" if jersey_num is not None
                           else f"Track_{track_id}"),
            "team":       event.get("team", ""),
            "points":     points,
            "quarter":    event.get("quarter", self._current_quarter),
            "gameClock":  evt_clock,
            "courtPos":   event.get("court_pos"),
        }

    # ── Persistence helpers ───────────────────────────────────────────────────

    async def _save_event_mongo(self, event: dict, match_id: str) -> None:
        """Persist one event to MongoDB events collection (no-op if mongo absent)."""
        if self._mongo_db is None:
            return
        try:
            ts_ms     = event.get("timestamp_ms", 0)
            elapsed_s = ts_ms / 1000.0
            game_clock = f"{int(elapsed_s // 60):02d}:{int(elapsed_s % 60):02d}"
            doc = {**event, "match_id": match_id, "saved_at": time.time(),
                   "game_clock": game_clock}
            self._mongo_db["events"].insert_one(doc)   # pymongo is sync — no await
        except Exception as e:
            logger.warning("MongoDB write error: %s", e)

    async def _push_redis(self, match_id: str, data: dict) -> None:
        """Publish data to Redis channel match:{match_id} (no-op if redis absent)."""
        if self._redis is None:
            return
        try:
            payload = _json_dumps(data)
            result = self._redis.publish(f"match:{match_id}", payload)
            # redis.asyncio returns a coroutine; plain redis returns an int
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error("Redis publish error: %s", e)

    async def _set_redis_stats(self, match_id: str, stats: dict, ttl: int = 3600) -> None:
        """Write stats snapshot to Redis key stats:{match_id} for REST polling (no-op if redis absent)."""
        if self._redis is None:
            return
        try:
            # Inject score derived from team_stats so the REST endpoint and
            # per-quarter blobs both carry an accurate score field.
            enriched = dict(stats)
            ts = stats.get("team_stats", {})
            enriched["score"] = {
                "team_a": ts.get("A", {}).get("pts", 0),
                "team_b": ts.get("B", {}).get("pts", 0),
            }
            payload = _json_dumps(enriched)
            result = self._redis.set(f"stats:{match_id}", payload, ex=ttl)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.error("Redis stats set error: %s", e)

    async def _merge_all_quarters(self, match_id: str) -> dict:
        """
        Read per-quarter Redis keys stats:{match_id}:q1..q4 and merge into one
        combined stats dict.  Player data is merged by (jersey_number, team) so
        that track_id restarts between quarter uploads don't create duplicate rows.
        """
        SUMABLE = (
            "pts", "fgm", "fga", "three_pm", "three_pa", "ftm", "fta",
            "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "min",
        )
        by_player: dict = {}   # (jersey_number, team) -> merged player dict
        latest_blob: dict = {}

        for q in range(1, 5):
            try:
                raw = self._redis.get(f"stats:{match_id}:q{q}")
                if asyncio.iscoroutine(raw):
                    raw = await raw
                if not raw:
                    continue
                blob = json.loads(raw)
                latest_blob = blob
                for pdata in (blob.get("player_stats") or {}).values():
                    jn   = pdata.get("jersey_number")   # None = unidentified
                    team = pdata.get("team", "")
                    # Unidentified players use their unique name (contains track_id)
                    # as key so they don't collapse into a single inflated "Player_0".
                    key  = (jn, team) if jn is not None else (pdata.get("name", ""), team)
                    if key not in by_player:
                        import copy as _copy
                        entry = _copy.deepcopy(pdata)
                        by_player[key] = entry
                    else:
                        entry = by_player[key]
                        for field in SUMABLE:
                            val = (pdata.get("total_stats") or {}).get(field, 0)
                            if isinstance(val, (int, float)):
                                entry.setdefault("total_stats", {})[field] = (
                                    entry.get("total_stats", {}).get(field, 0) + val
                                )
            except Exception as exc:
                logger.error("Merge q%d error: %s", q, exc)

        if not by_player:
            return latest_blob or {}

        # Recompute derived per-player fields
        for entry in by_player.values():
            ts  = entry.get("total_stats", {})
            fgm = ts.get("fgm", 0);  fga = ts.get("fga", 0)
            ts["fg_pct"] = round(fgm / fga, 3) if fga > 0 else 0.0
            pts = ts.get("pts", 0); reb = ts.get("reb", 0)
            ast = ts.get("ast", 0); stl = ts.get("stl", 0)
            blk = ts.get("blk", 0); tov = ts.get("tov", 0)
            ts["eff"] = (pts + reb + ast + stl + blk) - (fga - fgm + tov)

        player_stats_out = {str(i): v for i, v in enumerate(by_player.values())}

        # Re-derive team_stats from merged players
        TEAM_FIELDS = ("pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga")
        team_out = {
            "A": {f: 0 for f in TEAM_FIELDS + ("fg_pct",)},
            "B": {f: 0 for f in TEAM_FIELDS + ("fg_pct",)},
        }
        for pdata in by_player.values():
            t_key = pdata.get("team", "")
            if t_key not in ("A", "B"):
                continue
            ts = pdata.get("total_stats", {})
            for f in TEAM_FIELDS:
                team_out[t_key][f] += ts.get(f, 0)
        for t in team_out.values():
            t["fg_pct"] = round(t["fgm"] / t["fga"], 3) if t["fga"] > 0 else 0.0

        # Derive score from the already-summed team_stats (correct for both
        # full-game uploads and separate per-quarter uploads).
        merged_score = {
            "team_a": int(team_out.get("A", {}).get("pts", 0)),
            "team_b": int(team_out.get("B", {}).get("pts", 0)),
        }

        return {
            **{k: v for k, v in latest_blob.items()
               if k not in ("player_stats", "team_stats", "score")},
            "score":        merged_score,
            "player_stats": player_stats_out,
            "team_stats":   team_out,
        }

    def _patch_frame_scores(self) -> None:
        """
        Retroactively correct sc and q_sc in every frame snapshot.

        During processing, K-Means team calibration and jersey OCR confirmation
        lag behind actual scoring events — so events arrive with team="" and are
        not counted in team_stats at that moment. By _finalize() time all teams
        are known. We rebuild a correct score timeline here and patch _frame_store.
        """
        if not self._scoring_events or not self._frame_store or not self._stats:
            return

        # Final team assignment per track_id from StatsCalculator
        final_stats = self._stats.get_live_stats()
        tid_to_team: dict = {
            tid: p.get("team", "")
            for tid, p in final_stats.get("player_stats", {}).items()
        }

        # Supplement from _last_known_team (set when jersey OCR + roster confirmed a team).
        # This covers track_ids where StatsCalculator's roster was not updated in time.
        for tid, team in self._last_known_team.items():
            if team in ("A", "B") and not tid_to_team.get(tid):
                tid_to_team[tid] = team

        # Cross-reference via jersey for track_ids that changed mid-match.
        # tracker.track_jersey_map retains ALL track_ids (even dropped ones), so
        # old_tid → jersey → team still resolves even after ByteTrack reassigned the ID.
        if self._tracker:
            _jn_to_team: dict = {}
            for jtid, jteam in self._last_known_team.items():
                if jteam in ("A", "B"):
                    jn = self._tracker.track_jersey_map.get(jtid)
                    if jn is not None:
                        _jn_to_team[jn] = jteam
            for _, stid, _, _ in self._scoring_events:
                if not tid_to_team.get(stid):
                    jn = self._tracker.track_jersey_map.get(stid)
                    if jn is not None and jn in _jn_to_team:
                        tid_to_team[stid] = _jn_to_team[jn]

        # Build cumulative score timeline sorted by timestamp
        # Entry: (ts_ms, total_A, total_B, quarter)
        cum_a, cum_b = 0, 0
        timeline: list = [(0, 0, 0, self._start_quarter)]
        for ts_ms, tid, pts, q in sorted(self._scoring_events):
            team = tid_to_team.get(tid, "")
            if team == "A":
                cum_a += pts
            elif team == "B":
                cum_b += pts
            else:
                continue  # still unknown — skip
            timeline.append((ts_ms, cum_a, cum_b, q))

        if len(timeline) <= 1:
            return  # no attributable scoring events — nothing to patch

        # Patch each snapshot: binary search for latest timeline entry ≤ snap ts
        tl_ts = [e[0] for e in timeline]
        from bisect import bisect_right

        # Quarter boundary cumulative scores: q_sc = sc - score_at_quarter_start
        q_start: dict = {self._start_quarter: (0, 0)}

        for snap in self._frame_store:
            snap_ts = snap.get("ts", 0)
            idx = bisect_right(tl_ts, snap_ts) - 1
            if idx < 0:
                idx = 0
            _, sa, sb, q = timeline[idx]
            snap["sc"] = [sa, sb]

            # Derive q_sc: score accumulated within the current quarter
            if q not in q_start:
                # First time we see this quarter in the timeline — record start score
                q_start[q] = (sa, sb)
            qa, qb = q_start.get(q, (0, 0))
            snap["q_sc"] = [max(0, sa - qa), max(0, sb - qb)]
            snap["q"] = q

        logger.info(
            "_patch_frame_scores: patched %d frames, final score A=%d B=%d",
            len(self._frame_store), cum_a, cum_b,
        )

    async def _finalize(self, match_id: str) -> None:
        """Update MongoDB match status; push final stats; write output files."""

        # ── 0. Retroactively patch sc/q_sc using final team assignments ──────
        # During processing, team may be "" for early scoring events (K-Means not
        # yet calibrated, jersey OCR not yet confirmed). At finalize time all teams
        # are known → rebuild correct running score and patch every frame snapshot.
        self._patch_frame_scores()

        # Rewrite JSONL with retroactively patched sc/q_sc so the /stream endpoint
        # serves correct scores after processing completes.
        if self._frames_jsonl_path and self._frame_store:
            try:
                with self._jsonl_lock:
                    with open(self._frames_jsonl_path, 'w') as _jfp:
                        for _entry in self._frame_store:
                            _jfp.write(_json_dumps(_entry) + '\n')
                logger.info("JSONL stream rewritten with patched scores (%d frames)",
                            len(self._frame_store))
            except Exception as _je:
                logger.warning("JSONL rewrite failed: %s", _je)

        # ── 1. Write per-frame bbox JSON (frontend time-lookup) ───────────────
        # Use quarter-suffixed filename so Q1 and Q2 files don't overwrite each other.
        if self._frame_store and self._video_path:
            frames_path = os.path.splitext(self._video_path)[0] + f"_q{self._start_quarter}_frames.json"
            try:
                with open(frames_path, "w") as fp:
                    fp.write(_json_dumps(self._frame_store))
                self._frames_json_path = frames_path
                logger.info("Frame store written: %s  entries=%d",
                            frames_path, len(self._frame_store))
            except Exception as e:
                logger.warning("Frame store write failed: %s", e)

        # ── 2 & 3. Output folder artefacts ────────────────────────────────────
        # Wrapped in a single try/except so any failure here (disk full,
        # permission error, rendering crash) never prevents MongoDB + Redis
        # from running — dashboard always gets its processing_complete signal.
        out_dir = os.path.join(OUTPUT_DIR, match_id)
        try:
            os.makedirs(out_dir, exist_ok=True)

            csv_path = self._save_stats_csv(out_dir, match_id)
            if csv_path:
                self._output_csv_path = csv_path
                logger.info("Stats CSV: %s", csv_path)

            if RENDER_OUTPUT_VIDEO and self._frame_store:
                loop = asyncio.get_event_loop()
                video_out = await loop.run_in_executor(
                    None, self._render_output_video, out_dir, match_id, self._start_quarter
                )
                if video_out:
                    self._output_video_path = video_out

        except Exception as exc:
            logger.warning("Output file generation failed (dashboard unaffected): %s", exc)

        # ── 4. Persist to MongoDB (always runs) ────────────────────────────────
        if self._mongo_db is not None:
            try:
                extra: dict = {}
                if self._frames_json_path:
                    extra["frames_json_path"]  = self._frames_json_path
                if self._output_csv_path:
                    extra["output_csv_path"]   = self._output_csv_path
                if self._output_video_path:
                    extra["output_video_path"] = self._output_video_path
                self._mongo_db["matches"].update_one(
                    {"match_id": match_id},
                    {"$set": {"match_id": match_id,
                               "status": self._status,
                               "processed_at": time.time(),
                               **extra}},
                    upsert=True,
                )
            except Exception as e:
                logger.warning("MongoDB finalize error: %s", e)

        # ── 5. Redis broadcast + final stats key (always runs) ────────────────
        if self._redis is not None and self._stats is not None:
            try:
                final_stats = self._stats.get_live_stats()

                # Write per-quarter key (7-day TTL) — survives subsequent uploads.
                # This is the source of truth for "which quarters are done" and
                # enables cross-quarter stat merging without losing Q1 when Q2 runs.
                q_key = f"stats:{match_id}:q{self._start_quarter}"
                result = self._redis.set(q_key, _json_dumps(final_stats),
                                         ex=86400 * 7)
                if asyncio.iscoroutine(result):
                    await result
                logger.info("Quarter stats saved: key=%s", q_key)

                # Merge all uploaded quarters → combined key for /api/stats/live
                merged_stats = await self._merge_all_quarters(match_id)
                await self._set_redis_stats(match_id, merged_stats, ttl=86400)

                await self._push_redis(match_id, {
                    "type":              "processing_complete",
                    "status":            self._status,
                    "stats":             merged_stats,
                    "output_video_path": self._output_video_path or None,
                    "output_csv_path":   self._output_csv_path   or None,
                })
            except Exception as e:
                logger.error("Redis finalize error: %s", e)

        self._executor.shutdown(wait=False)

    # ── Output helpers ───────────────────────────────────────────────────────

    def _save_stats_csv(self, out_dir: str, match_id: str) -> Optional[str]:
        """Write final player stats to CSV matching the frontend export format."""
        if self._stats is None:
            return None
        try:
            live = self._stats.get_live_stats()
            player_stats = live.get("player_stats", {})
            if not player_stats:
                return None

            os.makedirs(out_dir, exist_ok=True)
            csv_path = os.path.join(out_dir, f"{match_id}_q{self._start_quarter}_stats.csv")

            headers = [
                "#", "Name", "Team", "MIN", "PTS",
                "2P", "3P", "FT", "FG%",
                "OFF", "DEF", "TOT", "AST", "STL", "TOV",
                "C", "R", "M", "+/-", "EFF",
            ]
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                for tid, ps in sorted(player_stats.items(),
                                      key=lambda x: x[1].get("jersey_number") or 0):
                    ts = ps.get("total_stats", {})
                    writer.writerow([
                        ps.get("jersey_number") or tid,
                        ps.get("name", f"Player_{tid}"),
                        ps.get("team", ""),
                        round(ts.get("min", 0) / 60, 1),
                        ts.get("pts", 0),
                        f"{ts.get('fgm', 0)}/{ts.get('fga', 0)}",
                        f"{ts.get('three_pm', 0)}/{ts.get('three_pa', 0)}",
                        f"{ts.get('ftm', 0)}/{ts.get('fta', 0)}",
                        round(ts.get("fg_pct", 0.0) * 100, 1),
                        ts.get("oreb", 0),
                        ts.get("dreb", 0),
                        ts.get("reb", 0),
                        ts.get("ast", 0),
                        ts.get("stl", 0),
                        ts.get("tov", 0),
                        ts.get("pf", 0),
                        ts.get("blk", 0),
                        0,   # not in current schema
                        ts.get("plus_minus", 0),
                        round(ts.get("eff", 0.0), 1),
                    ])
            return csv_path
        except Exception as exc:
            logger.warning("_save_stats_csv failed: %s", exc)
            return None

    def _render_output_video(self, out_dir: str, match_id: str, quarter: int = 1) -> Optional[str]:
        """
        Re-read the original video and draw bbox overlays from stored frame data.
        Writes to out_dir/{match_id}_q{quarter}_analyzed.mp4.  Returns path or None on error.
        Runs synchronously — call via run_in_executor to avoid blocking the loop.
        """
        if not self._frame_store or not self._video_path:
            return None
        try:
            os.makedirs(out_dir, exist_ok=True)

            cap = cv2.VideoCapture(self._video_path)
            if not cap.isOpened():
                logger.warning("_render_output_video: cannot open %s", self._video_path)
                return None

            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()  # metadata only — frames will come from NVDEC pipe below

            out_path = os.path.join(out_dir, f"{match_id}_q{quarter}_analyzed.mp4")

            # Stream raw BGR frames directly into ffmpeg via stdin pipe.
            # h264_nvenc (GPU hardware encoder) runs at ~300 fps 1080p — far
            # faster than libx264 and needs no intermediate mp4v file.
            # Fall back to libx264 ultrafast if NVENC is unavailable.
            import subprocess as _sp

            def _open_encoder(codec: str, extra_flags: list):
                return _sp.Popen(
                    ["ffmpeg", "-y",
                     "-f", "rawvideo", "-vcodec", "rawvideo",
                     "-pix_fmt", "bgr24",
                     "-s", f"{width}x{height}",
                     "-r", f"{src_fps:.4f}",
                     "-i", "pipe:0",
                     "-c:v", codec, *extra_flags,
                     "-an", "-movflags", "+faststart",
                     out_path],
                    stdin=_sp.PIPE, stderr=_sp.PIPE,
                )

            _nvenc_ok = False
            try:
                enc_proc = _open_encoder("h264_nvenc", ["-preset", "p2", "-b:v", "5M"])
                _nvenc_ok = True
                logger.info(
                    "Render encoder: h264_nvenc GPU  %dx%d @ %.1f fps",
                    width, height, src_fps,
                )
            except Exception as _ne:
                enc_proc = _open_encoder("libx264", ["-preset", "ultrafast", "-crf", "23"])
                logger.info(
                    "Render encoder: libx264 ultrafast (NVENC unavailable: %s)  %dx%d @ %.1f fps",
                    _ne, width, height, src_fps,
                )

            # NVDEC: decode source video on GPU, stream raw BGR frames via pipe.
            # -hwaccel cuda makes ffmpeg auto-select h264_cuvid / hevc_cuvid when
            # available; if the GPU can't handle the codec it silently falls back
            # to CPU decode — no extra code needed.
            _frame_bytes = width * height * 3
            dec_proc = _sp.Popen(
                [
                    "ffmpeg", "-loglevel", "error",
                    "-hwaccel", "cuda",
                    "-i", self._video_path,
                    "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1",
                ],
                stdout=_sp.PIPE,
                stderr=_sp.DEVNULL,
            )
            logger.info(
                "Render decoder: ffmpeg -hwaccel cuda → bgr24 pipe  %dx%d",
                width, height,
            )

            # Pre-build sorted timestamp list for O(log n) binary search per frame
            ts_list = [e["ts"] for e in self._frame_store]

            # BGR color palette (matches frontend CourtOverlay colors)
            C_A    = (212, 188,   0)   # Team A — cyan   #00BCD4
            C_B    = ( 22, 115, 249)   # Team B — orange #F97316
            C_UNK  = (175, 163, 156)   # unknown — gray  #9CA3AF
            C_BALL = ( 21, 204, 250)   # ball — yellow   #FACC15
            C_TEXT = (255, 255, 255)
            C_DARK = (  0,   0,   0)

            FONT      = cv2.FONT_HERSHEY_SIMPLEX
            FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

            # Team names from match roster metadata (fall back to "TIM A / B")
            name_a = "TIM A"
            name_b = "TIM B"

            def _draw_label_above(img, x1, y1, x2, text, color):
                """Filled pill label above bbox top-left."""
                scale, thick = 0.55, 1
                (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
                pad = 4
                rx1, ry1 = x1, max(0, y1 - th - pad * 2)
                rx2, ry2 = min(x1 + tw + pad * 2, img.shape[1] - 1), y1
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, -1)
                cv2.putText(img, text, (rx1 + pad, ry2 - pad),
                            FONT, scale, C_TEXT, thick, cv2.LINE_AA)

            def _draw_label_below(img, x1, y2, x2, text, color):
                """Filled pill label below bbox bottom-left."""
                scale, thick = 0.45, 1
                (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
                pad = 3
                rx1, ry1 = x1, y2
                rx2, ry2 = min(x1 + tw + pad * 2, img.shape[1] - 1), min(y2 + th + pad * 2, img.shape[0] - 1)
                cv2.rectangle(img, (rx1, ry1), (rx2, ry2), color, -1)
                cv2.putText(img, text, (rx1 + pad, ry2 - pad),
                            FONT, scale, C_TEXT, thick, cv2.LINE_AA)

            def _score_banner(img, score_a, score_b, quarter, w, h):
                """Semi-transparent scoreboard bar at the top of the frame."""
                bar_h = 36
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

                q_text  = f"Q{quarter}"
                sc_text = f"{score_a}  :  {score_b}"

                # Quarter pill
                (qw, qh), _ = cv2.getTextSize(q_text, FONT_BOLD, 0.65, 2)
                cv2.putText(img, q_text, (8, bar_h - 8),
                            FONT_BOLD, 0.65, (80, 200, 255), 2, cv2.LINE_AA)

                # Score centred
                (sw, sh), _ = cv2.getTextSize(sc_text, FONT_BOLD, 0.75, 2)
                cx = (w - sw) // 2
                cv2.putText(img, sc_text, (cx, bar_h - 7),
                            FONT_BOLD, 0.75, C_TEXT, 2, cv2.LINE_AA)

                # Team name labels flanking the score
                margin = 8
                (aw, _), _ = cv2.getTextSize(name_a, FONT, 0.45, 1)
                cv2.putText(img, name_a, (cx - aw - margin, bar_h - 9),
                            FONT, 0.45, C_A, 1, cv2.LINE_AA)
                cv2.putText(img, name_b, (cx + sw + margin, bar_h - 9),
                            FONT, 0.45, C_B, 1, cv2.LINE_AA)

            frame_idx = 0
            while True:
                raw = dec_proc.stdout.read(_frame_bytes)
                if len(raw) < _frame_bytes:
                    break
                frame = np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy()

                ts_ms = int(frame_idx / src_fps * 1000)

                # Binary search: nearest stored snapshot within 2 s
                idx = bisect.bisect_left(ts_list, ts_ms)
                if idx >= len(ts_list):
                    idx = len(ts_list) - 1
                elif idx > 0 and abs(ts_list[idx - 1] - ts_ms) < abs(ts_list[idx] - ts_ms):
                    idx -= 1

                if abs(ts_list[idx] - ts_ms) < 2000:
                    snap = self._frame_store[idx]

                    # ── Players ───────────────────────────────────────────
                    for p in snap.get("p", []):
                        b = p.get("b", [])
                        if len(b) != 4:
                            continue
                        x1, y1 = int(b[0] * width), int(b[1] * height)
                        x2, y2 = int(b[2] * width), int(b[3] * height)
                        if (x2 - x1) < 4 or (y2 - y1) < 4:
                            continue
                        t     = p.get("t", "")
                        color = C_A if t == "A" else C_B if t == "B" else C_UNK

                        # Bbox
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                        # Jersey + team label above bbox
                        jersey = p.get("j")
                        id_str = f"#{jersey}" if jersey is not None else f"T{p['i']}"
                        team_str = f" [{t}]" if t else ""
                        _draw_label_above(frame, x1, y1, x2, id_str + team_str, color)

                        # Action label below bbox (only when non-trivial)
                        action = (p.get("a") or "").upper()
                        if action and action not in ("STAND", ""):
                            action_color = (0, 200, 80) if action == "SHOOT" else (
                                (200, 150, 0) if action in ("PASS", "CATCH") else
                                (100, 100, 200)
                            )
                            _draw_label_below(frame, x1, y2, x2, action, action_color)

                        # Speed below action (only >0.5 km/h to avoid static noise)
                        spd = float(p.get("s") or 0.0)
                        if spd > 0.5:
                            spd_str = f"{spd:.1f} km/h"
                            spd_y   = y2 + 18 if action and action not in ("STAND", "") else y2
                            cv2.putText(frame, spd_str,
                                        (x1 + 2, min(spd_y + 14, height - 4)),
                                        FONT, 0.38, color, 1, cv2.LINE_AA)

                    # ── Ball ──────────────────────────────────────────────
                    bl = snap.get("bl")
                    if bl:
                        b = bl.get("b", [])
                        if len(b) == 4:
                            x1, y1 = int(b[0] * width), int(b[1] * height)
                            x2, y2 = int(b[2] * width), int(b[3] * height)
                            if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), C_BALL, 2)
                                cv2.putText(frame, "BALL", (x1, max(y1 - 4, 12)),
                                            FONT, 0.40, C_BALL, 1, cv2.LINE_AA)

                    # ── Score banner (drawn last so it's always on top) ───
                    sc   = snap.get("sc", [0, 0])
                    qtr  = snap.get("q", 1)
                    _score_banner(frame, sc[0], sc[1], qtr, width, height)

                try:
                    enc_proc.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    logger.warning(
                        "_render_output_video: encoder pipe closed early at frame %d",
                        frame_idx,
                    )
                    break
                frame_idx += 1

            dec_proc.stdout.close()
            try:
                dec_proc.wait(timeout=30)
            except _sp.TimeoutExpired:
                dec_proc.kill()
            enc_proc.stdin.close()
            try:
                rc = enc_proc.wait(timeout=300)
            except _sp.TimeoutExpired:
                enc_proc.kill()
                rc = -1
                logger.warning("_render_output_video: encoder timed out — killed")

            if rc != 0:
                _err = (enc_proc.stderr.read() or b"")[-400:]
                logger.warning(
                    "Encoder exit %d%s — output may be incomplete.\n%s",
                    rc,
                    " (h264_nvenc)" if _nvenc_ok else " (libx264)",
                    _err.decode(errors="replace"),
                )
            else:
                logger.info(
                    "Output video encoded (%s): %s  frames=%d",
                    "h264_nvenc" if _nvenc_ok else "libx264",
                    out_path, frame_idx,
                )

            return out_path

        except Exception as exc:
            logger.warning("_render_output_video failed: %s", exc)
            return None

    # ── Quarter management ────────────────────────────────────────────────────

    def _detect_quarter_change(self, frame_data: dict) -> bool:
        """
        True when current quarter's elapsed time exceeds QUARTER_DURATION_S
        and a next quarter exists (quarters 1–4 only).

        Override point: plug in scoreboard-OCR result via frame_data["quarter"]
        when that pipeline step is implemented.
        """
        # If frame_data carries an authoritative quarter from OCR scoreboard,
        # always respect it — even in lock_quarter mode.
        ocr_q = frame_data.get("quarter_ocr")
        if ocr_q and ocr_q != self._current_quarter:
            return True

        # Single-quarter upload: never time-advance (the whole video is one quarter)
        if getattr(self, "_lock_quarter", False):
            return False

        if self._current_quarter >= 4:
            return False

        frames_in_q = self._frame_count - self._quarter_start_frame
        elapsed_s   = frames_in_q / max(self._source_fps, 1.0)
        return elapsed_s >= QUARTER_DURATION_S

    def _handle_quarter_change(self) -> None:
        # Snapshot cumulative score at end of this quarter so q_sc resets to 0 for next
        self._q_pts_start         = list(self._pts_cache)
        self._current_quarter    += 1
        self._quarter_start_frame = self._frame_count
        logger.info("Quarter %d started (frame=%d)",
                    self._current_quarter, self._frame_count)

        # Preserve cumulative score before event_engine.reset() clears it
        carried_score = dict(self._events_engine.score) if self._events_engine else {}

        for component, method in [
            (self._tracker,       "reset"),
            (self._events_engine, "reset"),
            (self._action,        "reset"),
            (self._jersey,        "reset"),   # clears votes + team cache; IDs restart
        ]:
            if component and hasattr(component, method):
                try:
                    getattr(component, method)()
                except Exception as e:
                    logger.warning("%s.reset() error: %s",
                                   type(component).__name__, e)

        # Restore cumulative score so event_engine keeps the running game total
        if self._events_engine and carried_score:
            self._events_engine.score.update(carried_score)

        if self._stats:
            try:
                self._stats.reset_quarter(self._current_quarter)
            except Exception as e:
                logger.warning("stats.reset_quarter error: %s", e)

        # Reset per-quarter jersey-confirmation tracking so players confirmed
        # early in the new quarter still get a jersey_confirmed WS message.
        self._confirmed_jersey_sent.clear()
        self._last_known_team.clear()
        self._last_action_cache.clear()

    # ── Timing helpers ────────────────────────────────────────────────────────

    async def _throttle_fps(self, target_fps: float, frame_start: float) -> None:
        """Async-sleep to maintain target_fps. No-op if pipeline is already slow."""
        elapsed = time.perf_counter() - frame_start
        sleep_t = (1.0 / max(target_fps, 1.0)) - elapsed
        if sleep_t > 0:
            await asyncio.sleep(sleep_t)

    def _format_game_clock(self) -> str:
        """Return 'MM:SS' countdown for the current quarter."""
        frames_in_q = self._frame_count - self._quarter_start_frame
        elapsed_s   = frames_in_q / max(self._source_fps, 1.0)
        remaining_s = max(0.0, QUARTER_DURATION_S - elapsed_s)
        return f"{int(remaining_s // 60):02d}:{int(remaining_s % 60):02d}"

    # ── Public helpers ────────────────────────────────────────────────────────

    @property
    def current_quarter(self) -> int:
        return self._current_quarter

    def get_progress(self) -> dict:
        # NVDEC container metadata can under-report frame count; cap display
        # so progress never exceeds 100% and ETA never goes negative.
        total = self._total_frames
        count = min(self._frame_count, total) if total > 0 else self._frame_count
        return {
            "frames_processed": count,
            "total_frames":     total,
            "fps_actual":       round(self._fps_actual, 2),
            "fps_dropped":      self._frames_dropped,
            "status":           self._status,
            "quarter":          self._current_quarter,
            "game_clock":       self._format_game_clock(),
        }

    def stop(self) -> None:
        """Signal graceful stop. process_video() will clean up and update MongoDB."""
        logger.info("Stop requested (status=%s)", self._status)
        self._stop_signal.set()
        self._status = "stopping"


# ── Factory ───────────────────────────────────────────────────────────────────

def create_video_processor(
    models_path:   Optional[str] = None,
    mongo_db                     = None,
    redis_client                 = None,
    device:        Optional[str] = None,
) -> VideoProcessor:
    vp = VideoProcessor(
        models_path=models_path,
        mongo_db=mongo_db,
        redis_client=redis_client,
        device=device,
    )
    vp._init_pipeline(vp._models_path)
    return vp


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    print("=== VideoProcessor smoke test ===\n")

    # ── 1. _init_pipeline with missing models (SKIP) ──────────────────────────
    print("--- 1. _init_pipeline (missing models) ---")
    vp1 = VideoProcessor(models_path="/nonexistent/models")
    vp1._init_pipeline("/nonexistent/models")
    print(f"  No crash ✓")
    print(f"  detector = {vp1._detector}")
    print(f"  tracker  = {vp1._tracker}")
    print(f"  stats    = {type(vp1._stats).__name__ if vp1._stats else None}")

    # ── 2. _build_ws_message format ───────────────────────────────────────────
    print("\n--- 2. _build_ws_message() ---")

    dummy_frame_data = {
        "frame_id":    42,
        "timestamp_ms": 1680,
        "quarter":     1,
        "tracking": {
            "tracked_players": [
                {"track_id": 1, "bbox": [10, 20, 50, 100],
                 "center": [30, 60], "confidence": 0.9},
            ],
            "tracked_ball": {
                "bbox": [200, 100, 230, 130],
                "center": [215, 115],
                "confidence": 0.95,
            },
            "tracked_referees": [],
        },
        "court":   {"is_calibrated": False},
        "pose":    {"poses": []},
        "actions": {"actions": [
            {"track_id": 1, "action": "Dribble",
             "action_id": 2, "confidence": 0.6, "source": "rule_based"},
        ]},
        "events":  [],
    }
    dummy_stats = {
        "player_stats": {
            1: {"name": "Arya", "team": "A",
                "mpi": {"avg_speed_kmh": 12.5, "mpi_composite": 65.0},
                "total_stats": {"pts": 10}},
        },
        "team_stats": {
            "A": {"pts": 10, "fga": 6, "fgm": 3, "fg_pct": 0.5},
            "B": {"pts": 8,  "fga": 5, "fgm": 2, "fg_pct": 0.4},
        },
        "possession_pct": {"A": 55.0, "B": 45.0},
        "mvp_ranking": [],
    }

    vp2 = VideoProcessor()
    msg = vp2._build_ws_message(dummy_frame_data, dummy_stats)

    required_top = ["type", "timestamp", "quarter", "gameClock",
                    "score", "possession", "players", "ball", "event"]
    for k in required_top:
        assert k in msg, f"Missing top-level key: {k!r}"

    assert msg["type"]             == "frame_update",   msg["type"]
    assert msg["quarter"]          == 1
    assert isinstance(msg["gameClock"], str)
    assert "teamA" in msg["score"]      and "teamB" in msg["score"]
    assert "teamA" in msg["possession"] and "teamB" in msg["possession"]
    assert msg["score"]["teamA"]        == 10
    assert msg["score"]["teamB"]        == 8
    assert msg["possession"]["teamA"]   == 55.0
    assert msg["possession"]["teamB"]   == 45.0
    assert msg["event"]                  is None
    assert msg["ball"]["bbox"]           == [200, 100, 230, 130]
    assert isinstance(msg["ball"]["trajectory"], list)

    assert len(msg["players"]) == 1
    p0 = msg["players"][0]
    for pk in ["trackId", "jerseyNumber", "name", "team",
               "bbox", "courtPos", "action", "speedKmh", "keypoints"]:
        assert pk in p0, f"Missing player key: {pk!r}"

    assert p0["trackId"]    == 1
    assert p0["name"]        == "Arya"
    assert p0["team"]        == "A"
    assert p0["action"]      == "Dribble"
    assert p0["speedKmh"]   == 12.5
    assert isinstance(p0["keypoints"], list)

    print(f"  type={msg['type']!r}  quarter={msg['quarter']}  clock={msg['gameClock']}")
    print(f"  score={msg['score']}  possession={msg['possession']}")
    print(f"  players={len(msg['players'])}  ball_bbox={msg['ball']['bbox']}")
    print("  All required keys present ✓")

    # ── 3. _throttle_fps timing ───────────────────────────────────────────────
    print("\n--- 3. _throttle_fps() ---")

    async def _test_throttle():
        vp = VideoProcessor()

        # 25 fps → 40 ms per frame; call with frame_start = now → full wait
        t0 = time.perf_counter()
        await vp._throttle_fps(25, t0)
        elapsed = time.perf_counter() - t0
        assert 0.030 <= elapsed <= 0.065, (
            f"Expected ~40ms sleep, got {elapsed*1000:.1f}ms"
        )
        print(f"  25fps throttle: slept {elapsed*1000:.1f}ms  (target 40ms)  ✓")

        # Frame already took 100ms → no sleep (100ms > 40ms target)
        t_old = time.perf_counter() - 0.100
        t_before = time.perf_counter()
        await vp._throttle_fps(25, t_old)
        elapsed2 = time.perf_counter() - t_before
        assert elapsed2 < 0.015, (
            f"Should not sleep, but slept {elapsed2*1000:.1f}ms"
        )
        print(f"  Slow frame → no sleep ({elapsed2*1000:.2f}ms)  ✓")

        # 0 fps guard → no division by zero
        t_zero = time.perf_counter() - 1.0
        await vp._throttle_fps(0, t_zero)   # should not raise
        print("  fps=0 → no crash  ✓")

    asyncio.run(_test_throttle())

    # ── 4. get_progress() return format ──────────────────────────────────────
    print("\n--- 4. get_progress() ---")
    vp4 = VideoProcessor()
    prog = vp4.get_progress()

    required_prog = ["frames_processed", "total_frames", "fps_actual",
                     "fps_dropped", "status", "quarter", "game_clock"]
    for k in required_prog:
        assert k in prog, f"Missing progress key: {k!r}"

    assert prog["status"]           == "idle"
    assert prog["frames_processed"] == 0
    assert prog["total_frames"]     == 0
    assert prog["quarter"]          == 1
    assert prog["game_clock"]       == "10:00"
    print(f"  {prog}")
    print("  All keys present and correct ✓")

    # ── 5. _process_frame with None → {} ─────────────────────────────────────
    print("\n--- 5. _process_frame(None) ---")
    vp5 = VideoProcessor()
    result = vp5._process_frame(None, frame_id=0, timestamp_ms=0)
    assert result == {}, f"Expected empty dict, got {result}"
    print("  _process_frame(None) → {} ✓")

    # With a real blank frame (no pipeline) → returns full skeleton
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    r2 = vp5._process_frame(blank, frame_id=7, timestamp_ms=280)
    assert r2["frame_id"]   == 7
    assert r2["timestamp_ms"] == 280
    assert r2["quarter"]    == 1
    assert isinstance(r2["events"], list)
    assert isinstance(r2["tracking"]["tracked_players"], list)
    print(f"  _process_frame(640×480 blank):  frame_id={r2['frame_id']}"
          f"  events={r2['events']}  ✓")

    # ── 6. stop() ─────────────────────────────────────────────────────────────
    print("\n--- 6. stop() ---")
    vp6 = VideoProcessor()
    vp6._status = "processing"
    vp6.stop()
    assert vp6._stop_signal.is_set(),  "stop_signal not set"
    assert vp6._status == "stopping",  f"status={vp6._status}"
    print("  stop() → stop_signal set, status='stopping'  ✓")

    # ── 7. _format_game_clock ─────────────────────────────────────────────────
    print("\n--- 7. _format_game_clock() ---")
    vp7 = VideoProcessor()
    vp7._source_fps          = 30.0
    vp7._quarter_start_frame = 0

    vp7._frame_count = 0
    assert vp7._format_game_clock() == "10:00", vp7._format_game_clock()
    print("  Frame 0     → 10:00  ✓")

    vp7._frame_count = 900   # 30fps × 30s elapsed
    assert vp7._format_game_clock() == "09:30", vp7._format_game_clock()
    print("  Frame 900   → 09:30  ✓")

    vp7._frame_count = 18000  # 30fps × 600s → time's up
    assert vp7._format_game_clock() == "00:00", vp7._format_game_clock()
    print("  Frame 18000 → 00:00  ✓")

    # ── 8. _detect_quarter_change ─────────────────────────────────────────────
    print("\n--- 8. _detect_quarter_change() ---")
    vp8 = VideoProcessor()
    vp8._source_fps          = 30.0
    vp8._quarter_start_frame = 0

    # 1 frame before end of quarter
    vp8._frame_count = QUARTER_DURATION_S * 30 - 1
    assert not vp8._detect_quarter_change({})
    print(f"  Frame {vp8._frame_count} (1 before end) → no change  ✓")

    # Exactly at end
    vp8._frame_count = QUARTER_DURATION_S * 30
    assert vp8._detect_quarter_change({})
    print(f"  Frame {vp8._frame_count} (at end)       → change  ✓")

    # Q4 → no more quarters
    vp8._current_quarter = 4
    assert not vp8._detect_quarter_change({})
    print("  Q4 → no further change  ✓")

    # OCR override path
    vp8._current_quarter = 2
    assert vp8._detect_quarter_change({"quarter_ocr": 3})
    print("  OCR override Q2→3     → change  ✓")

    # ── 9. Ball trajectory accumulation ──────────────────────────────────────
    print("\n--- 9. Ball trajectory ---")
    vp9 = VideoProcessor()
    fd1 = dict(dummy_frame_data)
    fd1["tracking"] = dict(dummy_frame_data["tracking"])
    fd1["tracking"]["tracked_ball"] = {
        "bbox":      [200, 100, 230, 130],
        "center":    [215, 115],
        "court_pos": [14.0, 7.5],
    }
    msg1 = vp9._build_ws_message(fd1, dummy_stats)
    assert msg1["ball"]["courtPos"] == [14.0, 7.5]
    assert len(vp9._ball_trajectory) == 1

    msg2 = vp9._build_ws_message(fd1, dummy_stats)
    assert len(vp9._ball_trajectory) == 2
    assert len(msg2["ball"]["trajectory"]) == 2
    print(f"  After 2 frames: trajectory len={len(msg2['ball']['trajectory'])}  ✓")

    print("\n=== All tests passed ===")
