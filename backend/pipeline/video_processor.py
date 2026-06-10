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
import concurrent.futures
import csv
import json
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


def _fmt_wall_clock() -> str:
    """Current wall-clock time as HH:MM:SS — used in scoring logs for live context."""
    import datetime
    return datetime.datetime.now().strftime("%H:%M:%S")


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
PROCESS_STRIDE        = int(os.getenv("PROCESS_STRIDE",   "2"))   # 2=every other frame
COURT_EVERY_N_FRAMES  = int(os.getenv("COURT_EVERY_N",    "5"))   # court keypoint YOLOv8-pose
OCR_EVERY_N_FRAMES    = int(os.getenv("OCR_EVERY_N",     "15"))   # PaddleOCR
ACTION_EVERY_N_FRAMES = int(os.getenv("ACTION_EVERY_N",  "20"))   # SlowFast action

# ── Action label normalisation ────────────────────────────────────────────────
# Backend action.py returns Title-case ("Shoot", "Dribble", "Rebound_attempt").
# Frontend ActionLabel type uses UPPER ("SHOOT", "DRIBBLE", "REBOUND").
# Normalise here once so every output path (WS + snapshot) is consistent.
_ACTION_WS_MAP: dict[str, str] = {
    "shoot": "SHOOT", "pass": "PASS", "dribble": "DRIBBLE",
    "catch": "CATCH", "steal": "STEAL", "block": "BLOCK",
    "rebound_attempt": "REBOUND", "rebound": "REBOUND",
    "jump": "JUMP", "stand": "STAND", "jump_action": "JUMP",
}

def _norm_action(action_str: str) -> str:
    return _ACTION_WS_MAP.get((action_str or "stand").lower(), (action_str or "STAND").upper())


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
        self._frame_width:         int   = 0
        self._frame_height:        int   = 0
        self._roster:              dict  = {}

        # Court result cache — avoid running YOLOv8-pose every frame (camera is slow)
        self._last_court_result:  dict = {"is_calibrated": False}
        # Sticky courtPos cache: {track_id → [x,y]} — last known valid position per player
        self._last_court_pos:     dict = {}
        # Sticky team cache: {track_id → "A"|"B"} — persists even when OCR not running
        self._last_known_team:    dict = {}

        # Per-frame bbox store — written to JSON at finalize for frontend time-lookup
        self._frame_store:        list = []   # [{ts, p:[{i,j,t,b}], bl}]
        self._video_path:         str  = ""   # set in process_video, used in _finalize
        self._frames_json_path:   str  = ""   # output path written in _finalize
        self._output_video_path:  str  = ""   # rendered video with overlay
        self._output_csv_path:    str  = ""   # player stats CSV

        # Threading
        self._frame_queue = queue.Queue(maxsize=FRAME_QUEUE_MAXSIZE)
        self._stop_signal = threading.Event()
        self._executor    = concurrent.futures.ThreadPoolExecutor(max_workers=2,
                                thread_name_prefix="pipeline")

        # Data buffers
        self._frame_buffer:    deque = deque(maxlen=FRAME_BUFFER_SIZE)
        self._ball_trajectory: deque = deque(maxlen=BALL_TRAJ_MAXLEN)

        # Speed cache: track_id → avg_speed_kmh, updated every WS cycle (~5 frames)
        # Used by _snapshot_frame so the per-frame JSON has speed without calling get_live_stats() every frame.
        self._speed_cache:     dict  = {}   # {track_id: float}

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

        logger.info("Pipeline ready (device=%s)", dev)

    # ── Public async entry point ──────────────────────────────────────────────

    async def process_video(
        self,
        video_path: str,
        match_id:   str,
        roster:     Optional[dict] = None,
        ws_manager                 = None,
    ) -> None:
        """
        Process video end-to-end as an async coroutine (FastAPI BackgroundTask).
        Emits WebSocket updates every STATS_UPDATE_INTERVAL frames.
        """
        self._roster          = roster or {}
        self._status          = "processing"
        self._start_time      = time.perf_counter()
        self._stop_signal.clear()
        self._video_path      = str(video_path)
        self._frame_store     = []

        if self._stats is None:
            self._init_pipeline(self._models_path)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            self._status = "error"
            logger.error("Cannot open video: %s", video_path)
            return

        self._total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._source_fps   = cap.get(cv2.CAP_PROP_FPS) or FPS_DEFAULT
        self._frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 1920
        self._frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

        logger.info("Video: %s  frames=%d  src_fps=%.1f",
                    Path(video_path).name, self._total_frames, self._source_fps)

        grab_thread = threading.Thread(
            target=self._frame_grabber, args=(cap,),
            daemon=True, name="frame-grabber",
        )
        grab_thread.start()

        loop     = asyncio.get_event_loop()
        frame_id = 0       # source frame counter (every frame read from video)
        proc_id  = 0       # processed frame counter (after stride filter)

        try:
            while not self._stop_signal.is_set():
                frame_start = time.perf_counter()

                # Non-blocking poll so the event loop stays responsive
                try:
                    frame = self._frame_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.002)
                    continue

                if frame is None:   # sentinel — video exhausted
                    break

                # Skip intermediate frames on CPU to hit a reasonable throughput.
                # PROCESS_STRIDE=1 disables skipping entirely.
                if PROCESS_STRIDE > 1 and frame_id % PROCESS_STRIDE != 0:
                    frame_id += 1
                    self._frame_count = frame_id
                    continue

                ts_ms = int(frame_id / self._source_fps * 1000)

                # Heavy sync pipeline runs off the event loop
                frame_data = await loop.run_in_executor(
                    self._executor, self._process_frame, frame, proc_id, ts_ms
                )
                proc_id += 1

                # Store compact bbox snapshot — used by frontend for time-based replay lookup
                self._frame_store.append(self._snapshot_frame(frame_data, ts_ms))

                # Persist events
                scoring_types = {"MADE_FG", "MADE_3", "SHOT_MADE", "MADE_FT", "FT_MADE"}
                has_score_event = False
                for evt in frame_data.get("events", []):
                    if evt:
                        await self._save_event_mongo(evt, match_id)
                        if evt.get("type", "").upper() in scoring_types:
                            has_score_event = True

                # WebSocket update: every 5 processed frames OR immediately on scoring
                do_broadcast = (proc_id % 5 == 0) or has_score_event
                if do_broadcast and self._stats:
                    stats = self._stats.get_live_stats()
                    # Refresh speed cache so _snapshot_frame can embed speed in JSON
                    for tid, pstat in stats.get("player_stats", {}).items():
                        spd = pstat.get("mpi", {}).get("avg_speed_kmh", 0.0)
                        if spd > 0:
                            self._speed_cache[tid] = round(spd, 1)
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
                    if ws_manager:
                        try:
                            await ws_manager.broadcast(match_id, msg)
                        except Exception as e:
                            logger.debug("WS broadcast error: %s", e)

                frame_id += 1
                self._frame_count = frame_id
                elapsed = time.perf_counter() - self._start_time
                self._fps_actual = frame_id / elapsed if elapsed > 0 else 0.0

                if frame_id % FPS_LOG_INTERVAL == 0:
                    logger.info(
                        "Frame %d/%d (proc %d)  FPS=%.1f  dropped=%d  Q%d  stride=%d",
                        frame_id, self._total_frames, proc_id,
                        self._fps_actual, self._frames_dropped,
                        self._current_quarter, PROCESS_STRIDE,
                    )

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
            await self._finalize(match_id)
            logger.info("Processing complete: match=%s  frames=%d  status=%s",
                        match_id, frame_id, self._status)

    # ── Frame grabber (background thread) ────────────────────────────────────

    def _frame_grabber(self, cap: cv2.VideoCapture) -> None:
        """Read frames from cv2.VideoCapture; drop if pipeline queue is full."""
        try:
            while not self._stop_signal.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    # Block up to 10 s so frames aren't silently dropped on slow CPU.
                    # For live streams switch to put_nowait to avoid back-pressure.
                    self._frame_queue.put(frame, block=True, timeout=10.0)
                except queue.Full:
                    self._frames_dropped += 1
        except Exception as e:
            logger.warning("Frame grabber error: %s", e)
        finally:
            cap.release()
            # Sentinel so the main loop knows the stream ended
            try:
                self._frame_queue.put(None, timeout=2.0)
            except queue.Full:
                pass

    # ── Per-frame pipeline ────────────────────────────────────────────────────

    def _process_frame(
        self,
        frame:        np.ndarray,
        frame_id:     int,
        timestamp_ms: int,
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

        # ── 1. Detection ──────────────────────────────────────────────────
        if self._detector:
            try:
                detections = self._detector.detect(frame)
                frame_data["detections"] = detections
            except Exception as e:
                logger.debug("Detector frame %d: %s", frame_id, e)

        # ── 2. Tracking ───────────────────────────────────────────────────
        if self._tracker:
            try:
                tracking = self._tracker.update(frame, detections)
                frame_data["tracking"] = tracking
                tracked_players = tracking.get("tracked_players", [])
                ball_info       = tracking.get("tracked_ball")
            except Exception as e:
                logger.debug("Tracker frame %d: %s", frame_id, e)

        # ── 3. Court mapping ──────────────────────────────────────────────
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

        # ── 4. Pose estimation ────────────────────────────────────────────
        if self._pose and tracked_players:
            try:
                pose = self._pose.estimate(frame, tracked_players)
                frame_data["pose"] = pose
            except Exception as e:
                logger.debug("Pose frame %d: %s", frame_id, e)

        # ── 5. Jersey OCR — run every OCR_EVERY_N_FRAMES (jersey number is stable) ──
        if self._jersey and tracked_players and frame_id % OCR_EVERY_N_FRAMES == 0:
            try:
                jersey = self._jersey.process(frame, tracked_players, self._tracker)
                frame_data["jersey"] = jersey
            except Exception as e:
                logger.debug("Jersey OCR frame %d: %s", frame_id, e)

        # ── Annotate tracked_players with team from roster (roster is source of truth) ──
        if self._tracker and self._roster:
            for player in tracked_players:
                tid = player["track_id"]
                jersey_num = self._tracker.get_jersey_number(tid)
                if jersey_num is not None:
                    entry = self._roster.get(str(jersey_num), {})
                    if isinstance(entry, dict) and entry.get("team"):
                        player["team"] = entry["team"]
                        self._last_known_team[tid] = entry["team"]
                elif tid in self._last_known_team:
                    # OCR not running this frame — use last confirmed team
                    player["team"] = self._last_known_team[tid]

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

        # ── 7. Event engine ───────────────────────────────────────────────
        if self._events_engine:
            try:
                events = self._events_engine.process(frame_data)
                frame_data["events"] = events or []
            except Exception as e:
                logger.warning("EventEngine frame %d: %s", frame_id, e, exc_info=True)

        # ── 8. Stats update ───────────────────────────────────────────────
        if self._stats:
            tracking_snapshot: dict = {}
            for player in tracked_players:
                cp = player.get("court_pos")
                if cp:
                    tracking_snapshot[player["track_id"]] = (
                        cp[0], cp[1], timestamp_ms
                    )
                elif self._frame_width > 0 and self._frame_height > 0:
                    # Fallback: normalize pixel center to approximate court meters.
                    # court is 28m × 15m; pixel origin top-left → y is inverted.
                    center = player.get("center")
                    if center:
                        x_m = center[0] / self._frame_width  * 28.0
                        y_m = center[1] / self._frame_height * 15.0
                        tracking_snapshot[player["track_id"]] = (x_m, y_m, timestamp_ms)

            # Build stats-compatible roster: {track_id: {name, jersey_number, team}}
            # using tracker jersey map + raw upload roster (jersey_str → name)
            stats_roster: dict = {}
            if self._tracker and self._roster:
                for tid, jersey in self._tracker.track_jersey_map.items():
                    entry = self._roster.get(str(jersey), {})
                    if isinstance(entry, dict):
                        name = entry.get("name", f"#{jersey}")
                        team = entry.get("team", "")
                    else:
                        name = str(entry) if entry else f"#{jersey}"
                        team = ""
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
            action_label = action_map.get(tid, "")
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

        return {"ts": timestamp_ms, "p": players_out, "bl": ball_out}

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

            players.append({
                "trackId":      tid,
                "jerseyNumber": jersey_number,
                "name":         pstat.get("name", f"Player_{tid}"),
                "team":         player.get("team", "") or pstat.get("team", ""),
                "bbox":         norm_bbox,
                "courtPos":     cur_court_pos,
                "action":       _norm_action(act_entry.get("action", "Stand")),
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
            "event": latest_event,
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

        return {
            "type":       "event",
            "eventType":  event_type,
            "playerId":   jersey_num if jersey_num is not None else track_id,
            "playerName": (f"#{jersey_num}" if jersey_num is not None
                           else f"Track_{track_id}"),
            "team":       event.get("team", ""),
            "points":     points,
            "quarter":    event.get("quarter", self._current_quarter),
            "gameClock":  self._format_game_clock(),
            "courtPos":   event.get("court_pos"),
        }

    # ── Persistence helpers ───────────────────────────────────────────────────

    async def _save_event_mongo(self, event: dict, match_id: str) -> None:
        """Persist one event to MongoDB events collection (no-op if mongo absent)."""
        if self._mongo_db is None:
            return
        try:
            doc = {**event, "match_id": match_id, "saved_at": time.time()}
            self._mongo_db["events"].insert_one(doc)   # pymongo is sync — no await
        except Exception as e:
            logger.warning("MongoDB write error: %s", e)

    async def _push_redis(self, match_id: str, data: dict) -> None:
        """Publish data to Redis channel match:{match_id} (no-op if redis absent)."""
        if self._redis is None:
            return
        try:
            payload = json.dumps(data, default=str)
            result = self._redis.publish(f"match:{match_id}", payload)
            # redis.asyncio returns a coroutine; plain redis returns an int
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            logger.debug("Redis publish error: %s", e)

    async def _finalize(self, match_id: str) -> None:
        """Update MongoDB match status; push final stats; write output files."""

        # ── 1. Write per-frame bbox JSON (frontend time-lookup) ───────────────
        if self._frame_store and self._video_path:
            frames_path = os.path.splitext(self._video_path)[0] + "_frames.json"
            try:
                with open(frames_path, "w") as fp:
                    fp.write(json.dumps(self._frame_store, separators=(',', ':')))
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
                    None, self._render_output_video, out_dir, match_id
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

        # ── 5. Redis broadcast (always runs) ───────────────────────────────────
        if self._redis is not None and self._stats is not None:
            try:
                await self._push_redis(match_id, {
                    "type":              "processing_complete",
                    "status":            self._status,
                    "stats":             self._stats.get_live_stats(),
                    "output_video_path": self._output_video_path or None,
                    "output_csv_path":   self._output_csv_path   or None,
                })
            except Exception as e:
                logger.debug("Redis finalize error: %s", e)

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
            csv_path = os.path.join(out_dir, f"{match_id}_stats.csv")

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

    def _render_output_video(self, out_dir: str, match_id: str) -> Optional[str]:
        """
        Re-read the original video and draw bbox overlays from stored frame data.
        Writes to out_dir/{match_id}_analyzed.mp4.  Returns path or None on error.
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

            out_path = os.path.join(out_dir, f"{match_id}_analyzed.mp4")
            fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
            writer   = cv2.VideoWriter(out_path, fourcc, src_fps, (width, height))
            if not writer.isOpened():
                cap.release()
                logger.warning("_render_output_video: VideoWriter could not be opened")
                return None

            # Pre-build sorted timestamp list for O(log n) binary search per frame
            ts_list = [e["ts"] for e in self._frame_store]

            # BGR color palette (matches frontend CourtOverlay colors)
            C_A    = (212, 188,   0)   # Team A — cyan   #00BCD4
            C_B    = ( 22, 115, 249)   # Team B — orange #F97316
            C_UNK  = (175, 163, 156)   # unknown — gray  #9CA3AF
            C_BALL = ( 21, 204, 250)   # ball — yellow   #FACC15

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                ts_ms = int(frame_idx / src_fps * 1000)

                # Binary search: nearest stored snapshot within 2 s
                idx = bisect.bisect_left(ts_list, ts_ms)
                if idx >= len(ts_list):
                    idx = len(ts_list) - 1
                elif idx > 0 and abs(ts_list[idx - 1] - ts_ms) < abs(ts_list[idx] - ts_ms):
                    idx -= 1

                if abs(ts_list[idx] - ts_ms) < 2000:
                    snap = self._frame_store[idx]

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
                        label = f"#{p['j']}" if p.get("j") is not None else f"ID:{p['i']}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
                        cv2.putText(frame, label, (x1 + 3, y1 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                                    cv2.LINE_AA)

                    bl = snap.get("bl")
                    if bl:
                        b = bl.get("b", [])
                        if len(b) == 4:
                            x1, y1 = int(b[0] * width), int(b[1] * height)
                            x2, y2 = int(b[2] * width), int(b[3] * height)
                            if (x2 - x1) >= 4 and (y2 - y1) >= 4:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), C_BALL, 2)
                                cv2.putText(frame, "BALL", (x1, max(y1 - 4, 12)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_BALL, 1,
                                            cv2.LINE_AA)

                writer.write(frame)
                frame_idx += 1

            cap.release()
            writer.release()
            logger.info("Output video: %s  (frames=%d)", out_path, frame_idx)
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
        # If frame_data carries an authoritative quarter from OCR scoreboard
        ocr_q = frame_data.get("quarter_ocr")
        if ocr_q and ocr_q != self._current_quarter:
            return True

        if self._current_quarter >= 4:
            return False

        frames_in_q = self._frame_count - self._quarter_start_frame
        elapsed_s   = frames_in_q / max(self._source_fps, 1.0)
        return elapsed_s >= QUARTER_DURATION_S

    def _handle_quarter_change(self) -> None:
        self._current_quarter    += 1
        self._quarter_start_frame = self._frame_count
        logger.info("Quarter %d started (frame=%d)",
                    self._current_quarter, self._frame_count)

        for component, method in [
            (self._tracker,       "reset"),
            (self._events_engine, "reset"),
            (self._action,        "reset"),
        ]:
            if component and hasattr(component, method):
                try:
                    getattr(component, method)()
                except Exception as e:
                    logger.warning("%s.reset() error: %s",
                                   type(component).__name__, e)

        if self._stats:
            try:
                self._stats.reset_quarter(self._current_quarter)
            except Exception as e:
                logger.warning("stats.reset_quarter error: %s", e)

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

    def get_progress(self) -> dict:
        return {
            "frames_processed": self._frame_count,
            "total_frames":     self._total_frames,
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
