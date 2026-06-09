"""
Video upload endpoint — POST /api/upload-video
Triggers background processing via VideoProcessor after upload.

Additional endpoints:
  GET  /api/upload/progress/{match_id}  — poll processing progress
  POST /api/upload/stop/{match_id}      — request graceful stop
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_PATH   = os.getenv("UPLOAD_PATH", "./uploads")
MODELS_PATH   = os.getenv("MODELS_PATH", os.path.join(
    os.path.dirname(__file__), "..", "..", "models"
))
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024   # 5 GB
CHUNK_SIZE    = 1 * 1024 * 1024           # 1 MB streaming chunks

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# ── Pipeline imports (graceful — not available in dev without GPU) ────────────

try:
    from pipeline.video_processor import create_video_processor
    _VP_AVAILABLE = True
except ImportError:
    create_video_processor = None
    _VP_AVAILABLE = False
    logger.warning("VideoProcessor unavailable — processing will be skipped")

try:
    from db.mongo import get_db
except ImportError:
    def get_db():
        return None
    logger.warning("db.mongo unavailable — MongoDB will be skipped")

try:
    from db.redis_client import get_redis
except ImportError:
    async def get_redis():
        return None
    logger.warning("db.redis_client unavailable — Redis will be skipped")

try:
    from api.routes.events import manager as _ws_manager
except ImportError:
    _ws_manager = None
    logger.warning("WebSocket manager unavailable")

# ── Per-match processor registry ──────────────────────────────────────────────

# dict[match_id, VideoProcessor]  — populated by _run_pipeline before processing starts
_processors: dict = {}


# ── Background pipeline helper ────────────────────────────────────────────────

async def _run_pipeline(video_path: str, match_id: str, roster: dict) -> None:
    """
    Async background task: wires DB/Redis, creates VideoProcessor, runs pipeline.
    Registered in _processors before process_video() starts so that the progress
    endpoint can respond immediately after the first poll.
    """
    db    = get_db()                        # sync pymongo db (or None)
    redis = None
    try:
        redis = await get_redis()
    except Exception as e:
        logger.warning("Redis unavailable, skipping: %s", e)

    processor = create_video_processor(
        models_path=MODELS_PATH,
        mongo_db=db,
        redis_client=redis,
    )
    _processors[match_id] = processor      # register before blocking call

    try:
        await processor.process_video(
            video_path=video_path,
            match_id=match_id,
            roster=roster,
            ws_manager=_ws_manager,
        )
    except Exception as e:
        logger.error("Pipeline error match_id=%s: %s", match_id, e, exc_info=True)


# ── POST /upload-video ────────────────────────────────────────────────────────

@router.post("/upload-video")
async def upload_video(
    background_tasks: BackgroundTasks,
    file:     UploadFile    = File(...),
    match_id: str           = Form(...),
    roster:   Optional[str] = Form(None),   # JSON: {"7": "Bima", "12": "Arya"}
):
    """
    Upload a video file and immediately start background analytics processing.

    Form fields
    -----------
    file       : video file (.mp4 / .mov / .avi / .mkv, max 5 GB)
    match_id   : unique match identifier (e.g. "match_20240606_001")
    roster     : optional JSON string mapping jersey_number → player_name
    """
    # ── Validate extension ─────────────────────────────────────────────────
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # ── Parse roster ───────────────────────────────────────────────────────
    # Accepts two formats:
    #   new: { "7": {"name": "Bima", "team": "A"}, ... }
    #   old: { "7": "Bima", ... }  (legacy — team defaults to "")
    roster_data: dict = {}
    if roster:
        try:
            raw = json.loads(roster)
            for jersey, val in raw.items():
                if isinstance(val, dict):
                    roster_data[jersey] = {"name": val.get("name", ""), "team": val.get("team", "")}
                else:
                    roster_data[jersey] = {"name": str(val), "team": ""}
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid roster JSON: {e}")

    # ── Save file (chunked streaming — avoids loading 5 GB into RAM) ───────
    os.makedirs(UPLOAD_PATH, exist_ok=True)

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name       = os.path.basename(file.filename or "video")
    unique_filename = f"{timestamp}_{match_id}_{safe_name}"
    saved_path      = os.path.join(UPLOAD_PATH, unique_filename)

    total_bytes = 0
    try:
        with open(saved_path, "wb") as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE:
                    f.close()
                    os.remove(saved_path)
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 5 GB limit",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("File save error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")

    logger.info(
        "Upload complete: match_id=%s  file=%s  size=%.1f MB",
        match_id, unique_filename, total_bytes / 1_048_576,
    )

    # ── Trigger background processing ──────────────────────────────────────
    if _VP_AVAILABLE:
        background_tasks.add_task(
            _run_pipeline,
            video_path=str(saved_path),
            match_id=match_id,
            roster=roster_data,
        )
        status  = "processing"
        message = "Video uploaded, pipeline started"
    else:
        status  = "uploaded"
        message = "Video uploaded. Processing unavailable (dev mode — VideoProcessor not loaded)."

    return JSONResponse(
        status_code=200,
        content={
            "match_id":    match_id,
            "video_id":    timestamp,
            "filename":    file.filename,
            "stored_as":   unique_filename,
            "size_bytes":  total_bytes,
            "upload_time": datetime.now().isoformat(),
            "status":      status,
            "message":     message,
        },
    )


# ── GET /upload/progress/{match_id} ──────────────────────────────────────────

@router.get("/upload/progress/{match_id}")
async def get_processing_progress(match_id: str):
    """
    Poll the processing progress of a running video analysis.

    Returns
    -------
    {
      "match_id":         str,
      "frames_processed": int,
      "total_frames":     int,
      "fps_actual":       float,
      "status":           "processing" | "done" | "error" | "stopping",
      "quarter":          int,
      "game_clock":       "MM:SS"
    }

    404 — match_id not registered (upload first, or pipeline hasn't started yet)
    503 — VideoProcessor unavailable in this environment
    """
    if not _VP_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="VideoProcessor not available in this environment",
        )

    processor = _processors.get(match_id)
    if processor is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No active processing session for match_id='{match_id}'. "
                "Upload a video first, or wait a moment for the pipeline to initialise."
            ),
        )

    progress = processor.get_progress()
    return {
        "match_id":         match_id,
        "frames_processed": progress["frames_processed"],
        "total_frames":     progress["total_frames"],
        "fps_actual":       progress["fps_actual"],
        "status":           progress["status"],
        "quarter":          progress["quarter"],
        "game_clock":       progress["game_clock"],
    }


# ── POST /upload/stop/{match_id} ─────────────────────────────────────────────

@router.post("/upload/stop/{match_id}")
async def stop_processing(match_id: str):
    """
    Request graceful stop of a running analysis.
    Poll /upload/progress/{match_id} until status == 'done'.
    """
    if not _VP_AVAILABLE:
        raise HTTPException(status_code=503, detail="VideoProcessor unavailable")

    processor = _processors.get(match_id)
    if processor is None:
        raise HTTPException(
            status_code=404,
            detail=f"No active processing session for match_id='{match_id}'",
        )

    processor.stop()
    return {
        "match_id": match_id,
        "status":   "stopping",
        "message":  "Stop signal sent. Processing will finish the current frame then exit.",
    }
