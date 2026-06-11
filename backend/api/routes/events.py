"""
Event endpoints and WebSocket:
  GET  /events                 — event log from MongoDB
  POST /events/correct         — manual correction for unidentified player events
  WS   /ws/live                — real-time frame updates (Redis channel match:{match_id})
  WS   /ws/events              — real-time game events  (Redis channel events:{match_id})
"""

import asyncio
import json
import logging
import os
import time

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

# ── WebSocket manager (imported by upload.py for pipeline broadcasts) ─────────

from api.websocket import WebSocketManager

manager = WebSocketManager()

# ── MongoDB (sync pymongo — read-only queries only) ───────────────────────────

try:
    from pymongo import MongoClient
    _MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    _mongo_client = MongoClient(_MONGO_URL, serverSelectionTimeoutMS=2000)
    _mongo_db = _mongo_client["smart_vision_basketball"]
except Exception as e:
    _mongo_db = None
    logger.warning("MongoDB unavailable: %s", e)

# ── Redis async client ────────────────────────────────────────────────────────

try:
    from db.redis_client import get_redis as _get_redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    async def _get_redis():
        return None
    logger.warning("Redis unavailable — WS will not receive pipeline updates")


# ── Helper: run Redis listener alongside WebSocket receive loop ───────────────

async def _redis_listener(match_id: str, channel: str, ws_manager: WebSocketManager):
    """
    Subscribe to a Redis channel and forward every message to all WebSocket
    clients watching match_id.  Cancellation-safe.
    """
    redis = None
    try:
        redis = await _get_redis()
    except Exception as e:
        logger.warning("Redis connect error: %s", e)

    await ws_manager.listen_redis(match_id, channel=channel, redis_client=redis)


# ── Manual correction models ──────────────────────────────────────────────────

class EventCorrection(BaseModel):
    match_id:     str
    track_id:     int    # ByteTrack ID of the unidentified player
    team:         str    # "A" or "B"
    jersey_number: int   # jersey number of the real player
    player_name:  str    # display name from roster


def _patch_stats_blob(blob: dict, c: EventCorrection) -> bool:
    """
    Mutate blob in-place: rename the unidentified player entry and fix
    team_stats + score attribution.  Returns True if a match was found.
    """
    player_stats = blob.get("player_stats", {})

    # Find the entry for this track_id (key is string in JSON)
    target_key = None
    for k, p in player_stats.items():
        jn = p.get("jersey_number")
        nm = p.get("name", "")
        # Match by null/zero jersey AND name matching Player_{track_id}
        if (jn is None or jn == 0) and nm == f"Player_{c.track_id}":
            target_key = k
            break

    if target_key is None:
        return False

    entry    = player_stats[target_key]
    old_team = entry.get("team", "")
    ts       = entry.get("total_stats") or {}
    pts      = int(ts.get("pts", 0))

    # Update player identity
    entry["jersey_number"] = c.jersey_number
    entry["name"]          = c.player_name
    entry["team"]          = c.team

    # Fix team_stats
    team_stats = blob.setdefault("team_stats", {})
    def _ts_add(team: str, field: str, delta: int) -> None:
        team_stats.setdefault(team, {})[field] = (
            max(0, team_stats.get(team, {}).get(field, 0) + delta)
        )

    # Move ALL countable stats from old team → new team
    TEAM_FIELDS = ("pts", "fgm", "fga", "reb", "ast", "stl", "blk", "tov")
    for field in TEAM_FIELDS:
        val = int(ts.get(field, 0))
        if val and old_team in ("A", "B") and old_team != c.team:
            _ts_add(old_team, field, -val)
            _ts_add(c.team,   field,  val)
        elif val and old_team not in ("A", "B"):
            _ts_add(c.team, field, val)

    # Fix running score
    score = blob.setdefault("score", {"team_a": 0, "team_b": 0})
    if pts > 0:
        if old_team not in ("A", "B"):
            key = "team_a" if c.team == "A" else "team_b"
            score[key] = score.get(key, 0) + pts
        elif old_team != c.team:
            old_key = "team_a" if old_team == "A" else "team_b"
            new_key = "team_a" if c.team   == "A" else "team_b"
            score[old_key] = max(0, score.get(old_key, 0) - pts)
            score[new_key] = score.get(new_key, 0) + pts

    return True


# ── POST /events/correct ──────────────────────────────────────────────────────

@router.post("/events/correct")
async def correct_event(correction: EventCorrection):
    """
    Manually assign an unidentified player's accumulated stats to a known
    player.  Patches both the running stats Redis key and all per-quarter keys,
    then broadcasts jersey_confirmed so connected clients update live.
    """
    patched_any = False

    try:
        redis = await _get_redis()
        if redis is None:
            raise HTTPException(status_code=503, detail="Redis unavailable")

        # Patch running stats + per-quarter blobs
        for suffix in ["", ":q1", ":q2", ":q3", ":q4"]:
            raw = await redis.get(f"stats:{correction.match_id}{suffix}")
            if not raw:
                continue
            blob = json.loads(raw)
            if _patch_stats_blob(blob, correction):
                patched_any = True
            await redis.set(
                f"stats:{correction.match_id}{suffix}",
                json.dumps(blob),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.warning("correct_event Redis error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # Persist correction record to MongoDB (non-fatal if unavailable)
    if _mongo_db is not None:
        try:
            _mongo_db.event_corrections.insert_one({
                **correction.model_dump(),
                "corrected_at": time.time(),
            })
        except Exception as e:
            logger.warning("MongoDB correction write: %s", e)

    # Broadcast jersey_confirmed so all WS clients patch their event log
    await manager.broadcast(correction.match_id, {
        "type":         "jersey_confirmed",
        "trackId":      correction.track_id,
        "jerseyNumber": correction.jersey_number,
        "playerName":   correction.player_name,
        "team":         correction.team,
    })

    return {"status": "ok", "patched": patched_any}


# ── GET /events ───────────────────────────────────────────────────────────────

@router.get("/events")
async def get_events(
    match_id: str = Query(None),
    limit:    int = Query(100, ge=1, le=1000),
):
    """Return event log from MongoDB (newest first)."""
    if _mongo_db is None:
        raise HTTPException(status_code=503, detail="MongoDB unavailable")

    try:
        query = {}
        if match_id:
            query["match_id"] = match_id

        events = list(
            _mongo_db.events
            .find(query, {"_id": 0})
            .sort("timestamp", -1)
            .limit(limit)
        )
        return {"match_id": match_id, "count": len(events), "events": events}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── WS /ws/live ───────────────────────────────────────────────────────────────

@router.websocket("/ws/live")
async def websocket_live(websocket: WebSocket, match_id: str = Query(...)):
    """
    Real-time frame updates.

    Query param  : match_id (required)
    Push source  : Redis channel  match:{match_id}
                   (published by VideoProcessor every STATS_UPDATE_INTERVAL frames)
    Pull source  : client can also send JSON messages that get echoed to all
                   other subscribers (for manual overrides / debug)
    """
    await manager.connect(websocket, match_id)

    # Redis listener runs concurrently while we wait for client messages
    redis_task = asyncio.create_task(
        _redis_listener(match_id, f"match:{match_id}", manager),
        name=f"redis-live-{match_id}",
    )

    try:
        while True:
            raw = await websocket.receive_text()
            # Forward any client-sent messages to all subscribers of the same match
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"raw": raw}
            await manager.broadcast(match_id, msg)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WS live error match=%s: %s", match_id, e)
    finally:
        redis_task.cancel()
        await manager.disconnect(websocket, match_id)


# ── WS /ws/events ─────────────────────────────────────────────────────────────

@router.websocket("/ws/events")
async def websocket_events(websocket: WebSocket, match_id: str = Query(...)):
    """
    Real-time game events (scores, fouls, rebounds, etc.).

    Query param  : match_id (required)
    Push source  : Redis channel  events:{match_id}
                   (published by EventEngine when an event fires)
    Pull source  : client can send event JSON that gets persisted to MongoDB
                   and broadcast to all match subscribers
    """
    await manager.connect(websocket, match_id)

    redis_task = asyncio.create_task(
        _redis_listener(match_id, f"events:{match_id}", manager),
        name=f"redis-events-{match_id}",
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                event = json.loads(raw)
            except json.JSONDecodeError:
                logger.debug("WS events: invalid JSON from client")
                continue

            # Persist to MongoDB
            if _mongo_db is not None:
                try:
                    _mongo_db.events.insert_one({"match_id": match_id, **event})
                except Exception as e:
                    logger.warning("MongoDB event write: %s", e)

            # Broadcast to all match subscribers
            await manager.broadcast(match_id, {"type": "game_event", **event})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.debug("WS events error match=%s: %s", match_id, e)
    finally:
        redis_task.cancel()
        await manager.disconnect(websocket, match_id)
