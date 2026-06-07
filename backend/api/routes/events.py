"""
Event endpoints and WebSocket:
  GET  /events            — event log from MongoDB
  WS   /ws/live           — real-time frame updates (Redis channel match:{match_id})
  WS   /ws/events         — real-time game events  (Redis channel events:{match_id})
"""

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect

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
