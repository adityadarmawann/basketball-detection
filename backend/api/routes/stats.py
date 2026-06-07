"""
Statistics endpoints:
- GET /stats/live              — Real-time stats snapshot
- GET /stats/player/{id}       — Per-player stats
- GET /stats/team/{team_id}    — Per-team stats
- GET /stats/quarter/{q}       — Per-quarter stats

MPI endpoints:
- GET /mpi/team                — All players MPI for a match (Redis)
- GET /mpi/{player_id}         — Single player MPI (Redis)
"""

import json
import logging

from fastapi import APIRouter, HTTPException, Query
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = MongoClient(MONGO_URL)
db = client["smart_vision_basketball"]

from db.redis_client import get_redis

# ── Default empty-stats structure returned when no live data is available ──────

def _empty_team_stats() -> dict:
    return {"pts": 0, "reb": 0, "ast": 0, "stl": 0,
            "blk": 0, "tov": 0, "fgm": 0, "fga": 0, "fg_pct": 0.0}

def _default_response() -> dict:
    return {
        "player_stats":  {},
        "team_stats":    {"A": _empty_team_stats(), "B": _empty_team_stats()},
        "mvp_ranking":   [],
        "possession_pct": {"A": 0.5, "B": 0.5},
        "score":         {"team_a": 0, "team_b": 0},
        "quarter":       1,
        "game_clock":    "10:00",
    }


@router.get("/stats/live")
async def get_live_stats(match_id: str = Query(None)):
    """
    Get current live stats snapshot.

    Priority:
      1. Redis key  stats:{match_id}  (written by VideoProcessor every N frames)
      2. Default empty structure      (never returns a plain message string)

    Query params
    ------------
    match_id : str, optional
        Match identifier. If omitted returns default structure.
    """
    if not match_id:
        return _default_response()

    # ── 1. Try Redis ──────────────────────────────────────────────────────────
    try:
        redis = await get_redis()
        if redis is not None:
            raw = await redis.get(f"stats:{match_id}")
            if raw:
                data = json.loads(raw)
                # Ensure all required top-level keys are present
                base = _default_response()
                base.update(data)
                return base
    except Exception as e:
        logger.warning("Redis read error for match %s: %s", match_id, e)

    # ── 2. Fallback: return default structure ─────────────────────────────────
    return _default_response()

# ── MPI helpers ───────────────────────────────────────────────────────────────

def _default_mpi_player() -> dict:
    return {
        "track_id":        0,
        "name":            "",
        "jersey_number":   0,
        "team":            "A",
        "mpi_composite":   0.0,
        "distance_m":      0.0,
        "avg_speed_kmh":   0.0,
        "max_speed_kmh":   0.0,
        "jump_height_cm":  0.0,
        "agility_score":   0.0,
        "endurance_score": 0.0,
        "fatigue_score":   0.0,
    }


async def _fetch_mpi_players(match_id: str) -> list:
    """
    Read MPI player list from Redis key  mpi:{match_id}.
    Accepts both list and {track_id: player_dict} storage formats.
    Returns [] on any failure (Redis unavailable, key missing, bad JSON).
    """
    try:
        redis = await get_redis()
        if redis is None:
            return []
        raw = await redis.get(f"mpi:{match_id}")
        if not raw:
            return []
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return list(data.values())
        return []
    except Exception as e:
        logger.warning("Redis MPI read error match=%s: %s", match_id, e)
        return []


# ── MPI endpoints — /mpi/team MUST be before /mpi/{player_id} ────────────────
# FastAPI matches routes top-to-bottom; the literal path "team" must be
# registered first so it isn't captured by the {player_id} path parameter.

@router.get("/mpi/team")
async def get_team_mpi(match_id: str = Query(...)):
    """
    Get MPI physical metrics for all players in a match.

    Reads from Redis key  mpi:{match_id}  (written by VideoProcessor after
    each stats_calculator update).  Returns an empty players list when the
    key is absent or Redis is unavailable — never raises 500.

    Query params
    ------------
    match_id : str  (required)
    """
    players = await _fetch_mpi_players(match_id)
    return {
        "match_id": match_id,
        "players":  players,
    }


@router.get("/mpi/{player_id}")
async def get_player_mpi(
    player_id: int,
    match_id:  str = Query(...),
):
    """
    Get MPI physical metrics for a single player.

    Reads from Redis key  mpi:{match_id}  and filters by
    track_id == player_id.

    Query params
    ------------
    match_id : str  (required)

    Raises 404 when the player is not found in the MPI data.
    """
    players = await _fetch_mpi_players(match_id)
    for player in players:
        if player.get("track_id") == player_id:
            return player
    raise HTTPException(
        status_code=404,
        detail=f"No MPI data for player_id={player_id} in match '{match_id}'",
    )


# ── Existing per-player / per-team / per-quarter endpoints ────────────────────

@router.get("/stats/player/{player_id}")
async def get_player_stats(player_id: int):
    """
    Get aggregated stats for a specific player.
    """
    try:
        player_stats = list(db.player_stats.find(
            {"player_id": player_id},
            {"_id": 0}
        ).sort("quarter", 1))

        if not player_stats:
            return {
                "player_id": player_id,
                "message": "No stats found"
            }

        return {
            "player_id": player_id,
            "count": len(player_stats),
            "stats": player_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/team/{team_id}")
async def get_team_stats(team_id: str):
    """
    Get aggregated stats for a team (A or B).
    """
    try:
        team_stats = list(db.player_stats.find(
            {"team": team_id},
            {"_id": 0}
        ).sort("quarter", 1))

        if not team_stats:
            return {"team_id": team_id, "message": "No stats found"}

        return {
            "team_id": team_id,
            "count": len(team_stats),
            "stats": team_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats/quarter/{quarter}")
async def get_quarter_stats(quarter: int):
    """
    Get stats for a specific quarter (1-4).
    """
    if quarter not in [1, 2, 3, 4]:
        raise HTTPException(status_code=400, detail="Quarter must be 1-4")
    
    try:
        quarter_stats = list(db.player_stats.find(
            {"quarter": quarter},
            {"_id": 0}
        ))

        return {
            "quarter": quarter,
            "count": len(quarter_stats),
            "stats": quarter_stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
