"""
Statistics endpoints:
- GET /stats/live — Real-time stats snapshot
- GET /stats/player/{player_id} — Per-player stats
- GET /stats/team/{team_id} — Per-team stats
- GET /stats/quarter/{q} — Per-quarter stats
"""

from fastapi import APIRouter, HTTPException
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

router = APIRouter()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = MongoClient(MONGO_URL)
db = client["smart_vision_basketball"]

@router.get("/stats/live")
async def get_live_stats():
    """
    Get current live stats snapshot from Redis.
    """
    try:
        latest_stats = db.player_stats.find().sort("_id", -1).limit(1)
        stats_list = list(latest_stats)

        if not stats_list:
            return {
                "timestamp": None,
                "quarter": None,
                "game_clock": None,
                "score": {"team_a": 0, "team_b": 0},
                "possession": {"team_a": 0.5, "team_b": 0.5},
                "message": "Waiting for video processing..."
            }

        return {
            "timestamp": stats_list[0].get("updated_at"),
            "quarter": stats_list[0].get("quarter"),
            "game_clock": None,
            "score": {"team_a": 0, "team_b": 0},
            "message": "Stats available"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
