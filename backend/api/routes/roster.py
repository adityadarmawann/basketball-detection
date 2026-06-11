"""
Roster endpoints:
- POST /roster — Add players to roster
- GET /roster/{match_id} — Get match roster
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from pathlib import Path
import os, uuid
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

router = APIRouter()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = MongoClient(MONGO_URL)
db = client["smart_vision_basketball"]

PHOTOS_DIR = Path(__file__).parent.parent.parent / "uploads" / "photos"
PHOTOS_DIR.mkdir(parents=True, exist_ok=True)

class PlayerInput(BaseModel):
    jersey_number: int
    name: str
    team: str  # "A" or "B"

class RosterInput(BaseModel):
    match_id: str
    players: List[PlayerInput]

@router.post("/roster")
async def add_roster(roster: RosterInput):
    """
    Add players to match roster.
    """
    try:
        # Validate
        if not roster.match_id:
            raise HTTPException(status_code=400, detail="match_id required")
        
        if not roster.players:
            raise HTTPException(status_code=400, detail="At least one player required")

        # Check for duplicate jersey numbers per team
        for team in ['A', 'B']:
            jerseys = [p.jersey_number for p in roster.players if p.team == team]
            if len(jerseys) != len(set(jerseys)):
                raise HTTPException(status_code=400, detail="Duplicate jersey numbers")

        # Insert players
        player_docs = []
        for i, p in enumerate(roster.players):
            player_docs.append({
                "match_id": roster.match_id,
                "jersey_number": p.jersey_number,
                "name": p.name,
                "team": p.team,
                "track_id": None,  # Will be assigned during tracking
                "created_at": datetime.now()
            })

        if player_docs:
            db.players.insert_many(player_docs)

        # Update match status
        db.matches.update_one(
            {"match_id": roster.match_id},
            {
                "$set": {
                    "status": "roster_added",
                    "updated_at": datetime.now()
                }
            }
        )

        return {
            "match_id": roster.match_id,
            "players_added": len(roster.players),
            "message": "Roster saved successfully"
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/roster/{match_id}/player/{jersey_number}/photo")
async def upload_player_photo(
    match_id: str,
    jersey_number: int,
    team: str = Query(..., description="'A' or 'B'"),
    file: UploadFile = File(...),
):
    """Upload / replace photo for a specific player."""
    ext = Path(file.filename).suffix.lower() if file.filename else ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(status_code=400, detail="Only jpg/png/webp allowed")

    filename = f"{match_id}_{team}_{jersey_number}_{uuid.uuid4().hex[:8]}{ext}"
    dest = PHOTOS_DIR / filename
    content = await file.read()
    dest.write_bytes(content)

    photo_url = f"/api/media/photos/{filename}"
    db.players.update_one(
        {"match_id": match_id, "jersey_number": jersey_number, "team": team},
        {"$set": {"photo_url": photo_url, "updated_at": datetime.now()}},
    )
    return {"photo_url": photo_url}


@router.get("/roster/{match_id}")
async def get_roster(match_id: str):
    """
    Get roster for a match.
    """
    try:
        players = list(db.players.find(
            {"match_id": match_id},
            {"_id": 0}
        ))

        return {
            "match_id": match_id,
            "count": len(players),
            "players": players,
            "message": f"{len(players)} players found"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
