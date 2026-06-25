"""
Match endpoints:
- POST /match — Create new match
- GET /match/{match_id} — Get match details
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
from bson import ObjectId
import os
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Import database
from pymongo import MongoClient
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = MongoClient(MONGO_URL)
db = client["smart_vision_basketball"]

class MatchInput(BaseModel):
    team_a: str
    team_b: str
    category: str
    round: str
    region: str = ""

@router.post("/match")
async def create_match(match: MatchInput):
    """
    Create a new match.
    """
    try:
        # Validate
        if not match.team_a or not match.team_b:
            raise HTTPException(status_code=400, detail="Team names required")

        if match.team_a == match.team_b:
            raise HTTPException(status_code=400, detail="Teams must be different")

        # Create match document
        match_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        match_doc = {
            "match_id": match_id,
            "team_a": {"name": match.team_a, "color": "#00BCD4"},
            "team_b": {"name": match.team_b, "color": "#1A1A2E"},
            "category": match.category,
            "round": match.round,
            "region": match.region,
            "status": "setup",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }

        # Save to MongoDB
        db.matches.insert_one(match_doc)

        return {
            "match_id": match_id,
            "team_a": match.team_a,
            "team_b": match.team_b,
            "category": match.category,
            "round": match.round,
            "region": match.region,
            "status": "setup",
            "created_at": datetime.now().isoformat(),
            "message": "Match created successfully"
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/match/{match_id}")
async def get_match(match_id: str):
    """
    Get match details.
    """
    try:
        match_doc = db.matches.find_one({"match_id": match_id})

        if not match_doc:
            raise HTTPException(status_code=404, detail="Match not found")

        return {
            "match_id": match_doc["match_id"],
            "team_a": match_doc["team_a"]["name"],
            "team_b": match_doc["team_b"]["name"],
            "category": match_doc.get("category"),
            "round": match_doc.get("round"),
            "region": match_doc.get("region", ""),
            "status": match_doc.get("status"),
            "created_at": match_doc["created_at"].isoformat() if "created_at" in match_doc else None
        }

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
