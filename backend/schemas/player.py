"""
Player-related Pydantic schemas.
"""

from pydantic import BaseModel, Field
from typing import List

class Player(BaseModel):
    track_id: int = Field(..., description="Tracking ID from ByteTrack")
    jersey_number: int = Field(..., ge=0, le=99, description="Jersey number 0-99")
    name: str = Field(..., min_length=1, description="Player name")
    team: str = Field(..., pattern="^[AB]$", description="Team A or B")
    action: str = Field("STAND", description="Current action label")
    speed_kmh: float = Field(default=0.0, ge=0)
    
class Roster(BaseModel):
    match_id: str
    team_a_name: str
    team_b_name: str
    players: List[Player]
