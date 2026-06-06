from pydantic import BaseModel
from typing import List, Optional

class PlayerCreate(BaseModel):
    jersey_number: int
    name: str
    team: str  # "A" or "B"

class MatchCreate(BaseModel):
    team_a: str
    team_b: str
    category: str  # "Men's" or "Women's"
    round: str

class StatsSnapshot(BaseModel):
    quarter: int
    game_clock: str
    score_a: int
    score_b: int
    possession_a: float
    possession_b: float
