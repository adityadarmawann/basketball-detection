from pydantic import BaseModel
from typing import List, Optional

class EventCreate(BaseModel):
    match_id: str
    event_type: str
    player_id: int
    team: str
    quarter: int
    game_clock: str
    court_pos: List[float]

class EventResponse(BaseModel):
    id: str
    match_id: str
    event_type: str
    player_id: int
    team: str
    quarter: int
    game_clock: str
