from pydantic import BaseModel

class MatchCreate(BaseModel):
    team_a: str
    team_b: str
    category: str
    round: str

class MatchResponse(BaseModel):
    match_id: str
    team_a: str
    team_b: str
    category: str
    round: str
    status: str
