"""
Event endpoints and WebSocket:
- GET /events — Event log
- WS /ws/live — Real-time frame updates
- WS /ws/events — Real-time event stream
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
import json
from datetime import datetime
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

router = APIRouter()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = MongoClient(MONGO_URL)
db = client["smart_vision_basketball"]

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@router.get("/events")
async def get_events(match_id: str = None, limit: int = 100):
    """
    Get event log from MongoDB.
    """
    try:
        query = {}
        if match_id:
            query["match_id"] = match_id

        events = list(db.events.find(query, {"_id": 0}).sort("timestamp", -1).limit(limit))

        return {
            "match_id": match_id,
            "count": len(events),
            "events": events
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/live")
async def websocket_live_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time frame updates.
    Broadcasts player positions, ball state, and court overlay data.
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Save to MongoDB if valid
            if "type" in message:
                db.frame_updates.insert_one({
                    "timestamp": datetime.now(),
                    **message
                })

            # Broadcast to all connected clients
            await manager.broadcast({
                "type": "frame_update",
                "data": message,
                "timestamp": datetime.now().isoformat()
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.close(code=1000, reason=str(e))

@router.websocket("/ws/events")
async def websocket_events_endpoint(websocket: WebSocket):
    """
    WebSocket for real-time game events.
    Broadcasts scoring, fouls, turnovers, etc.
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Save to MongoDB
            db.events.insert_one({
                "timestamp": datetime.now(),
                **message
            })

            # Broadcast to all connected clients
            await manager.broadcast({
                "type": "game_event",
                "event": message,
                "timestamp": datetime.now().isoformat()
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        await websocket.close(code=1000, reason=str(e))
