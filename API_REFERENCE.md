# 📡 Smart Vision Basketball Analytics — API Reference

## Backend Endpoints (Phase 1 - Ready to Test)

All endpoints are implemented with boilerplate code and ready to extend.

### Health Check
```
GET /health
Response:
{
  "status": "ok",
  "service": "smart-vision-api"
}
```

### Match Management

#### Create Match
```
POST /api/match
Content-Type: application/json

Request Body:
{
  "team_a": "UI Negeri",
  "team_b": "Binus Legends",
  "category": "Men's",
  "round": "Final"
}

Response:
{
  "match_id": "20240605_153045",
  "team_a": "UI Negeri",
  "team_b": "Binus Legends",
  "category": "Men's",
  "round": "Final",
  "status": "setup",
  "created_at": "2024-06-05T15:30:45",
  "message": "Match created successfully"
}
```

#### Get Match Details
```
GET /api/match/{match_id}

Response:
{
  "match_id": "20240605_153045",
  "message": "Match details will be available after creation"
}
```

### Roster Management

#### Add Roster
```
POST /api/roster
Content-Type: application/json

Request Body:
{
  "match_id": "20240605_153045",
  "players": [
    {"jersey_number": 7, "name": "Arya Kurnia", "team": "A"},
    {"jersey_number": 10, "name": "Rio Putra", "team": "A"},
    {"jersey_number": 5, "name": "Dhea Anwar", "team": "B"}
  ]
}

Response:
{
  "match_id": "20240605_153045",
  "players_added": 3,
  "message": "Roster saved successfully"
}
```

#### Get Roster
```
GET /api/roster/{match_id}

Response:
{
  "match_id": "20240605_153045",
  "players": [],
  "message": "Roster will be available after match setup"
}
```

### Video Upload

#### Upload Video File
```
POST /api/upload-video
Content-Type: multipart/form-data

Body:
file: <binary video file>

Supported formats: .mp4, .mov, .avi, .mkv
Max size: 5GB

Response:
{
  "video_id": "20240605_153045",
  "filename": "game_final_20240605.mp4",
  "stored_as": "20240605_153045_game_final_20240605.mp4",
  "size": 2147483648,
  "upload_time": "2024-06-05T15:30:45",
  "status": "uploaded",
  "message": "Video uploaded successfully. Processing will start shortly."
}
```

### Statistics

#### Get Live Stats
```
GET /api/stats/live

Response:
{
  "timestamp": null,
  "quarter": null,
  "game_clock": null,
  "score": {"team_a": 0, "team_b": 0},
  "possession": {"team_a": 0.5, "team_b": 0.5},
  "message": "Waiting for video processing..."
}
```

#### Get Player Stats
```
GET /api/stats/player/{player_id}

Response:
{
  "player_id": 7,
  "message": "Player stats will be available after video processing"
}
```

#### Get Team Stats
```
GET /api/stats/team/{team_id}
Parameters:
- team_id: "A" or "B"

Response:
{
  "team_id": "A",
  "message": "Team stats will be available after video processing"
}
```

#### Get Quarter Stats
```
GET /api/stats/quarter/{quarter}
Parameters:
- quarter: 1, 2, 3, or 4

Response:
{
  "quarter": 1,
  "message": "Quarter stats will be available after video processing"
}
```

### Events

#### Get Event Log
```
GET /api/events
Query Parameters:
- match_id: string (optional)
- limit: int (default: 100)

Response:
{
  "match_id": null,
  "limit": 100,
  "events": [],
  "message": "Events will be logged during video processing"
}
```

### WebSocket

#### Real-time Frame Updates
```
WS /ws/live

Connection:
const ws = new WebSocket('ws://localhost:8000/ws/live');

Incoming Message (Frame Update):
{
  "type": "frame_update",
  "timestamp": 1717596645000,
  "quarter": 1,
  "gameClock": "08:42",
  "score": {"teamA": 15, "teamB": 12},
  "possession": {"teamA": 0.70, "teamB": 0.30},
  "players": [
    {
      "trackId": 1,
      "jerseyNumber": 7,
      "name": "A. Kurnia",
      "team": "A",
      "bbox": [100, 50, 150, 200],
      "courtPos": [5.2, 7.5],
      "action": "SHOOT",
      "speedKmh": 3.2,
      "keypoints": [[110, 60, 0.95], [120, 70, 0.92], ...]
    }
  ],
  "ball": {
    "bbox": [160, 45, 175, 60],
    "courtPos": [5.5, 7.6],
    "trajectory": [[5.5, 7.6], [5.4, 7.5], [5.3, 7.4]]
  },
  "event": null
}

Incoming Message (Game Event):
{
  "type": "event",
  "eventType": "FGM",
  "playerId": 7,
  "playerName": "A. Kurnia",
  "points": 2,
  "quarter": 1,
  "gameClock": "08:31",
  "courtPos": [5.2, 7.5]
}
```

#### Real-time Events Stream
```
WS /ws/events

Connection:
const ws = new WebSocket('ws://localhost:8000/ws/events');

Messages follow same format as /ws/live
```

---

## Frontend Type Definitions

All types are defined in `frontend/src/types/index.ts`:

```typescript
interface Player {
  trackId: number;
  jerseyNumber: number;
  name: string;
  team: 'A' | 'B';
  bbox: [number, number, number, number];
  courtPos: [number, number];
  action: ActionLabel;
  speedKmh: number;
  keypoints: Array<[number, number, number]>;
}

interface PlayerStats {
  playerId: number;
  name: string;
  jerseyNumber: number;
  team: 'A' | 'B';
  minutes: number;
  pts: number;
  // ... 20+ stat fields
  eff: number;
}

interface MpiMetrics {
  playerId: number;
  quarter: number;
  distanceCoveredM: number;
  avgSpeedKmh: number;
  maxSpeedKmh: number;
  jumpHeightCm: number;
  accelerationMs2: number;
  agility: number;
  endurance: number;
  fatigue: number;
  mpiComposite: number;
}

interface GameEvent {
  type: 'event';
  eventType: 'FGM' | 'FGA' | 'REB' | 'AST' | 'STL' | 'BLK' | 'TOV' | 'FOUL';
  playerId: number;
  playerName: string;
  points?: number;
  quarter: number;
  gameClock: string;
  courtPos: [number, number];
}
```

---

## Testing with cURL

### Create Match
```bash
curl -X POST http://localhost:8000/api/match \
  -H "Content-Type: application/json" \
  -d '{
    "team_a": "UI Negeri",
    "team_b": "Binus Legends",
    "category": "Men'\''s",
    "round": "Final"
  }'
```

### Upload Video
```bash
curl -X POST http://localhost:8000/api/upload-video \
  -F "file=@/path/to/video.mp4"
```

### Get Health
```bash
curl http://localhost:8000/health
```

---

## Swagger API Documentation

After starting the backend, access interactive Swagger UI:
```
http://localhost:8000/docs
```

All endpoints are listed and can be tested directly from the browser!

---

**API is ready for integration! See QUICKSTART.md to start the servers.** 🚀
