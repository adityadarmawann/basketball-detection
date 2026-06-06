# 🗄️ Smart Vision Basketball Analytics — Database Schema Reference

## MongoDB Collections

All MongoDB collections are ready to be created. Use these schemas for database initialization.

### Collection: `matches`
```javascript
db.createCollection("matches", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["match_id", "team_a", "team_b", "created_at"],
      properties: {
        _id: { bsonType: "objectId" },
        match_id: { bsonType: "string", description: "Unique match identifier" },
        team_a: {
          bsonType: "object",
          properties: {
            name: { bsonType: "string" },
            color: { bsonType: "string" }
          }
        },
        team_b: {
          bsonType: "object",
          properties: {
            name: { bsonType: "string" },
            color: { bsonType: "string" }
          }
        },
        category: { bsonType: "string", enum: ["Men's", "Women's"] },
        round: { bsonType: "string" },
        status: { bsonType: "string", enum: ["setup", "processing", "done"] },
        created_at: { bsonType: "date" },
        updated_at: { bsonType: "date" }
      }
    }
  }
})

// Sample Document
{
  "_id": ObjectId("507f1f77bcf86cd799439011"),
  "match_id": "20240605_153045",
  "team_a": { "name": "UI Negeri", "color": "#00BCD4" },
  "team_b": { "name": "Binus Legends", "color": "#1A1A2E" },
  "category": "Men's",
  "round": "Final",
  "status": "processing",
  "created_at": ISODate("2024-06-05T15:30:45.000Z"),
  "updated_at": ISODate("2024-06-05T15:35:22.000Z")
}
```

### Collection: `players` (Roster)
```javascript
db.createCollection("players")

// Sample Documents
{
  "_id": ObjectId("507f1f77bcf86cd799439012"),
  "match_id": "20240605_153045",
  "jersey_number": 7,
  "name": "Arya Kurnia",
  "team": "A",
  "track_id": 1,
  "university": "UI Negeri"
}

{
  "_id": ObjectId("507f1f77bcf86cd799439013"),
  "match_id": "20240605_153045",
  "jersey_number": 10,
  "name": "Rio Putra",
  "team": "A",
  "track_id": 2,
  "university": "UI Negeri"
}

{
  "_id": ObjectId("507f1f77bcf86cd799439014"),
  "match_id": "20240605_153045",
  "jersey_number": 5,
  "name": "Dhea Anwar",
  "team": "B",
  "track_id": 3,
  "university": "Binus Legends"
}
```

### Collection: `events` (Event Log)
```javascript
db.createCollection("events")
db.events.createIndex({ "match_id": 1, "timestamp_ms": -1 })

// Sample Documents
{
  "_id": ObjectId("507f1f77bcf86cd799439015"),
  "match_id": "20240605_153045",
  "event_type": "FGA",
  "player_id": 7,
  "team": "A",
  "quarter": 1,
  "game_clock": "08:42",
  "timestamp_ms": 1717596645000,
  "court_pos": [5.2, 7.5],
  "detail": {
    "shot_type": "2PT",
    "distance_m": 4.5
  }
}

{
  "_id": ObjectId("507f1f77bcf86cd799439016"),
  "match_id": "20240605_153045",
  "event_type": "FGM",
  "player_id": 7,
  "team": "A",
  "quarter": 1,
  "game_clock": "08:31",
  "timestamp_ms": 1717596655000,
  "court_pos": [5.2, 7.5],
  "detail": {
    "points": 2,
    "shot_type": "2PT"
  }
}

{
  "_id": ObjectId("507f1f77bcf86cd799439017"),
  "match_id": "20240605_153045",
  "event_type": "REB",
  "player_id": 10,
  "team": "A",
  "quarter": 1,
  "game_clock": "08:28",
  "timestamp_ms": 1717596668000,
  "court_pos": [7.5, 7.5],
  "detail": {
    "reb_type": "DREB"
  }
}
```

### Collection: `player_stats` (Live Stats per Quarter)
```javascript
db.createCollection("player_stats")
db.player_stats.createIndex({ "match_id": 1, "player_id": 1, "quarter": 1 })

// Sample Document
{
  "_id": ObjectId("507f1f77bcf86cd799439018"),
  "match_id": "20240605_153045",
  "player_id": 7,
  "jersey_number": 7,
  "name": "A. Kurnia",
  "team": "A",
  "quarter": 1,
  "minutes": 600,  // in seconds
  "pts": 8,
  "fgm": 2,
  "fga": 5,
  "fgp": 40.0,
  "tpm": 0,
  "tpa": 1,
  "tpp": 0.0,
  "ftm": 4,
  "fta": 10,
  "ftp": 40.0,
  "oreb": 0,
  "dreb": 9,
  "treb": 9,
  "ast": 2,
  "stl": 3,
  "blk": 0,
  "tov": 9,
  "fouls": 3,
  "plus_minus": 20,
  "eff": 8,
  "dist_covered_m": 4200,
  "avg_speed_kmh": 14.2,
  "max_speed_kmh": 28.7,
  "updated_at": ISODate("2024-06-05T15:40:00.000Z")
}
```

### Collection: `mpi_metrics` (Physical Metrics)
```javascript
db.createCollection("mpi_metrics")
db.mpi_metrics.createIndex({ "match_id": 1, "player_id": 1, "quarter": 1 })

// Sample Document
{
  "_id": ObjectId("507f1f77bcf86cd799439019"),
  "match_id": "20240605_153045",
  "player_id": 7,
  "quarter": 1,
  "power_score": 72,
  "agility_score": 68,
  "endurance_score": 81,
  "efficiency_score": 65,
  "cognitive_score": 74,
  "mpi_composite": 72,
  "jump_height_cm": 62,
  "acceleration_ms2": 3.2,
  "deceleration_ms2": 2.8,
  "fatigue_index": 23,
  "is_approx": true,
  "accuracy_note": "±15-25% (CV-based estimation)",
  "updated_at": ISODate("2024-06-05T15:40:00.000Z")
}
```

---

## Redis Data Structures

Real-time state stored in Redis with automatic expiration.

### Key: `game:{match_id}:frame`
```json
{
  "timestamp": 1717596645000,
  "quarter": 1,
  "game_clock": "08:42",
  "shot_clock": 18,
  "score_a": 15,
  "score_b": 12,
  "possession_a": 0.70,
  "possession_b": 0.30,
  "frame_number": 12450,
  "processing_fps": 25
}
```
TTL: 60 seconds

### Key: `game:{match_id}:players:{track_id}`
```json
{
  "jersey_number": 7,
  "name": "A. Kurnia",
  "team": "A",
  "bbox": [100, 50, 150, 200],
  "court_pos": [5.2, 7.5],
  "action": "SHOOT",
  "speed_kmh": 3.2,
  "last_frame": 12450
}
```
TTL: 5 seconds

### Key: `game:{match_id}:ball`
```json
{
  "bbox": [160, 45, 175, 60],
  "court_pos": [5.5, 7.6],
  "trajectory": [[5.5, 7.6], [5.4, 7.5], [5.3, 7.4]],
  "last_frame": 12450
}
```
TTL: 5 seconds

### Key: `game:{match_id}:events`
```
Sorted Set (by timestamp_ms)
- Score: timestamp_ms (numeric)
- Member: JSON stringified event object
```
Max: 1000 recent events

### Key: `session:{user_id}:watching`
```json
{
  "match_id": "20240605_153045",
  "connected_at": 1717596645000,
  "last_ping": 1717596700000
}
```
TTL: 600 seconds

---

## Database Initialization Script (Python)

```python
# backend/db/init_db.py (to be created)
from pymongo import MongoClient, ASCENDING
from datetime import datetime

def initialize_database(mongo_url: str, db_name: str = "smart_vision_basketball"):
    client = MongoClient(mongo_url)
    db = client[db_name]
    
    # Create collections with indexes
    
    # Matches collection
    db.matches.create_index([("match_id", ASCENDING)], unique=True)
    
    # Players collection
    db.players.create_index([("match_id", ASCENDING)])
    db.players.create_index([("jersey_number", ASCENDING), ("team", ASCENDING)])
    
    # Events collection
    db.events.create_index([("match_id", ASCENDING), ("timestamp_ms", -1)])
    db.events.create_index([("event_type", ASCENDING)])
    
    # Player stats collection
    db.player_stats.create_index([("match_id", ASCENDING), ("player_id", ASCENDING), ("quarter", ASCENDING)])
    
    # MPI metrics collection
    db.mpi_metrics.create_index([("match_id", ASCENDING), ("player_id", ASCENDING), ("quarter", ASCENDING)])
    
    print("✅ Database initialized successfully!")
    return db

if __name__ == "__main__":
    initialize_database("mongodb://localhost:27017")
```

---

## Querying Examples

### Get all events for a match
```python
events = db.events.find({"match_id": "20240605_153045"}).sort("timestamp_ms", -1)
```

### Get player stats for a quarter
```python
stats = db.player_stats.find({
    "match_id": "20240605_153045",
    "quarter": 1
}).sort("pts", -1)
```

### Get MPI metrics for a player
```python
mpi = db.mpi_metrics.find({
    "match_id": "20240605_153045",
    "player_id": 7
})
```

### Real-time stats update (bulk insert/update)
```python
# Update multiple player stats at once
db.player_stats.insert_many([...], ordered=False)
```

---

**Database schema is ready for implementation! Create these collections and indexes to begin storing game data.** 🗄️
