"""
Database initialization script.
Creates MongoDB collections, indexes, and optional seed data.
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "smart_vision_basketball"

async def init_db():
    """Initialize database."""
    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[DB_NAME]
        
        print("🔧 Initializing database...")

        # Create collections with schema validation (optional)
        collections = [
            'matches',
            'players',
            'player_stats',
            'events',
            'mpi_metrics',
            'frame_updates',
            'videos',
        ]

        for collection in collections:
            if collection not in db.list_collection_names():
                db.create_collection(collection)
                print(f"  ✓ Created collection: {collection}")
            else:
                print(f"  ✓ Collection exists: {collection}")

        # Create indexes
        print("\n📊 Creating indexes...")
        
        db.matches.create_index("match_id")
        print("  ✓ Index: matches.match_id")
        
        db.players.create_index([("match_id", 1), ("jersey_number", 1)])
        print("  ✓ Index: players (match_id, jersey_number)")
        
        db.player_stats.create_index([("match_id", 1), ("player_id", 1)])
        db.player_stats.create_index([("quarter", 1)])
        print("  ✓ Index: player_stats (match_id, player_id) + (quarter)")
        
        db.events.create_index([("match_id", 1), ("timestamp", 1)])
        print("  ✓ Index: events (match_id, timestamp)")
        
        db.mpi_metrics.create_index([("match_id", 1), ("player_id", 1)])
        print("  ✓ Index: mpi_metrics (match_id, player_id)")

        # Seed test data (optional)
        if "--seed" in sys.argv:
            print("\n🌱 Seeding test data...")
            
            # Test match
            test_match = {
                "match_id": "demo_20260605",
                "team_a": {"name": "Team Alpha", "color": "#00BCD4"},
                "team_b": {"name": "Team Beta", "color": "#1A1A2E"},
                "category": "Men's",
                "round": "Final",
                "status": "completed",
                "created_at": datetime.now(),
                "updated_at": datetime.now()
            }
            result = db.matches.insert_one(test_match)
            print(f"  ✓ Test match: {result.inserted_id}")
            
            # Test players
            test_players = [
                {"match_id": "demo_20260605", "jersey_number": 10, "name": "John Doe", "team": "A", "track_id": None, "created_at": datetime.now()},
                {"match_id": "demo_20260605", "jersey_number": 23, "name": "Jane Smith", "team": "B", "track_id": None, "created_at": datetime.now()},
            ]
            db.players.insert_many(test_players)
            print(f"  ✓ Test players: {len(test_players)}")

        print("\n✅ Database initialization complete!")
        client.close()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(init_db())
