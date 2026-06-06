"""
MongoDB connection and utilities.
"""

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
DB_NAME = "smart_vision_basketball"

client = None
db = None

async def connect_mongo():
    """Initialize MongoDB connection."""
    global client, db
    try:
        client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        db = client[DB_NAME]
        print(f"✅ MongoDB connected: {MONGO_URL}")
        return db
    except ServerSelectionTimeoutError as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise

async def close_mongo():
    """Close MongoDB connection."""
    global client
    if client:
        client.close()
        print("MongoDB disconnected")

def get_db():
    """Get database instance."""
    return db

def create_indexes():
    """Create database indexes for better query performance."""
    global db
    if not db:
        return

    try:
        # Match indexes
        db.matches.create_index("match_id")
        
        # Player indexes
        db.players.create_index([("match_id", 1), ("jersey_number", 1)])
        
        # Stats indexes
        db.player_stats.create_index([("match_id", 1), ("player_id", 1)])
        db.player_stats.create_index([("quarter", 1)])
        
        # Events indexes
        db.events.create_index([("match_id", 1), ("timestamp", 1)])
        
        # MPI indexes
        db.mpi_metrics.create_index([("match_id", 1), ("player_id", 1)])
        
        print("✓ Database indexes created")
    except Exception as e:
        print(f"! Index creation warning: {e}")
