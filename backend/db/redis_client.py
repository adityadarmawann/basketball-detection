"""
Redis connection for real-time state management.
"""

import redis.asyncio as redis
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

redis_client = None

async def get_redis():
    """Get or create Redis connection."""
    global redis_client
    if redis_client is None:
        redis_client = await redis.from_url(REDIS_URL, decode_responses=True)
        print(f"✅ Redis connected: {REDIS_URL}")
    return redis_client

async def close_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        print("Redis disconnected")
