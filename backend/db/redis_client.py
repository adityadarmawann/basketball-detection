"""
Redis connection for real-time state management.
Gracefully degrades when the redis package is not installed.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

try:
    import redis.asyncio as _aioredis
    _REDIS_AVAILABLE = True
except ImportError:
    _aioredis = None
    _REDIS_AVAILABLE = False
    logger.warning("redis package not installed — Redis features disabled. "
                   "pip install redis>=5.0 to enable.")

redis_client = None


async def get_redis():
    """Get or create Redis async connection. Returns None if redis unavailable."""
    global redis_client
    if not _REDIS_AVAILABLE:
        return None
    if redis_client is None:
        redis_client = await _aioredis.from_url(REDIS_URL, decode_responses=True)
        print(f"✅ Redis connected: {REDIS_URL}")
    return redis_client


async def close_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        await redis_client.close()
        redis_client = None
        print("Redis disconnected")
