"""
Smart Vision Basketball Analytics Dashboard — Backend (FastAPI)
Entry point for the application.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import database connections
from db.mongo import connect_mongo, close_mongo
from db.redis_client import get_redis  # noqa: F401 — used by routes/upload.py

# Import routes
from api.routes import upload, stats, events, roster, match, sponsors

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting Smart Vision Basketball Analytics Dashboard...")
    await connect_mongo()
    # redis = await get_redis()
    yield
    # Shutdown
    print("🛑 Shutting down...")
    # await redis.close()
    # await close_mongo()

# Create FastAPI app
app = FastAPI(
    title="Smart Vision Basketball Analytics API",
    description="CV-based Basketball Analytics Platform for Campus League",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "smart-vision-api"}

# Include routers
app.include_router(upload.router,   prefix="/api", tags=["Upload"])
app.include_router(stats.router,    prefix="/api", tags=["Stats"])
app.include_router(events.router,   prefix="/api", tags=["Events"])
app.include_router(roster.router,   prefix="/api", tags=["Roster"])
app.include_router(match.router,    prefix="/api", tags=["Match"])
app.include_router(sponsors.router, prefix="/api", tags=["Sponsors"])

# Serve uploaded player photos and sponsor logos as static files
_uploads_dir = Path(__file__).parent / "uploads"
_uploads_dir.mkdir(exist_ok=True)
app.mount("/api/media", StaticFiles(directory=str(_uploads_dir)), name="media")

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        reload=True,
    )
