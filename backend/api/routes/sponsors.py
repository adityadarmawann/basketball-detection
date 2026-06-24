"""
Sponsor endpoints:
- GET    /sponsors/library              — list library entries (global)
- POST   /sponsors/library              — add to library
- DELETE /sponsors/library/{id}         — remove from library

- GET    /sponsors/{match_id}           — list all sponsors for a match
- POST   /sponsors/{match_id}           — add sponsor to match
- DELETE /sponsors/{match_id}/{id}      — remove sponsor from match
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from datetime import datetime
from pathlib import Path
from bson import ObjectId
import os, uuid
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

router = APIRouter()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client    = MongoClient(MONGO_URL)
db        = client["smart_vision_basketball"]

LOGOS_DIR = Path(__file__).parent.parent.parent / "uploads" / "logos"
LOGOS_DIR.mkdir(parents=True, exist_ok=True)

VALID_CATEGORIES = {"speed", "endurance", "score", "mvp"}


def _doc_to_dict(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    return doc


# ── Sponsor library (global pool) ────────────────────────────────────────────
# IMPORTANT: must be declared before /{match_id} routes to avoid path collision

@router.get("/sponsors/library")
async def list_library():
    docs = list(db.sponsor_library.find().sort("company_name", 1))
    return {"library": [_doc_to_dict(d) for d in docs]}


@router.post("/sponsors/library")
async def add_to_library(
    company_name: str = Form(...),
    logo: Optional[UploadFile] = File(None),
):
    if not company_name.strip():
        raise HTTPException(status_code=400, detail="company_name required")

    existing = db.sponsor_library.find_one(
        {"company_name": {"$regex": f"^{company_name.strip()}$", "$options": "i"}}
    )
    if existing:
        raise HTTPException(status_code=409, detail="Sponsor already in library")

    logo_url: Optional[str] = None
    if logo and logo.filename:
        ext = Path(logo.filename).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".svg"):
            raise HTTPException(status_code=400, detail="Logo must be jpg/png/webp/svg")
        filename = f"lib_{uuid.uuid4().hex[:12]}{ext}"
        dest     = LOGOS_DIR / filename
        dest.write_bytes(await logo.read())
        logo_url = f"/api/media/logos/{filename}"

    doc = {
        "company_name": company_name.strip(),
        "logo_url":     logo_url,
        "created_at":   datetime.now(),
    }
    result = db.sponsor_library.insert_one(doc)
    doc["id"] = str(result.inserted_id)
    doc.pop("_id", None)
    return doc


@router.delete("/sponsors/library/{lib_id}")
async def delete_from_library(lib_id: str):
    try:
        oid = ObjectId(lib_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid id")

    doc = db.sponsor_library.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail="Not found")

    if doc.get("logo_url"):
        fname = Path(doc["logo_url"]).name
        logo_path = LOGOS_DIR / fname
        if logo_path.exists():
            logo_path.unlink()

    db.sponsor_library.delete_one({"_id": oid})
    return {"deleted": lib_id}


# ── Match sponsors ────────────────────────────────────────────────────────────

@router.get("/sponsors/{match_id}")
async def list_sponsors(match_id: str):
    docs = list(db.sponsors.find({"match_id": match_id}))
    return {"sponsors": [_doc_to_dict(d) for d in docs]}


@router.post("/sponsors/{match_id}")
async def add_sponsor(
    match_id: str,
    category: str     = Form(...),
    company_name: str = Form(...),
    logo: Optional[UploadFile] = File(None),
    existing_logo_url: Optional[str] = Form(None),
):
    if category not in VALID_CATEGORIES:
        raise HTTPException(status_code=400, detail=f"category must be one of {VALID_CATEGORIES}")
    if not company_name.strip():
        raise HTTPException(status_code=400, detail="company_name required")

    logo_url: Optional[str] = None
    if logo and logo.filename:
        ext = Path(logo.filename).suffix.lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".svg"):
            raise HTTPException(status_code=400, detail="Logo must be jpg/png/webp/svg")
        filename = f"{match_id}_{category}_{uuid.uuid4().hex[:8]}{ext}"
        dest     = LOGOS_DIR / filename
        dest.write_bytes(await logo.read())
        logo_url = f"/api/media/logos/{filename}"
    elif existing_logo_url:
        logo_url = existing_logo_url

    doc = {
        "match_id":     match_id,
        "category":     category,
        "company_name": company_name.strip(),
        "logo_url":     logo_url,
        "created_at":   datetime.now(),
    }
    result = db.sponsors.insert_one(doc)
    doc["id"] = str(result.inserted_id)
    doc.pop("_id", None)
    return doc


@router.delete("/sponsors/{match_id}/{sponsor_id}")
async def delete_sponsor(match_id: str, sponsor_id: str):
    try:
        oid = ObjectId(sponsor_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid sponsor id")

    doc = db.sponsors.find_one({"_id": oid, "match_id": match_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Sponsor not found")

    if doc.get("logo_url"):
        fname = Path(doc["logo_url"]).name
        logo_path = LOGOS_DIR / fname
        if logo_path.exists():
            logo_path.unlink()

    db.sponsors.delete_one({"_id": oid})
    return {"deleted": sponsor_id}
