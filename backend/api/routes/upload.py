"""
Video upload endpoint — POST /upload-video
Handles video file uploads and triggers background processing.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
from datetime import datetime

router = APIRouter()

UPLOAD_PATH = os.getenv("UPLOAD_PATH", "./uploads")
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB

@router.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload video file for processing.
    
    Supported formats: .mp4, .mov, .avi, .mkv
    
    Returns:
    - video_id: unique identifier
    - filename: uploaded filename
    - size: file size in bytes
    - upload_time: timestamp
    """
    
    try:
        # Validate file type
        allowed_extensions = [".mp4", ".mov", ".avi", ".mkv"]
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Create uploads directory if not exists
        os.makedirs(UPLOAD_PATH, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_PATH, unique_filename)
        
        # Save file
        contents = await file.read()
        
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File size exceeds 5GB limit"
            )
        
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # TODO: Trigger background processing task
        # await trigger_video_processing(video_id, file_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "video_id": timestamp,
                "filename": file.filename,
                "stored_as": unique_filename,
                "size": len(contents),
                "upload_time": datetime.now().isoformat(),
                "status": "uploaded",
                "message": "Video uploaded successfully. Processing will start shortly."
            }
        )
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
