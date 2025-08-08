from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import aiofiles
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])

# Allowed file types
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS)

@router.post("/document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document (PDF, TXT, or DOCX) for processing.
    
    This endpoint:
    1. Validates file type and size
    2. Saves file to uploads directory
    3. Returns file info for further processing
    """
    try:
        # Check if filename exists
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Validate file type
        if not allowed_file(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Validate file size
        contents = await file.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Create uploads directory if it doesn't exist
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(uploads_dir, file.filename)
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(contents)
        
        logger.info(f"File uploaded successfully: {file.filename}")
        
        return JSONResponse({
            "message": "File uploaded successfully",
            "filename": file.filename,
            "file_path": file_path,
            "file_size": len(contents),
            "status": "ready_for_processing"
        })
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@router.get("/status/{filename}")
async def get_upload_status(filename: str):
    """
    Get the status of an uploaded file.
    """
    file_path = os.path.join("uploads", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    file_size = os.path.getsize(file_path)
    
    return JSONResponse({
        "filename": filename,
        "file_path": file_path,
        "file_size": file_size,
        "status": "uploaded",
        "processed": False  # Will be updated in later steps
    })

@router.delete("/document/{filename}")
async def delete_document(filename: str):
    """
    Delete an uploaded document.
    """
    file_path = os.path.join("uploads", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        os.remove(file_path)
        logger.info(f"File deleted successfully: {filename}")
        
        return JSONResponse({
            "message": "File deleted successfully",
            "filename": filename
        })
        
    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")
