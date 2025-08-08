from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import aiofiles
from typing import List
import logging
from services.pdf_processor import extract_text_from_file, get_file_stats

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
    3. Extracts text from the file
    4. Returns file info and extracted text stats
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
        
        # Extract text from the file
        try:
            pages = extract_text_from_file(file_path)
            stats = get_file_stats(pages)
            
            logger.info(f"File uploaded and processed successfully: {file.filename}")
            logger.info(f"Extracted {stats['total_pages']} pages, {stats['total_chars']} characters")
            
            return JSONResponse({
                "message": "File uploaded and processed successfully",
                "filename": file.filename,
                "file_path": file_path,
                "file_size": len(contents),
                "status": "processed",
                "extraction_stats": stats,
                "pages_extracted": len(pages),
                "ready_for_chunking": True
            })
            
        except Exception as extraction_error:
            logger.error(f"Error extracting text from {file.filename}: {str(extraction_error)}")
            
            # Still return success for upload, but indicate extraction failed
            return JSONResponse({
                "message": "File uploaded but text extraction failed",
                "filename": file.filename,
                "file_path": file_path,
                "file_size": len(contents),
                "status": "upload_success_extraction_failed",
                "error": str(extraction_error),
                "ready_for_chunking": False
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
    
    # Try to get extraction stats
    try:
        pages = extract_text_from_file(file_path)
        stats = get_file_stats(pages)
        extraction_success = True
        extraction_error = None
    except Exception as e:
        stats = {}
        extraction_success = False
        extraction_error = str(e)
    
    return JSONResponse({
        "filename": filename,
        "file_path": file_path,
        "file_size": file_size,
        "status": "uploaded",
        "extraction_success": extraction_success,
        "extraction_error": extraction_error,
        "extraction_stats": stats if extraction_success else None,
        "ready_for_chunking": extraction_success
    })

@router.get("/text/{filename}")
async def get_extracted_text(filename: str):
    """
    Get the extracted text from an uploaded file.
    """
    file_path = os.path.join("uploads", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        pages = extract_text_from_file(file_path)
        stats = get_file_stats(pages)
        
        return JSONResponse({
            "filename": filename,
            "extraction_stats": stats,
            "pages": pages,
            "total_pages": len(pages)
        })
        
    except Exception as e:
        logger.error(f"Error extracting text from {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")

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
