from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import aiofiles
from typing import List, Optional
import logging
from services.pdf_processor import extract_text_from_file, get_file_stats
from services.chunking import chunk_text, get_chunk_stats, optimize_chunk_size, preview_chunks
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["upload"])

# Pydantic models for requests
class ChunkRequest(BaseModel):
    chunk_size: Optional[int] = 1000
    overlap: Optional[int] = 200
    respect_boundaries: Optional[bool] = True

class ChunkPreviewRequest(BaseModel):
    chunk_size: Optional[int] = 1000
    overlap: Optional[int] = 200
    preview_length: Optional[int] = 100

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

@router.post("/chunk/{filename}")
async def chunk_document(filename: str, request: ChunkRequest):
    """
    Chunk an uploaded document into overlapping text segments.
    
    This endpoint:
    1. Retrieves extracted text from uploaded file
    2. Splits text into chunks with specified parameters
    3. Returns chunks with metadata and statistics
    """
    file_path = os.path.join("uploads", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Extract text from file
        pages = extract_text_from_file(file_path)
        
        # Create chunks
        chunks = chunk_text(
            pages, 
            chunk_size=request.chunk_size or 1000,
            overlap=request.overlap or 200,
            respect_boundaries=request.respect_boundaries or True
        )
        
        # Get statistics
        chunk_stats = get_chunk_stats(chunks)
        
        logger.info(f"Created {len(chunks)} chunks for {filename}")
        
        return JSONResponse({
            "filename": filename,
            "chunking_params": {
                "chunk_size": request.chunk_size,
                "overlap": request.overlap,
                "respect_boundaries": request.respect_boundaries
            },
            "chunk_stats": chunk_stats,
            "chunks": chunks,
            "total_chunks": len(chunks)
        })
        
    except Exception as e:
        logger.error(f"Error chunking document {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error chunking document: {str(e)}")

@router.post("/chunk-preview/{filename}")
async def preview_document_chunks(filename: str, request: ChunkPreviewRequest):
    """
    Preview how a document would be chunked without creating full chunks.
    """
    file_path = os.path.join("uploads", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Extract text from file
        pages = extract_text_from_file(file_path)
        
        # Create chunk previews
        previews = preview_chunks(
            pages,
            chunk_size=request.chunk_size or 1000,
            overlap=request.overlap or 200,
            preview_length=request.preview_length or 100
        )
        
        # Get recommended chunk size
        recommended_size = optimize_chunk_size(pages)
        
        return JSONResponse({
            "filename": filename,
            "preview_params": {
                "chunk_size": request.chunk_size,
                "overlap": request.overlap,
                "preview_length": request.preview_length
            },
            "recommended_chunk_size": recommended_size,
            "chunk_previews": previews,
            "total_chunks": len(previews)
        })
        
    except Exception as e:
        logger.error(f"Error previewing chunks for {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error previewing chunks: {str(e)}")

@router.get("/chunk-recommend/{filename}")
async def get_chunk_recommendations(filename: str):
    """
    Get recommended chunking parameters for a document.
    """
    file_path = os.path.join("uploads", filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Extract text from file
        pages = extract_text_from_file(file_path)
        file_stats = get_file_stats(pages)
        
        # Calculate recommendations
        recommended_chunk_size = optimize_chunk_size(pages)
        recommended_overlap = min(200, recommended_chunk_size // 5)  # 20% overlap
        
        # Estimate chunk count
        total_chars = file_stats["total_chars"]
        estimated_chunks = total_chars // (recommended_chunk_size - recommended_overlap)
        
        return JSONResponse({
            "filename": filename,
            "file_stats": file_stats,
            "recommendations": {
                "chunk_size": recommended_chunk_size,
                "overlap": recommended_overlap,
                "estimated_chunks": estimated_chunks,
                "reasoning": _get_recommendation_reasoning(file_stats, recommended_chunk_size)
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting recommendations for {filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")

def _get_recommendation_reasoning(file_stats: dict, chunk_size: int) -> str:
    """Generate reasoning for chunk size recommendation."""
    total_chars = file_stats["total_chars"]
    
    if total_chars < 5000:
        return f"Small document ({total_chars} chars): Using smaller chunks for granular retrieval"
    elif total_chars < 20000:
        return f"Medium document ({total_chars} chars): Using standard chunk size for balanced context"
    else:
        return f"Large document ({total_chars} chars): Using larger chunks for comprehensive context"
