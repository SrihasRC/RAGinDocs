"""
Complete document processing pipeline endpoint
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import aiofiles
import time
import logging
from typing import List, Dict, Any
from services.pdf_processor import extract_text_from_file, get_file_stats
from services.chunking import chunk_text
from services.embedding import embedding_service
from services.vector_store import vector_store

logger = logging.getLogger(__name__)

# Create a new router for the complete pipeline
pipeline_router = APIRouter(prefix="/pipeline", tags=["complete-pipeline"])

# Initialize services - use the global instance
# embedding_service is already imported and initialized
# Load the embedding model if not already loaded
if not embedding_service.model:
    if not embedding_service.load_model():
        logger.error("Failed to load embedding model")

# Initialize vector store
if not vector_store.initialize():
    logger.error("Failed to initialize vector store")

@pipeline_router.post("/document")
async def upload_and_process_complete(file: UploadFile = File(...)):
    """
    Complete document processing pipeline:
    1. Upload and validate file
    2. Extract text
    3. Create chunks  
    4. Generate embeddings
    5. Store in vector database
    6. Return complete processing results
    """
    start_time = time.time()
    
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Check file extension
        allowed_extensions = {".pdf", ".txt", ".docx"}
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not allowed. Allowed types: {', '.join(allowed_extensions)}"
            )
        
        # Read and validate file size
        contents = await file.read()
        max_size = 10 * 1024 * 1024  # 10MB
        if len(contents) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {max_size // (1024*1024)}MB"
            )
        
        # Create uploads directory
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(uploads_dir, file.filename)
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(contents)
        
        logger.info(f"File saved: {file_path} ({len(contents)} bytes)")
        
        # Step 1: Extract text
        logger.info(f"Starting text extraction for {file.filename}")
        try:
            pages = extract_text_from_file(file_path)
            logger.info(f"Extracted {len(pages)} pages from {file.filename}")
            
            # Log first few characters of extracted text for debugging
            if pages:
                total_text = " ".join([page["text"] for page in pages])
                logger.info(f"Total text length: {len(total_text)} characters")
                logger.info(f"First 200 chars: {total_text[:200]}")
            else:
                logger.warning(f"No pages extracted from {file.filename}")
                
        except Exception as e:
            logger.error(f"Text extraction failed for {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Text extraction failed: {str(e)}")
        
        # Check if we have text
        if not pages:
            logger.warning(f"No text extracted from {file.filename}")
            return JSONResponse({
                "message": "File processed but no text found",
                "filename": file.filename,
                "file_size": len(contents),
                "processing_time": time.time() - start_time,
                "extracted_text": "",
                "chunks_created": 0,
                "embeddings_generated": 0,
                "chunks": [],
                "embeddings_sample": []
            })
        
        # Combine all page text
        full_text = " ".join([page["text"] for page in pages])
        
        # Step 2: Create chunks
        logger.info(f"Starting chunking for {file.filename}")
        try:
            # Convert pages format for chunking function
            chunks = chunk_text(
                text_pages=pages,  # Use the pages from extraction
                chunk_size=1000,
                overlap=200,
                respect_boundaries=True
            )
            logger.info(f"Created {len(chunks)} chunks from {file.filename}")
            
        except Exception as e:
            logger.error(f"Chunking failed for {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")
        
        # Step 3: Generate embeddings
        logger.info(f"Starting embedding generation for {file.filename}")
        try:
            chunk_texts = [chunk["text"] for chunk in chunks]
            embeddings = embedding_service.generate_embeddings(chunk_texts)
            logger.info(f"Generated {len(embeddings)} embeddings for {file.filename}")
            
        except Exception as e:
            logger.error(f"Embedding generation failed for {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")
        
        # Step 4: Store in vector database
        logger.info(f"Starting vector storage for {file.filename}")
        try:
            # Prepare data for vector storage
            vector_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                vector_data.append({
                    "chunk_id": f"{file.filename}_{i}",
                    "text": chunk["text"],
                    "embedding": embedding,
                    "metadata": {
                        "page_number": chunk.get("page_number", 1),
                        "source_file": chunk.get("source_file", file.filename),
                        "file_type": chunk.get("file_type", "unknown"),
                        "char_count": chunk.get("char_count", 0),
                        "word_count": chunk.get("word_count", 0),
                        "chunk_index": i,
                        "start_char": chunk.get("start_char", 0),
                        "end_char": chunk.get("end_char", 0)
                    }
                })
            
            # Store in vector database
            result = vector_store.add_embeddings(
                embeddings=[item["embedding"] for item in vector_data],
                texts=[item["text"] for item in vector_data],
                metadatas=[item["metadata"] for item in vector_data],
                ids=[item["chunk_id"] for item in vector_data]
            )
            
            logger.info(f"Stored {len(vector_data)} vectors for {file.filename}")
            
        except Exception as e:
            logger.error(f"Vector storage failed for {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Vector storage failed: {str(e)}")
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare response
        response_data = {
            "file_id": f"{file.filename}_{int(time.time())}",
            "filename": file.filename,
            "file_size": len(contents),
            "chunks_created": len(chunks),
            "embeddings_generated": len(embeddings),
            "processing_time": processing_time,
            "extracted_text": full_text,
            "chunks": [
                {
                    "chunk_id": f"{file.filename}_{i}",
                    "text": chunk["text"],
                    "chunk_index": i,
                    "metadata": {
                        "page_number": chunk.get("page_number", 1),
                        "source_file": chunk.get("source_file", file.filename),
                        "file_type": chunk.get("file_type", "unknown"),
                        "char_count": chunk.get("char_count", 0),
                        "word_count": chunk.get("word_count", 0)
                    }
                }
                for i, chunk in enumerate(chunks[:10])  # Return first 10 chunks
            ],
            "embeddings_created": len(embeddings),
            "embeddings_sample": embeddings[:3]  # First 3 embeddings
        }
        
        logger.info(f"Complete processing finished for {file.filename} in {processing_time:.2f}s")
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        # Clean up uploaded file
        try:
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {str(e)}")
