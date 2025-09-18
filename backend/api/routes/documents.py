from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import shutil
from pathlib import Path

from models.documents import DocumentResponse, DocumentListResponse, DocumentMetadata
from services.shared_instances import get_document_processor, get_vector_store
from config.settings import config

router = APIRouter(prefix="/documents", tags=["documents"])

# Get shared service instances
document_processor = get_document_processor()
vector_store = get_vector_store()

@router.post("/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Validate file type
        allowed_extensions = {'.pdf', '.docx', '.doc', '.txt'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Create upload directory if it doesn't exist
        os.makedirs(config.upload_dir, exist_ok=True)
        
        # Save uploaded file
        file_path = Path(config.upload_dir) / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the document
        with open(file_path, "rb") as f:
            file_content = f.read()
        processed_content = await document_processor.process_document(file_content, file.filename)
        
        if not processed_content:
            raise HTTPException(
                status_code=500,
                detail="Failed to process document"
            )
        
        # Store in vector database
        success = await vector_store.store_document_content(processed_content)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store document in vector database"
            )
        
        # Prepare response
        from datetime import datetime
        metadata = DocumentMetadata(
            id=processed_content["metadata"].get("document_id", "unknown"),
            file_name=file.filename,
            file_type=file_extension,
            file_size=file_path.stat().st_size,
            upload_date=datetime.now(),
            page_count=processed_content["metadata"].get("total_pages", 0),
            processing_status="completed"
        )
        
        return DocumentResponse(
            success=True,
            message="Document uploaded and processed successfully",
            document=metadata
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/list", response_model=DocumentListResponse)
async def list_documents():
    """List all uploaded documents"""
    try:
        documents = await vector_store.list_all_documents()
        
        # Convert to DocumentMetadata objects
        document_list = []
        for doc_meta in documents:
            try:
                metadata = DocumentMetadata(
                    id=doc_meta.get("document_id", "unknown"),
                    file_name=doc_meta.get("filename", doc_meta.get("file_name", "Unknown")),
                    file_size=doc_meta.get("file_size", 0),
                    file_type=doc_meta.get("file_type", "unknown"),
                    upload_date=doc_meta.get("upload_date", "1970-01-01T00:00:00"),
                    page_count=doc_meta.get("total_pages", doc_meta.get("page_count", 0)),
                    processing_status=doc_meta.get("processing_status", "completed")
                )
                document_list.append(metadata)
            except Exception as e:
                print(f"Error processing document metadata: {e}")
                continue
        
        return DocumentListResponse(
            documents=document_list,
            total=len(document_list)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing documents: {str(e)}"
        )

@router.get("/document/{doc_id}", response_model=dict)
async def get_document_info(doc_id: str):
    """Get detailed information about a specific document"""
    try:
        metadata = await vector_store.get_document_metadata(doc_id)
        
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        return {
            "document_id": doc_id,
            "metadata": metadata,
            "status": "available"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving document info: {str(e)}"
        )

@router.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its content"""
    try:
        # First check if document exists
        metadata = await vector_store.get_document_metadata(doc_id)
        
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        # Delete from vector store
        success = await vector_store.delete_document(doc_id)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete document from vector database"
            )
        
        # Delete physical file if it exists
        try:
            file_path = metadata.get("upload_path")
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not delete physical file: {e}")
        
        return JSONResponse(
            content={
                "message": "Document deleted successfully",
                "document_id": doc_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting document: {str(e)}"
        )

@router.post("/reprocess/{doc_id}")
async def reprocess_document(doc_id: str):
    """Reprocess an existing document"""
    try:
        # Get document metadata
        metadata = await vector_store.get_document_metadata(doc_id)
        
        if not metadata:
            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )
        
        file_path = metadata.get("upload_path")
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(
                status_code=404,
                detail="Original file not found"
            )
        
        # Delete existing content
        await vector_store.delete_document(doc_id)
        
        # Reprocess the document
        with open(file_path, "rb") as f:
            file_content = f.read()
        processed_content = await document_processor.process_document(file_content, os.path.basename(file_path))
        
        if not processed_content:
            raise HTTPException(
                status_code=500,
                detail="Failed to reprocess document"
            )
        
        # Store in vector database
        success = await vector_store.store_document_content(processed_content)
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to store reprocessed document"
            )
        
        return JSONResponse(
            content={
                "message": "Document reprocessed successfully",
                "document_id": doc_id,
                "processing_stats": {
                    "text_chunks": len(processed_content["text_documents"]),
                    "table_chunks": len(processed_content["table_documents"]),
                    "image_chunks": len(processed_content["image_documents"]),
                    "total_pages": processed_content["metadata"].get("total_pages", 0)
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reprocessing document: {str(e)}"
        )
