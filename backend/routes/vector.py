"""
Vector store API routes for managing embeddings and similarity search.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from services.vector_store import vector_store
from services.embedding import embedding_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/vector", tags=["vector-store"])

# Request/Response Models
class AddEmbeddingsRequest(BaseModel):
    """Request model for adding embeddings to vector store."""
    chunks: List[Dict[str, Any]] = Field(..., description="List of chunks with embeddings")

class SearchRequest(BaseModel):
    """Request model for similarity search."""
    query_text: str = Field(..., description="Query text for similarity search")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filter")

class SearchByEmbeddingRequest(BaseModel):
    """Request model for similarity search using embedding."""
    query_embedding: List[float] = Field(..., description="Query embedding vector")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results to return")
    filter: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filter")

class DeleteRequest(BaseModel):
    """Request model for deleting embeddings."""
    filter: Dict[str, Any] = Field(..., description="Metadata filter for deletion")

class SearchResponse(BaseModel):
    """Response model for search results."""
    documents: List[str] = Field(..., description="Retrieved documents")
    metadatas: List[Dict[str, Any]] = Field(..., description="Document metadata")
    distances: List[float] = Field(..., description="Similarity distances")
    ids: List[str] = Field(..., description="Document IDs")
    count: int = Field(..., description="Number of results returned")

class StatsResponse(BaseModel):
    """Response model for collection statistics."""
    initialized: bool = Field(..., description="Whether vector store is initialized")
    collection_name: Optional[str] = Field(None, description="Name of the collection")
    document_count: Optional[int] = Field(None, description="Number of documents in collection")
    persist_directory: Optional[str] = Field(None, description="Persistence directory")

# Dependency to ensure vector store is initialized
async def ensure_vector_store_initialized():
    """Ensure the vector store is initialized before processing requests."""
    if vector_store.collection is None:
        logger.info("Initializing vector store...")
        success = vector_store.initialize()
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to initialize vector store"
            )
    return vector_store

@router.get("/stats", response_model=StatsResponse)
async def get_vector_store_stats():
    """Get vector store statistics."""
    try:
        stats = vector_store.get_collection_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Error getting vector store stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize")
async def initialize_vector_store():
    """Initialize the vector store."""
    try:
        success = vector_store.initialize()
        if success:
            stats = vector_store.get_collection_stats()
            return {
                "message": "Vector store initialized successfully",
                "stats": stats
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to initialize vector store")
    except Exception as e:
        logger.error(f"Error initializing vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add")
async def add_embeddings_to_store(
    request: AddEmbeddingsRequest,
    store: Any = Depends(ensure_vector_store_initialized)
):
    """Add embeddings to the vector store."""
    try:
        chunks = request.chunks
        
        # Validate chunks have embeddings
        embeddings = []
        texts = []
        metadatas = []
        
        for chunk in chunks:
            if "embedding" not in chunk:
                raise HTTPException(
                    status_code=400,
                    detail="All chunks must have embeddings. Use /embeddings/chunks first."
                )
            
            embeddings.append(chunk["embedding"])
            texts.append(chunk.get("text", ""))
            
            # Create metadata without embedding (too large for metadata)
            metadata = {k: v for k, v in chunk.items() if k != "embedding"}
            metadatas.append(metadata)
        
        # Add to vector store
        success = store.add_embeddings(
            embeddings=embeddings,
            texts=texts,
            metadatas=metadatas
        )
        
        if success:
            stats = store.get_collection_stats()
            return {
                "message": f"Successfully added {len(chunks)} embeddings to vector store",
                "added_count": len(chunks),
                "total_count": stats.get("document_count", 0)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add embeddings to vector store")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search", response_model=SearchResponse)
async def search_similar_documents(
    request: SearchRequest,
    store: Any = Depends(ensure_vector_store_initialized)
):
    """Search for similar documents using text query."""
    try:
        # Ensure embedding service is loaded
        if embedding_service.model is None:
            success = embedding_service.load_model()
            if not success:
                raise HTTPException(status_code=500, detail="Failed to load embedding model")
        
        # Perform search
        results = store.search_by_text(
            query_text=request.query_text,
            embedding_service=embedding_service,
            n_results=request.n_results,
            where_filter=request.filter
        )
        
        return SearchResponse(
            documents=results["documents"],
            metadatas=results["metadatas"],
            distances=results["distances"],
            ids=results["ids"],
            count=len(results["documents"])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search-by-embedding", response_model=SearchResponse)
async def search_by_embedding(
    request: SearchByEmbeddingRequest,
    store: Any = Depends(ensure_vector_store_initialized)
):
    """Search for similar documents using embedding vector."""
    try:
        results = store.search_similar(
            query_embedding=request.query_embedding,
            n_results=request.n_results,
            where_filter=request.filter
        )
        
        return SearchResponse(
            documents=results["documents"],
            metadatas=results["metadatas"],
            distances=results["distances"],
            ids=results["ids"],
            count=len(results["documents"])
        )
        
    except Exception as e:
        logger.error(f"Error searching by embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents")
async def delete_documents(
    request: DeleteRequest,
    store: Any = Depends(ensure_vector_store_initialized)
):
    """Delete documents by metadata filter."""
    try:
        success = store.delete_by_filter(request.filter)
        
        if success:
            stats = store.get_collection_stats()
            return {
                "message": "Documents deleted successfully",
                "filter_used": request.filter,
                "remaining_count": stats.get("document_count", 0)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete documents")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset")
async def reset_vector_store(
    store: Any = Depends(ensure_vector_store_initialized)
):
    """Reset (clear) the vector store."""
    try:
        success = store.reset_collection()
        
        if success:
            return {"message": "Vector store reset successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reset vector store")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting vector store: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def vector_store_health():
    """Health check for vector store service."""
    stats = vector_store.get_collection_stats()
    return {
        "status": "healthy" if stats.get("initialized", False) else "not_initialized",
        "stats": stats
    }
