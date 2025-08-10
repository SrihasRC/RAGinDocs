"""
Query API routes for RAG-based document querying and intelligent responses.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from services.rag import rag_service
from services.llm import llm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["query"])

# Request/Response Models
class QueryRequest(BaseModel):
    """Request model for document querying."""
    query: str = Field(..., description="User query text", min_length=1)
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    similarity_threshold: Optional[float] = Field(default=0.3, ge=0.0, le=1.0, description="Minimum similarity threshold")
    document_filter: Optional[Dict[str, Any]] = Field(None, description="Filter for specific documents")
    include_metadata: bool = Field(default=True, description="Include chunk metadata in response")
    use_llm: bool = Field(default=True, description="Use LLM for intelligent response generation")
    llm_model: Optional[str] = Field(None, description="Specific LLM model to use")

class LLMInfo(BaseModel):
    """Information about LLM usage."""
    used: bool = Field(..., description="Whether LLM was used")
    model: Optional[str] = Field(None, description="Model used for generation")
    tokens: Optional[int] = Field(None, description="Total tokens used")
    status: Optional[str] = Field(None, description="Generation status")
    error: Optional[str] = Field(None, description="Error message if failed")
    reason: Optional[str] = Field(None, description="Reason for not using LLM")
    fallback: Optional[bool] = Field(None, description="Whether fallback was used")

class SourceInfo(BaseModel):
    """Source information for retrieved chunks."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    source_file: str = Field(..., description="Source document filename")
    page_number: int = Field(..., description="Page number in source document")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    text_preview: str = Field(..., description="Preview of chunk text")

class RetrievalStats(BaseModel):
    """Statistics about the retrieval process."""
    top_similarity: float = Field(..., description="Highest similarity score found")
    avg_similarity: float = Field(..., description="Average similarity score")
    context_length: int = Field(..., description="Total context length in characters")

class QueryResults(BaseModel):
    """Results from query processing."""
    context: str = Field(..., description="Retrieved context from documents")
    suggested_answer: str = Field(..., description="AI-suggested answer based on context")
    chunks_found: int = Field(..., description="Number of relevant chunks found")
    sources: List[SourceInfo] = Field(..., description="Source information for chunks")
    llm_info: LLMInfo = Field(..., description="Information about LLM usage")
    retrieval_stats: RetrievalStats = Field(..., description="Retrieval statistics")

class QueryResponse(BaseModel):
    """Response model for document queries."""
    query: str = Field(..., description="Original user query")
    enhanced_query: Optional[str] = Field(None, description="Enhanced query used for retrieval")
    timestamp: str = Field(..., description="Query processing timestamp")
    status: str = Field(..., description="Query processing status")
    results: QueryResults = Field(..., description="Query results and context")
    timestamp: str = Field(..., description="Query processing timestamp")
    status: str = Field(..., description="Query status (success/error/no_results)")
    results: QueryResults = Field(..., description="Query results and context")
    error: Optional[str] = Field(None, description="Error message if status is error")

class SimpleQueryRequest(BaseModel):
    """Simple query request for quick testing."""
    query: str = Field(..., description="User query text")

class ServiceStatusResponse(BaseModel):
    """Response model for service status."""
    rag_service: str = Field(..., description="RAG service status")
    vector_store: Dict[str, Any] = Field(..., description="Vector store status")
    embedding_service: Dict[str, Any] = Field(..., description="Embedding service status")

# Dependency to ensure RAG service is ready
async def ensure_rag_service_ready():
    """Ensure the RAG service is ready before processing requests."""
    try:
        status = rag_service.get_service_status()
        
        # Check if core services are ready
        if (status["vector_store"]["status"] != "ready" or 
            status["embedding_service"]["status"] != "ready"):
            raise HTTPException(
                status_code=503,
                detail="RAG service dependencies not ready. Please upload documents first."
            )
        
        return rag_service
        
    except Exception as e:
        logger.error(f"RAG service not ready: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"RAG service not available: {str(e)}"
        )

@router.get("/status", response_model=ServiceStatusResponse)
async def get_query_service_status():
    """Get status of RAG query service and dependencies."""
    try:
        status = rag_service.get_service_status()
        return ServiceStatusResponse(**status)
    except Exception as e:
        logger.error(f"Error getting service status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/document", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    service: Any = Depends(ensure_rag_service_ready)
):
    """
    Query documents using RAG (Retrieval-Augmented Generation) with LLM integration.
    
    This endpoint:
    1. Converts the query to embeddings
    2. Searches the vector database for similar chunks
    3. Retrieves relevant context from documents
    4. Uses LLM to generate intelligent responses (if configured)
    5. Provides a suggested answer based on the context
    """
    try:
        logger.info(f"Processing query: '{request.query[:50]}...'")
        
        # Process query using RAG service (now async)
        result = await service.process_query(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            document_filter=request.document_filter,
            include_metadata=request.include_metadata,
            use_llm=request.use_llm,
            llm_model=request.llm_model
        )
        
        # Convert to response model
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Unknown error"))
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simple")
async def simple_query(
    request: SimpleQueryRequest,
    service: Any = Depends(ensure_rag_service_ready)
):
    """
    Simple query endpoint for quick testing without detailed metadata.
    """
    try:
        result = await service.process_query(
            query=request.query,
            top_k=3,
            similarity_threshold=0.3,
            include_metadata=False,
            use_llm=True  # Enable LLM for simple queries too
        )
        
        if result["status"] == "error":
            return {"error": result.get("error", "Unknown error")}
        
        return {
            "query": request.query,
            "answer": result["results"]["suggested_answer"],
            "chunks_found": result["results"]["chunks_found"],
            "status": result["status"]
        }
        
    except Exception as e:
        logger.error(f"Error in simple query: {str(e)}")
        return {"error": str(e)}

@router.get("/llm/models")
async def get_llm_models():
    """Get available LLM models and configuration status."""
    try:
        models_info = llm_service.get_available_models()
        service_status = llm_service.get_service_status()
        
        return {
            "available_models": models_info["models"],
            "default_model": models_info["default"],
            "service_configured": models_info["configured"],
            "service_status": service_status,
            "recommendation": {
                "free_model": "llama-3.1-8b",
                "description": "Meta Llama 3.1 8B Instruct - Fast, free, and excellent for Q&A",
                "setup_note": "Set OPENROUTER_API_KEY environment variable to enable LLM responses"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting LLM models: {str(e)}")
        return {
            "error": str(e),
            "available_models": {},
            "service_configured": False
        }

@router.get("/health")
async def query_health():
    """Health check for query service."""
    try:
        status = rag_service.get_service_status()
        llm_status = llm_service.get_service_status()
        
        health_status = "healthy"
        if status["vector_store"]["status"] != "ready":
            health_status = "vector_store_not_ready"
        elif status["embedding_service"]["status"] != "ready":
            health_status = "embedding_service_not_ready"
        
        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                **status,
                "llm_service": llm_status
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
