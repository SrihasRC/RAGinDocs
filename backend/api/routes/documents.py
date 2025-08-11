"""
Simple API routes for multimodal RAG system.
Clean foundation for future implementation.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    
class QueryResponse(BaseModel):
    answer: str
    status: str

# Create router
router = APIRouter(prefix="/api/v2", tags=["multimodal-rag"])

@router.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query documents using multimodal RAG (placeholder)."""
    return QueryResponse(
        answer="Multimodal RAG not implemented yet. Check back soon!",
        status="development"
    )

@router.get("/status")
async def get_status():
    """Get system status."""
    return {
        "status": "development",
        "version": "2.0.0",
        "message": "Multimodal RAG system in development"
    }
