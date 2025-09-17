from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Dict, Any

from models.query import QueryRequest, QueryResponse
from services.rag_service import RAGService

router = APIRouter(prefix="/query", tags=["query"])

# Initialize RAG service
rag_service = RAGService()

@router.post("/ask", response_model=QueryResponse)
async def ask_question(query_request: QueryRequest):
    """Process a multimodal RAG query"""
    try:
        if not query_request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        response = await rag_service.process_query(query_request)
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )

@router.get("/similar")
async def get_similar_documents(
    query: str,
    content_types: str = "text,tables,images",
    k: int = 10
):
    """Get similar documents without generating an answer"""
    try:
        if not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        # Parse content types
        content_type_list = [ct.strip() for ct in content_types.split(",")]
        
        results = await rag_service.get_similar_documents(
            query=query,
            content_types=content_type_list,
            k=k
        )
        
        return JSONResponse(content={
            "query": query,
            "results": results,
            "total_results": sum(len(docs) for docs in results.values())
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting similar documents: {str(e)}"
        )

@router.post("/analyze-intent")
async def analyze_query_intent(query: str):
    """Analyze query to suggest best content types"""
    try:
        if not query.strip():
            raise HTTPException(
                status_code=400,
                detail="Query cannot be empty"
            )
        
        analysis = await rag_service.analyze_query_intent(query)
        
        return JSONResponse(content={
            "query": query,
            "analysis": analysis
        })
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing query intent: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Check if query service is healthy"""
    try:
        # Simple health check - just verify services are accessible
        # Don't perform actual AI operations to avoid rate limits
        vector_store_accessible = rag_service.vector_store is not None
        llm_configured = rag_service.llm is not None
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "query_processor", 
            "vector_store_accessible": vector_store_accessible,
            "llm_configured": llm_configured
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "query_processor",
                "error": str(e)
            }
        )
