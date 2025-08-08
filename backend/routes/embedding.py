"""
Embedding API routes for vector generation and similarity computation.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from services.embedding import embedding_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

# Request/Response Models
class EmbedTextRequest(BaseModel):
    """Request model for text embedding."""
    text: str = Field(..., description="Text to generate embedding for")

class EmbedTextsRequest(BaseModel):
    """Request model for multiple text embeddings."""
    texts: List[str] = Field(..., description="List of texts to generate embeddings for")

class EmbedChunksRequest(BaseModel):
    """Request model for chunk embedding."""
    chunks: List[Dict[str, Any]] = Field(..., description="List of text chunks to embed")

class EmbeddingResponse(BaseModel):
    """Response model for single embedding."""
    embedding: List[float] = Field(..., description="Generated embedding vector")
    model_name: str = Field(..., description="Name of the embedding model used")
    dimension: int = Field(..., description="Dimension of the embedding vector")

class EmbeddingsResponse(BaseModel):
    """Response model for multiple embeddings."""
    embeddings: List[List[float]] = Field(..., description="List of generated embedding vectors")
    model_name: str = Field(..., description="Name of the embedding model used")
    dimension: int = Field(..., description="Dimension of the embedding vectors")
    count: int = Field(..., description="Number of embeddings generated")

class ChunksWithEmbeddingsResponse(BaseModel):
    """Response model for chunks with embeddings."""
    chunks: List[Dict[str, Any]] = Field(..., description="Chunks enhanced with embeddings")
    model_name: str = Field(..., description="Name of the embedding model used")
    count: int = Field(..., description="Number of chunks processed")

class SimilarityRequest(BaseModel):
    """Request model for similarity computation."""
    embedding1: List[float] = Field(..., description="First embedding vector")
    embedding2: List[float] = Field(..., description="Second embedding vector")

class SimilarityResponse(BaseModel):
    """Response model for similarity computation."""
    similarity: float = Field(..., description="Cosine similarity score between -1 and 1")

class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    loaded: bool = Field(..., description="Whether the model is loaded")
    model_name: Optional[str] = Field(None, description="Name of the loaded model")
    embedding_dimension: Optional[int] = Field(None, description="Dimension of embeddings")
    max_sequence_length: Optional[str] = Field(None, description="Maximum sequence length")

# Dependency to ensure model is loaded
async def ensure_model_loaded():
    """Ensure the embedding model is loaded before processing requests."""
    if embedding_service.model is None:
        logger.info("Loading embedding model...")
        success = embedding_service.load_model()
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to load embedding model"
            )
    return embedding_service

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the embedding model."""
    try:
        info = embedding_service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/model/load")
async def load_model():
    """Load the embedding model."""
    try:
        success = embedding_service.load_model()
        if success:
            return {"message": "Model loaded successfully", "model_name": embedding_service.model_name}
        else:
            raise HTTPException(status_code=500, detail="Failed to load model")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text", response_model=EmbeddingResponse)
async def embed_text(
    request: EmbedTextRequest,
    service: Any = Depends(ensure_model_loaded)
):
    """Generate embedding for a single text."""
    try:
        embedding = service.generate_single_embedding(request.text)
        
        return EmbeddingResponse(
            embedding=embedding,
            model_name=service.model_name,
            dimension=len(embedding)
        )
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/texts", response_model=EmbeddingsResponse)
async def embed_texts(
    request: EmbedTextsRequest,
    service: Any = Depends(ensure_model_loaded)
):
    """Generate embeddings for multiple texts."""
    try:
        embeddings = service.generate_embeddings(request.texts)
        
        return EmbeddingsResponse(
            embeddings=embeddings,
            model_name=service.model_name,
            dimension=len(embeddings[0]) if embeddings else 0,
            count=len(embeddings)
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chunks", response_model=ChunksWithEmbeddingsResponse)
async def embed_chunks(
    request: EmbedChunksRequest,
    service: Any = Depends(ensure_model_loaded)
):
    """Process chunks and add embeddings to each."""
    try:
        enhanced_chunks = service.process_chunks_with_embeddings(request.chunks)
        
        return ChunksWithEmbeddingsResponse(
            chunks=enhanced_chunks,
            model_name=service.model_name,
            count=len(enhanced_chunks)
        )
    except Exception as e:
        logger.error(f"Error embedding chunks: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/similarity", response_model=SimilarityResponse)
async def compute_similarity(
    request: SimilarityRequest,
    service: Any = Depends(ensure_model_loaded)
):
    """Compute cosine similarity between two embeddings."""
    try:
        similarity = service.compute_similarity(request.embedding1, request.embedding2)
        
        return SimilarityResponse(similarity=similarity)
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def embedding_health():
    """Health check for embedding service."""
    model_info = embedding_service.get_model_info()
    return {
        "status": "healthy" if model_info["loaded"] else "model_not_loaded",
        "model_info": model_info
    }
