import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load environment variables
load_dotenv()

# Import routes
from api.routes.documents import router as documents_router

# Create FastAPI app
app = FastAPI(
    title="RAGinDocs Multimodal API",
    description="Advanced multimodal RAG system with text, table, and image processing",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Specific frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(documents_router)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to RAGinDocs Multimodal API!",
        "version": "2.0.0",
        "status": "development",
        "features": [
            "Multimodal document processing (text, tables, images)",
            "Google Gemini 1.5 Flash integration",
            "Multi-vector ChromaDB storage",
            "Summary-based retrieval"
        ],
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "system": "multimodal-rag",
        "ready": True
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )