from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import upload, query, embedding, vector
import uvicorn

# Create FastAPI app
app = FastAPI(
    title="RAGinDocs API",
    description="Real-time QA + Intelligent Action Generation from PDFs/Docs using Hybrid RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router)
app.include_router(query.router)
app.include_router(embedding.router)
app.include_router(vector.router)

@app.get("/")
def read_root():
    return {
        "message": "Welcome to RAGinDocs API!",
        "version": "1.0.0",
        "phase": "Phase 1 - Core RAG Pipeline",
        "docs": "/docs",
        "endpoints": {
            "upload": "/upload/document",
            "query": "/query/document", 
            "embeddings": "/embeddings/text",
            "vector": "/vector/search",
            "health": "/query/health"
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "phase": "Phase 1",
        "services": {
            "upload": "ready",
            "query": "ready",
            "embeddings": "ready",
            "vector_db": "ready"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )