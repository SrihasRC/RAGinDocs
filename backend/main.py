from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import documents, queries
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="RAGinDocs Backend Service",
    description="Backend service for RAGinDocs application",
    version="2.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(queries.router, prefix="/api/queries", tags=["Queries"])

@app.get("/")
async def root():
    return {"message": "Welcome to the RAGinDocs Backend Service!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)