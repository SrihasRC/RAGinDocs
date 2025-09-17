from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    content_types: Optional[List[str]] = Field(default=["text", "tables", "images"], description="Content types to search")
    max_results: Optional[int] = Field(default=5, ge=1, le=20, description="Maximum number of results")

class QuerySource(BaseModel):
    type: str
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[QuerySource]
    metadata: Dict[str, Any]