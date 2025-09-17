from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentMetadata(BaseModel):
    id: str
    file_name: str
    file_type: str
    file_size: int
    upload_date: datetime
    page_count: Optional[int] = None
    processing_status: str = "pending"

class DocumentResponse(BaseModel):
    success: bool
    message: str
    document: Optional[DocumentMetadata] = None
    
class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total: int