from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class DocumentMetadata(BaseModel):
    id: str
    fileName: str
    fileType: str
    fileSize: int
    uploadDate: datetime
    pageCount: Optional[int] = None
    processingStatus: str = "pending"

class DocumentResponse(BaseModel):
    success: bool
    message: str
    document: Optional[DocumentMetadata] = None
    
class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total: int