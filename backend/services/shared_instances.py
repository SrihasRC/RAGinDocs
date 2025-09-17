"""
Shared service instances to ensure consistency across routes
"""

from services.vector_store import LangChainVectorStore
from services.document_processor import LangChainDocumentProcessor

# Singleton instances
_vector_store_instance = None
_document_processor_instance = None

def get_vector_store() -> LangChainVectorStore:
    """Get the shared vector store instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = LangChainVectorStore()
    return _vector_store_instance

def get_document_processor() -> LangChainDocumentProcessor:
    """Get the shared document processor instance"""
    global _document_processor_instance
    if _document_processor_instance is None:
        _document_processor_instance = LangChainDocumentProcessor()
    return _document_processor_instance
