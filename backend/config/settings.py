"""
Configuration management for multimodal RAG system.
Handles Google Gemini API keys, ChromaDB settings, and processing parameters.
"""

import os
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Suppress Google gRPC warnings for local development
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_TRACE", "")
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "")
os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "")

class MultimodalConfig:
    """Configuration settings for multimodal RAG system."""
    
    def __init__(self):
        # API Keys
        self.google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
        self.groq_api_key: str = os.getenv("GROQ_API_KEY", "")
        
        # File processing settings
        self.allowed_extensions: List[str] = ['.pdf', '.docx']
        self.max_file_size_mb: int = 50
        
        # Document processing settings
        self.chunk_size: int = 1000
        self.chunk_overlap: int = 200
        self.max_chars_per_chunk: int = 10000
        
        # Directory settings
        self.temp_dir: Path = Path("data/temp")
        self.chroma_dir: Path = Path("data/chroma")
        self.upload_dir: Path = Path("data/uploads")
        
        # ChromaDB collection settings
        self.chroma_collection_prefix: str = "ragindocs_"
        self.text_collection: str = "text_summaries"
        self.table_collection: str = "table_summaries"  
        self.image_collection: str = "image_summaries"
        self.metadata_collection: str = "document_metadata"
        
        # Create directories if they don't exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    def is_configured(self) -> bool:
        """Check if the system is properly configured."""
        return bool(self.google_api_key)
    
    def get_collection_name(self, collection_type: str) -> str:
        """Get full collection name with prefix."""
        return f"{self.chroma_collection_prefix}{collection_type}"

# Global configuration instance
config = MultimodalConfig()
