"""
Configuration management for multimodal RAG system.
Handles Google Gemini API keys, ChromaDB settings, and processing parameters.
"""

import os
from pathlib import Path
from typing import List, Optional


class MultimodalConfig:
    """Configuration settings for multimodal RAG system."""
    
    def __init__(self):
        # Google Gemini API
        self.google_api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        self.gemini_model: str = "gemini-1.5-flash"
        
        # File processing
        self.supported_formats: List[str] = ["pdf", "docx", "txt"]
        self.max_file_size_mb: int = 50
        
        # Directories
        self.temp_dir: Path = Path("data/temp")
        self.chroma_dir: Path = Path("data/chroma")
        
        # ChromaDB settings
        self.chroma_collection_prefix: str = "ragindocs_"
        
        # Create directories if they don't exist
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)
    
    def is_configured(self) -> bool:
        """Check if the system is properly configured."""
        return bool(self.google_api_key)


# Global configuration instance
config = MultimodalConfig()
