"""
Embedding service for converting text chunks to vector representations
using sentence-transformers models for semantic similarity.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings from text chunks."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding service with a pre-trained model.
        
        Args:
            model_name: Name of the sentence-transformer model to use
                       Default: all-MiniLM-L6-v2 (fast, good quality, 384 dimensions)
        """
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dim: Optional[int] = None
        
    def load_model(self) -> bool:
        """
        Load the sentence transformer model.
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Get embedding dimension by encoding a test sentence
            test_embedding = self.model.encode("test")
            self.embedding_dim = len(test_embedding)
            
            logger.info(f"Model loaded successfully. Embedding dimension: {self.embedding_dim}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            return False
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors (each vector is a list of floats)
            
        Raises:
            RuntimeError: If model is not loaded
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not texts:
            return []
        
        try:
            logger.info(f"Generating embeddings for {len(texts)} text chunks")
            
            # Generate embeddings (returns numpy array)
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                normalize_embeddings=True  # Normalize for better similarity computation
            )
            
            # Convert to list of lists for JSON serialization
            embeddings_list = embeddings.tolist()
            
            logger.info(f"Generated {len(embeddings_list)} embeddings successfully")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            
        Returns:
            Embedding vector as list of floats
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Compute cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {str(e)}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "max_sequence_length": getattr(self.model, 'max_seq_length', 'unknown')
        }
    
    def process_chunks_with_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process text chunks and add embeddings to each chunk.
        
        Args:
            chunks: List of chunk dictionaries (from chunking service)
            
        Returns:
            List of chunks with embeddings added
        """
        if not chunks:
            return []
        
        try:
            # Extract text content from chunks
            texts = [chunk.get('text', '') for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.generate_embeddings(texts)
            
            # Add embeddings to chunks
            enhanced_chunks = []
            for chunk, embedding in zip(chunks, embeddings):
                enhanced_chunk = chunk.copy()
                enhanced_chunk['embedding'] = embedding
                enhanced_chunk['embedding_model'] = self.model_name
                enhanced_chunks.append(enhanced_chunk)
            
            logger.info(f"Enhanced {len(enhanced_chunks)} chunks with embeddings")
            return enhanced_chunks
            
        except Exception as e:
            logger.error(f"Failed to process chunks with embeddings: {str(e)}")
            raise

# Global instance for reuse across requests
embedding_service = EmbeddingService()
