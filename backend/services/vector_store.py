"""
Vector storage service using ChromaDB for embedding storage and similarity search.
"""

import chromadb
from chromadb.config import Settings
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

class VectorStoreService:
    """Service for vector storage and similarity search using ChromaDB."""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize the vector store service.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client: Optional[ClientAPI] = None
        self.collection: Optional[Collection] = None
        self.collection_name = "ragindocs_embeddings"
        
    def initialize(self) -> bool:
        """
        Initialize ChromaDB client and collection.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info(f"Initializing ChromaDB in {self.persist_directory}")
            
            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAGinDocs document embeddings"}
            )
            
            logger.info(f"ChromaDB initialized. Collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {str(e)}")
            return False
    
    def add_embeddings(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> bool:
        """
        Add embeddings to the vector store.
        
        Args:
            embeddings: List of embedding vectors
            texts: List of text chunks
            metadatas: List of metadata dictionaries
            ids: Optional list of custom IDs (auto-generated if None)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.collection:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        if not (len(embeddings) == len(texts) == len(metadatas)):
            raise ValueError("Embeddings, texts, and metadatas must have same length")
        
        try:
            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(embeddings))]
            
            # Convert metadata to strings for ChromaDB compatibility
            processed_metadatas = []
            for metadata in metadatas:
                processed_metadata = {}
                for key, value in metadata.items():
                    if isinstance(value, (dict, list)):
                        processed_metadata[key] = json.dumps(value)
                    else:
                        processed_metadata[key] = str(value)
                processed_metadatas.append(processed_metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,  # type: ignore
                documents=texts,
                metadatas=processed_metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(embeddings)} embeddings to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add embeddings: {str(e)}")
            return False
    
    def search_similar(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            Dictionary with search results
        """
        if not self.collection:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        try:
            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],  # type: ignore
                n_results=n_results,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # Process results
            processed_results = {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": [],
                "distances": results["distances"][0] if results["distances"] else [],
                "ids": results["ids"][0] if results["ids"] else []
            }
            
            # Process metadata (convert JSON strings back to objects)
            if results["metadatas"] and results["metadatas"][0]:
                for metadata in results["metadatas"][0]:
                    processed_metadata = {}
                    for key, value in metadata.items():
                        try:
                            # Try to parse as JSON first (only if it's a string)
                            if isinstance(value, str):
                                processed_metadata[key] = json.loads(value)
                            else:
                                processed_metadata[key] = value
                        except (json.JSONDecodeError, TypeError):
                            # Keep as original value if not JSON
                            processed_metadata[key] = value
                    processed_results["metadatas"].append(processed_metadata)
            
            logger.info(f"Found {len(processed_results['documents'])} similar documents")
            return processed_results
            
        except Exception as e:
            logger.error(f"Failed to search similar embeddings: {str(e)}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": []
            }
    
    def search_by_text(
        self,
        query_text: str,
        embedding_service,
        n_results: int = 5,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Search using text query (will be converted to embedding).
        
        Args:
            query_text: Query text
            embedding_service: Embedding service to convert text to vector
            n_results: Number of results to return
            where_filter: Optional metadata filter
            
        Returns:
            Dictionary with search results
        """
        try:
            # Convert query to embedding
            query_embedding = embedding_service.generate_single_embedding(query_text)
            
            # Search using embedding
            return self.search_similar(
                query_embedding=query_embedding,
                n_results=n_results,
                where_filter=where_filter
            )
            
        except Exception as e:
            logger.error(f"Failed to search by text: {str(e)}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": []
            }
    
    def delete_by_filter(self, where_filter: Dict[str, Any]) -> bool:
        """
        Delete embeddings by metadata filter.
        
        Args:
            where_filter: Metadata filter for deletion
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.collection:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        try:
            self.collection.delete(where=where_filter)
            logger.info(f"Deleted embeddings matching filter: {where_filter}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings: {str(e)}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        if not self.collection:
            return {"initialized": False}
        
        try:
            count = self.collection.count()
            return {
                "initialized": True,
                "collection_name": self.collection_name,
                "document_count": count,
                "persist_directory": str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {
                "initialized": True,
                "error": str(e)
            }
    
    def reset_collection(self) -> bool:
        """
        Reset (clear) the collection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            raise RuntimeError("Vector store not initialized. Call initialize() first.")
        
        try:
            # Delete existing collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "RAGinDocs document embeddings"}
            )
            
            logger.info(f"Reset collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset collection: {str(e)}")
            return False

# Global instance for reuse across requests
vector_store = VectorStoreService()
