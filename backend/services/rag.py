"""
RAG (Retrieval-Augmented Generation) service for intelligent query processing.
Combines document retrieval with context-aware response generation.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import datetime

from services.embedding import embedding_service
from services.vector_store import vector_store

logger = logging.getLogger(__name__)

class RAGService:
    """Service for RAG-based query processing and intelligent responses."""
    
    def __init__(self):
        """Initialize the RAG service."""
        self.max_context_length = 4000  # Maximum characters for context
        self.min_similarity_threshold = 0.3  # Minimum similarity score to include
        self.default_top_k = 5  # Default number of chunks to retrieve
        
    def process_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        document_filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Process a query using RAG approach.
        
        Args:
            query: User query text
            top_k: Number of top chunks to retrieve
            similarity_threshold: Minimum similarity score
            document_filter: Optional filter for specific documents
            include_metadata: Whether to include chunk metadata
            
        Returns:
            Dictionary with query results and context
        """
        try:
            # Use defaults if not provided
            top_k = top_k or self.default_top_k
            similarity_threshold = similarity_threshold or self.min_similarity_threshold
            
            logger.info(f"Processing query: '{query[:50]}...'")
            
            # Step 1: Initialize services if needed
            if not self._ensure_services_ready():
                return self._create_error_response("Services not available")
            
            # Step 2: Retrieve relevant context
            retrieval_results = self._retrieve_context(
                query=query,
                top_k=top_k,
                document_filter=document_filter
            )
            
            if not retrieval_results["documents"]:
                return self._create_no_results_response(query)
            
            # Step 3: Filter by similarity threshold
            filtered_results = self._filter_by_similarity(
                retrieval_results, 
                similarity_threshold
            )
            
            # Step 4: Build context from retrieved chunks
            context = self._build_context(filtered_results)
            
            # Step 5: Create comprehensive response
            response = self._create_response(
                query=query,
                context=context,
                retrieval_results=filtered_results,
                include_metadata=include_metadata
            )
            
            logger.info(f"Query processed successfully. Found {len(filtered_results['documents'])} relevant chunks.")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return self._create_error_response(str(e))
    
    def _ensure_services_ready(self) -> bool:
        """Ensure all required services are initialized."""
        try:
            # Check vector store
            if vector_store.collection is None:
                logger.info("Initializing vector store...")
                if not vector_store.initialize():
                    return False
            
            # Check embedding service
            if embedding_service.model is None:
                logger.info("Loading embedding model...")
                if not embedding_service.load_model():
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {str(e)}")
            return False
    
    def _retrieve_context(
        self, 
        query: str, 
        top_k: int,
        document_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Retrieve relevant context using vector similarity search."""
        try:
            # Perform vector search
            results = vector_store.search_by_text(
                query_text=query,
                embedding_service=embedding_service,
                n_results=top_k,
                where_filter=document_filter
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {str(e)}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "ids": []
            }
    
    def _filter_by_similarity(
        self, 
        results: Dict[str, Any], 
        threshold: float
    ) -> Dict[str, Any]:
        """Filter results by similarity threshold (lower distance = higher similarity)."""
        filtered_results = {
            "documents": [],
            "metadatas": [],
            "distances": [],
            "ids": []
        }
        
        for i, distance in enumerate(results.get("distances", [])):
            # Convert distance to similarity: similarity = 1 - distance
            similarity = 1 - distance
            
            if similarity >= threshold:
                filtered_results["documents"].append(results["documents"][i])
                filtered_results["metadatas"].append(results["metadatas"][i])
                filtered_results["distances"].append(distance)
                filtered_results["ids"].append(results["ids"][i])
        
        return filtered_results
    
    def _build_context(self, results: Dict[str, Any]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        current_length = 0
        
        for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
            # Create context entry with source info
            source_info = ""
            if metadata:
                source_file = metadata.get("source_file", "Unknown")
                page_num = metadata.get("page_number", "?")
                source_info = f" [Source: {source_file}, Page {page_num}]"
            
            context_entry = f"Context {i+1}{source_info}:\n{doc}\n"
            
            # Check if adding this entry would exceed max length
            if current_length + len(context_entry) > self.max_context_length:
                logger.info(f"Context truncated at {current_length} characters")
                break
            
            context_parts.append(context_entry)
            current_length += len(context_entry)
        
        return "\n".join(context_parts)
    
    def _create_response(
        self,
        query: str,
        context: str,
        retrieval_results: Dict[str, Any],
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Create comprehensive response with retrieved context."""
        # Calculate similarity scores (1 - distance)
        similarities = [1 - dist for dist in retrieval_results["distances"]]
        
        # Prepare source information
        sources = []
        if include_metadata:
            for i, metadata in enumerate(retrieval_results["metadatas"]):
                source = {
                    "chunk_id": metadata.get("chunk_id", f"chunk_{i}"),
                    "source_file": metadata.get("source_file", "unknown"),
                    "page_number": metadata.get("page_number", 0),
                    "similarity_score": round(similarities[i], 4),
                    "text_preview": retrieval_results["documents"][i][:100] + "..." if len(retrieval_results["documents"][i]) > 100 else retrieval_results["documents"][i]
                }
                sources.append(source)
        
        # Create suggested answers based on context
        suggested_answer = self._generate_suggested_answer(query, context)
        
        response = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "results": {
                "context": context,
                "suggested_answer": suggested_answer,
                "chunks_found": len(retrieval_results["documents"]),
                "sources": sources if include_metadata else [],
                "retrieval_stats": {
                    "top_similarity": round(max(similarities), 4) if similarities else 0,
                    "avg_similarity": round(sum(similarities) / len(similarities), 4) if similarities else 0,
                    "context_length": len(context)
                }
            }
        }
        
        return response
    
    def _generate_suggested_answer(self, query: str, context: str) -> str:
        """Generate a suggested answer based on the query and retrieved context."""
        if not context.strip():
            return "No relevant information found in the documents."
        
        # Simple rule-based answer generation
        # In a production system, this would use a language model
        
        # Analyze query type
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what is", "what are", "define", "definition"]):
            suggestion = f"Based on the retrieved documents, here's what I found about your question:\n\n{context[:500]}..."
        elif any(word in query_lower for word in ["how to", "how do", "how can", "steps"]):
            suggestion = f"Here are the relevant steps and information from the documents:\n\n{context[:500]}..."
        elif any(word in query_lower for word in ["why", "reason", "because"]):
            suggestion = f"The documents provide the following explanation:\n\n{context[:500]}..."
        elif any(word in query_lower for word in ["when", "time", "date"]):
            suggestion = f"Regarding the timing information in your query:\n\n{context[:500]}..."
        elif any(word in query_lower for word in ["where", "location", "place"]):
            suggestion = f"The documents mention the following location information:\n\n{context[:500]}..."
        else:
            suggestion = f"Based on the retrieved context, here's the relevant information:\n\n{context[:500]}..."
        
        if len(context) > 500:
            suggestion += "\n\n(Note: This is a summary. Check the full context for complete information.)"
        
        return suggestion
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create error response."""
        return {
            "query": "",
            "timestamp": datetime.now().isoformat(),
            "status": "error",
            "error": error_message,
            "results": {
                "context": "",
                "suggested_answer": f"An error occurred: {error_message}",
                "chunks_found": 0,
                "sources": [],
                "retrieval_stats": {
                    "top_similarity": 0,
                    "avg_similarity": 0,
                    "context_length": 0
                }
            }
        }
    
    def _create_no_results_response(self, query: str) -> Dict[str, Any]:
        """Create response when no relevant documents are found."""
        return {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "status": "no_results",
            "results": {
                "context": "",
                "suggested_answer": "I couldn't find relevant information in the uploaded documents for your query. Try rephrasing your question or uploading more relevant documents.",
                "chunks_found": 0,
                "sources": [],
                "retrieval_stats": {
                    "top_similarity": 0,
                    "avg_similarity": 0,
                    "context_length": 0
                }
            }
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of RAG service and dependencies."""
        vector_stats = vector_store.get_collection_stats()
        embedding_info = embedding_service.get_model_info()
        
        return {
            "rag_service": "ready",
            "vector_store": {
                "status": "ready" if vector_stats.get("initialized", False) else "not_initialized",
                "document_count": vector_stats.get("document_count", 0),
                "collection": vector_stats.get("collection_name", "unknown")
            },
            "embedding_service": {
                "status": "ready" if embedding_info.get("loaded", False) else "not_loaded",
                "model": embedding_info.get("model_name", "unknown"),
                "dimension": embedding_info.get("embedding_dimension", 0)
            }
        }

# Global instance for reuse across requests
rag_service = RAGService()
