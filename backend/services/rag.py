"""
RAG (Retrieval-Augmented Generation) service for intelligent query processing.
Combines document retrieval with context-aware response generation.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re
from datetime import datetime

from services.embedding import embedding_service
from services.vector_store import vector_store
from services.llm import llm_service

logger = logging.getLogger(__name__)

class RAGService:
    """Service for RAG-based query processing and intelligent responses."""
    
    def __init__(self):
        """Initialize the RAG service."""
        self.max_context_length = 4000  # Balanced for better coverage without overwhelming LLM
        self.min_similarity_threshold = 0.15  # Reasonable threshold for quality
        self.default_top_k = 8  # Moderate results for good coverage
                # Generic query enhancement patterns that work across domains
        self.query_enhancement_keywords = {
            # Academic/Educational
            "outcomes": "completion student able",
            "objectives": "goals purposes aims",
            "modules": "topics sections chapters",
            "module": "topics sections chapters contents",
            "evaluation": "assessment grading testing",
            "references": "books sources bibliography",
            
            # Business/Technical
            "requirements": "specifications needs criteria",
            "features": "capabilities functions characteristics",
            "process": "steps procedures methodology",
            "benefits": "advantages value proposition",
            
            # General
            "definition": "meaning explanation description",
            "examples": "instances cases samples",
            "comparison": "differences similarities contrast",
            "summary": "overview synopsis highlights"
        }
        
    def _enhance_query(self, query: str) -> str:
        """Enhance the query with contextual keywords for better retrieval."""
        enhanced_query = query.lower()
        
        # Special handling for module-specific queries
        module_match = re.search(r'module\s*(\d+)', enhanced_query)
        if module_match:
            module_num = module_match.group(1)
            # Add multiple variations to catch the module
            enhanced_query += f" Module:{module_num} Module {module_num} module{module_num}"
        
        # Add contextual keywords based on query content for general queries
        for key_phrase, enhancement in self.query_enhancement_keywords.items():
            if key_phrase.lower() in enhanced_query:
                # Only add a subset of enhancement to avoid noise
                enhancement_words = enhancement.split()[:3]  # Take first 3 words only
                enhanced_query += f" {' '.join(enhancement_words)}"
                break  # Only apply the first matching enhancement to avoid noise
        
        # Add semantic variations for common query patterns
        query_words = enhanced_query.split()
        
        # For "what is" questions, add definition-related terms
        if "what" in query_words and "is" in query_words:
            enhanced_query += " definition meaning"
        
        # For "how to" questions, add process-related terms  
        elif "how" in query_words:
            enhanced_query += " steps process"
            
        # For listing questions, add enumeration terms
        elif any(word in query_words for word in ["list", "enumerate", "identify"]):
            enhanced_query += " items elements"
        
        return enhanced_query
        
    async def process_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        document_filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        use_llm: bool = True,
        llm_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a query using RAG approach.
        
        Args:
            query: User query text
            top_k: Number of top chunks to retrieve
            similarity_threshold: Minimum similarity score
            document_filter: Optional filter for specific documents
            include_metadata: Whether to include chunk metadata
            use_llm: Whether to use LLM for generating response
            llm_model: Specific LLM model to use
            
        Returns:
            Dictionary with query results and context
        """
        try:
            # Use defaults if not provided
            top_k = top_k or self.default_top_k
            similarity_threshold = similarity_threshold or self.min_similarity_threshold
            
            # Enhance query for better retrieval
            enhanced_query = self._enhance_query(query)
            
            logger.info(f"Processing query: '{query}' (enhanced: '{enhanced_query[:50]}...')")
            
            # Step 1: Initialize services if needed
            if not self._ensure_services_ready():
                return self._create_error_response("Services not available")
            
            # Step 2: Retrieve relevant context
            retrieval_results = self._retrieve_context(
                query=enhanced_query,  # Use enhanced query
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
            response = await self._create_response(
                query=query,
                enhanced_query=enhanced_query,
                context=context,
                retrieval_results=filtered_results,
                include_metadata=include_metadata,
                use_llm=use_llm,
                llm_model=llm_model
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
        
        logger.info(f"Filtering with threshold: {threshold}")
        
        for i, distance in enumerate(results.get("distances", [])):
            # Convert ChromaDB cosine distance to similarity score
            # ChromaDB cosine distance: 0 = identical, 2 = completely different
            # Convert to similarity: 0-1 scale where 1 = identical, 0 = completely different
            similarity = max(0.0, 1.0 - (distance / 2.0))
            logger.info(f"Chunk {i}: distance={distance:.4f}, similarity={similarity:.4f}")
            
            if similarity >= threshold:
                filtered_results["documents"].append(results["documents"][i])
                filtered_results["metadatas"].append(results["metadatas"][i])
                filtered_results["distances"].append(distance)
                filtered_results["ids"].append(results["ids"][i])
                logger.info(f"✓ Chunk {i} included (similarity {similarity:.4f} >= {threshold})")
            else:
                logger.info(f"✗ Chunk {i} filtered out (similarity {similarity:.4f} < {threshold})")
        
        logger.info(f"After filtering: {len(filtered_results['documents'])} chunks remain")
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
    
    async def _create_response(
        self,
        query: str,
        enhanced_query: str,
        context: str,
        retrieval_results: Dict[str, Any],
        include_metadata: bool = True,
        use_llm: bool = True,
        llm_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create comprehensive response with retrieved context."""
        # Calculate similarity scores using proper ChromaDB cosine distance conversion
        similarities = [max(0.0, 1.0 - (dist / 2.0)) for dist in retrieval_results["distances"]]
        
        # Prepare source information
        sources = []
        if include_metadata:
            # Debug logging to see what we're getting
            logger.info(f"RAG Debug - retrieval_results keys: {retrieval_results.keys()}")
            logger.info(f"RAG Debug - IDs: {retrieval_results.get('ids', 'No IDs found')}")
            logger.info(f"RAG Debug - IDs type: {type(retrieval_results.get('ids', None))}")
            
            for i, metadata in enumerate(retrieval_results["metadatas"]):
                # Get chunk_id from the search results IDs, not from metadata
                chunk_id = retrieval_results["ids"][i] if i < len(retrieval_results["ids"]) else f"chunk_{i}"
                logger.info(f"RAG Debug - Chunk {i}: chunk_id={chunk_id}, type={type(chunk_id)}")
                source = {
                    "chunk_id": str(chunk_id),  # Ensure it's a string
                    "source_file": metadata.get("source_file", "unknown"),
                    "page_number": metadata.get("page_number", 0),
                    "similarity_score": round(similarities[i], 4),
                    "text_preview": retrieval_results["documents"][i][:100] + "..." if len(retrieval_results["documents"][i]) > 100 else retrieval_results["documents"][i]
                }
                sources.append(source)
        
        # Generate intelligent response using LLM or fallback
        llm_info = {
            "used": False,
            "model": None,
            "tokens": 0,
            "status": None,
            "error": None,
            "reason": None,
            "fallback": False
        }
        
        if use_llm and llm_service.is_configured():
            try:
                logger.info(f"Attempting LLM generation with model: {llm_model or 'default'}")
                llm_response = await llm_service.generate_response(
                    query=query,
                    context=context,
                    model=llm_model or "llama-3.2-3b"
                )
                
                # Check if the LLM actually succeeded or returned an error/fallback
                if llm_response.get("status") == "success":
                    suggested_answer = llm_response["response"]
                    llm_info = {
                        "used": True,
                        "model": llm_response.get("model_used", "unknown"),
                        "tokens": llm_response.get("total_tokens", 0),
                        "status": llm_response.get("status", "success"),
                        "error": None,
                        "reason": None,
                        "fallback": False
                    }
                    logger.info(f"LLM generation successful: {llm_info['tokens']} tokens, model: {llm_info['model']}")
                else:
                    # LLM service returned error or fallback response
                    suggested_answer = llm_response["response"]
                    llm_info = {
                        "used": False,
                        "model": None,
                        "tokens": 0,
                        "status": llm_response.get("status", "error"),
                        "error": llm_response.get("error", "LLM service returned non-success status"),
                        "reason": "LLM service failed",
                        "fallback": True
                    }
                    logger.warning(f"LLM returned non-success status: {llm_response.get('status')}")
                    
            except Exception as e:
                logger.error(f"LLM generation failed: {str(e)}")
                suggested_answer = self._generate_suggested_answer(query, context)
                llm_info = {
                    "used": False,
                    "model": None,
                    "tokens": 0,
                    "status": "error",
                    "error": str(e),
                    "reason": "LLM request failed",
                    "fallback": True
                }
        else:
            suggested_answer = self._generate_suggested_answer(query, context)
            if not llm_service.is_configured():
                llm_info["reason"] = "LLM service not configured"
            else:
                llm_info["reason"] = "LLM disabled by user"
        
        response = {
            "query": query,
            "enhanced_query": enhanced_query,
            "timestamp": datetime.now().isoformat(),
            "status": "success",
            "results": {
                "context": context,
                "suggested_answer": suggested_answer,
                "chunks_found": len(retrieval_results["documents"]),
                "sources": sources if include_metadata else [],
                "llm_info": llm_info,
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
