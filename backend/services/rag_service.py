from typing import Dict, List, Any, Optional
import json

# LangChain imports
from langchain.schema.document import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from services.shared_instances import get_vector_store
from models.query import QueryRequest, QueryResponse, QuerySource
from config.settings import config

class RAGService:
    """LangChain-based RAG service for multimodal query processing"""
    
    def __init__(self):
        self.vector_store = get_vector_store()
        
        # Initialize the LLM only if API key is available
        if config.google_api_key and config.google_api_key != "your_google_api_key_here":
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.1,
                google_api_key=config.google_api_key
            )
        else:
            print("⚠️  Warning: Google API key not configured. RAG functionality will be limited.")
            print("   Please set GOOGLE_API_KEY in your .env file to enable AI features.")
            self.llm = None
        
        # Create the prompt template for RAG
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are an intelligent assistant that answers questions based on the provided context. 
The context may include text content, tables, and image descriptions from documents.

Context:
{context}

Question: {question}

Please provide a comprehensive answer based on the context provided. If the context contains:
- Text content: Use it to answer the question directly
- Tables: Explain relevant data points and relationships
- Images: Reference visual information when relevant

If you cannot answer the question based on the provided context, please say so clearly.

Answer:"""
        )
    
    async def process_query(self, query_request: QueryRequest) -> QueryResponse:
        """Process a multimodal RAG query"""
        try:
            # Step 1: Retrieve relevant documents
            search_results = await self.vector_store.search_multimodal(
                query=query_request.question,
                content_types=query_request.content_types,
                k=query_request.max_results if query_request.max_results else 5
            )
            
            # Step 2: Prepare context from all content types
            context_documents = []
            sources = []
            
            # Process text results
            if "text" in search_results:
                for doc in search_results["text"]:
                    context_documents.append(doc)
                    sources.append(QuerySource(
                        type="text",
                        content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        metadata=doc.metadata
                    ))
            
            # Process table results
            if "tables" in search_results:
                for doc in search_results["tables"]:
                    context_documents.append(doc)
                    sources.append(QuerySource(
                        type="table",
                        content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        metadata=doc.metadata
                    ))
            
            # Process image results
            if "images" in search_results:
                for doc in search_results["images"]:
                    context_documents.append(doc)
                    sources.append(QuerySource(
                        type="image",
                        content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        metadata=doc.metadata
                    ))
            
            # Step 3: Generate answer using LangChain
            if context_documents:
                answer = await self._generate_answer(query_request.question, context_documents)
            else:
                answer = "I couldn't find relevant information in the documents to answer your question."
            
            return QueryResponse(
                query=query_request.question,
                answer=answer,
                sources=sources,
                metadata={
                    "total_sources": len(sources),
                    "content_types_found": list(search_results.keys()),
                    "model_used": "gemini-2.0-flash-exp"
                }
            )
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return QueryResponse(
                query=query_request.question,
                answer=f"An error occurred while processing your query: {str(e)}",
                sources=[],
                metadata={"error": str(e)}
            )
    
    async def _generate_answer(self, query: str, documents: List[Document]) -> str:
        """Generate answer using LangChain document chain"""
        try:
            if not self.llm:
                return "AI answer generation is not available. Please configure GOOGLE_API_KEY in your .env file."
            
            # Create the document chain
            document_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=self.prompt_template
            )
            
            # Run the chain
            result = await document_chain.ainvoke({
                "context": documents,
                "question": query
            })
            
            return result
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    async def get_similar_documents(self, query: str, content_types: Optional[List[str]] = None, k: int = 10) -> Dict[str, List[Dict]]:
        """Get similar documents without generating an answer"""
        try:
            search_results = await self.vector_store.search_multimodal(
                query=query,
                content_types=content_types,
                k=k
            )
            
            # Format results for API response
            formatted_results = {}
            for content_type, documents in search_results.items():
                formatted_results[content_type] = [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "score": getattr(doc, "score", None)
                    }
                    for doc in documents
                ]
            
            return formatted_results
            
        except Exception as e:
            print(f"Error getting similar documents: {e}")
            return {}
    
    async def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to suggest best content types to search"""
        try:
            # Simple heuristics for content type suggestion
            query_lower = query.lower()
            
            suggested_types = []
            confidence_scores = {}
            
            # Text indicators
            text_keywords = ["explain", "describe", "what", "how", "why", "definition", "meaning"]
            if any(keyword in query_lower for keyword in text_keywords):
                suggested_types.append("text")
                confidence_scores["text"] = 0.8
            
            # Table indicators
            table_keywords = ["data", "numbers", "statistics", "compare", "table", "chart", "figures"]
            if any(keyword in query_lower for keyword in table_keywords):
                suggested_types.append("tables")
                confidence_scores["tables"] = 0.7
            
            # Image indicators
            image_keywords = ["image", "picture", "diagram", "graph", "visual", "chart", "figure"]
            if any(keyword in query_lower for keyword in image_keywords):
                suggested_types.append("images")
                confidence_scores["images"] = 0.6
            
            # Default to all types if no specific indicators
            if not suggested_types:
                suggested_types = ["text", "tables", "images"]
                confidence_scores = {"text": 0.5, "tables": 0.3, "images": 0.3}
            
            return {
                "suggested_content_types": suggested_types,
                "confidence_scores": confidence_scores,
                "analysis": {
                    "query_length": len(query),
                    "has_question_words": any(q in query_lower for q in ["what", "how", "why", "when", "where"]),
                    "is_comparison": "compare" in query_lower or "vs" in query_lower,
                    "seeks_data": any(d in query_lower for d in ["data", "number", "statistic"])
                }
            }
            
        except Exception as e:
            print(f"Error analyzing query intent: {e}")
            return {
                "suggested_content_types": ["text", "tables", "images"],
                "confidence_scores": {"text": 0.5, "tables": 0.3, "images": 0.3},
                "analysis": {"error": str(e)}
            }
