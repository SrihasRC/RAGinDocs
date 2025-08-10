"""
LLM service for generating intelligent responses using OpenRouter API.
Supports various models including free options like Llama 3.1 8B Instruct.
"""

import os
import logging
from typing import Dict, Any, Optional
import httpx
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMService:
    """Service for generating responses using OpenRouter LLM API."""
    
    def __init__(self):
        """Initialize the LLM service."""
        from dotenv import load_dotenv
        load_dotenv()  # Ensure env vars are loaded
        
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_model = "meta-llama/llama-3.2-3b-instruct:free"  # Updated free model
        self.max_tokens = 1000
        self.temperature = 0.7
        
        # Debug logging
        logger.info(f"LLM Service initialized. API key configured: {bool(self.api_key)}")
        if self.api_key:
            logger.info(f"API key preview: {self.api_key[:20]}...")
        
        # Updated Model configurations with correct free models
        self.models = {
            "llama-3.1-8b": {
                "name": "meta-llama/llama-3.1-8b-instruct:free",
                "max_tokens": 1000,
                "temperature": 0.7,
                "description": "Meta Llama 3.1 8B Instruct (Free)"
            },
            "llama-3.2-3b": {
                "name": "meta-llama/llama-3.2-3b-instruct:free",
                "max_tokens": 1000,
                "temperature": 0.7,
                "description": "Meta Llama 3.2 3B Instruct (Free)"
            },
            "qwen-7b": {
                "name": "qwen/qwen-2-7b-instruct:free",
                "max_tokens": 1000,
                "temperature": 0.7,
                "description": "Qwen 2 7B Instruct (Free)"
            },
            "gemma-7b": {
                "name": "google/gemma-2-9b-it:free",
                "max_tokens": 1000,
                "temperature": 0.7,
                "description": "Google Gemma 2 9B IT (Free)"
            }
        }
        
    def is_configured(self) -> bool:
        """Check if the LLM service is properly configured."""
        return bool(self.api_key)
    
    async def generate_response(
        self,
        query: str,
        context: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Generate an intelligent response based on query and context.
        
        Args:
            query: User's original question
            context: Retrieved context from documents
            model: Model to use (defaults to llama-3.1-8b)
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-1.0)
            
        Returns:
            Dictionary with generated response and metadata
        """
        if not self.is_configured():
            logger.warning("LLM service not configured - API key missing")
            return self._create_fallback_response(query, context)
        
        try:
            # Use model configuration
            model_key = model or "llama-3.2-3b"
            model_config = self.models.get(model_key, self.models["llama-3.2-3b"])
            
            # Override with provided parameters
            final_max_tokens = max_tokens or model_config["max_tokens"]
            final_temperature = temperature if temperature is not None else model_config["temperature"]
            
            # Create the prompt
            prompt = self._create_rag_prompt(query, context)
            
            # Make API request
            response_data = await self._call_openrouter_api(
                model=model_config["name"],
                prompt=prompt,
                max_tokens=final_max_tokens,
                temperature=final_temperature
            )
            
            return {
                "status": "success",
                "response": response_data["content"],
                "model_used": model_config["description"],
                "prompt_tokens": response_data.get("prompt_tokens", 0),
                "completion_tokens": response_data.get("completion_tokens", 0),
                "total_tokens": response_data.get("total_tokens", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            return self._create_error_response(str(e), query, context)
    
    def _create_rag_prompt(self, query: str, context: str) -> str:
        """Create an adaptive prompt for RAG-based Q&A that works with any document type."""
        if not context.strip():
            return f"""You are a helpful AI assistant. The user asked: "{query}"

I don't have any relevant documents to answer this question. Please provide a helpful response explaining that no relevant information was found and suggest how they might rephrase their question or what additional information might be needed."""

        # Analyze the context to understand document type and structure
        context_lower = context.lower()
        query_lower = query.lower()
        
        # Determine if this is a structured document (syllabus, manual, etc.)
        is_structured = any(indicator in context_lower for indicator in [
            'objectives', 'outcomes', 'modules', 'chapter', 'section', 
            'references', 'bibliography', 'evaluation', 'assessment'
        ])
        
        # Determine query intent
        is_list_query = any(word in query_lower for word in [
            'list', 'what are', 'enumerate', 'identify', 'name'
        ])
        
        is_definition_query = any(word in query_lower for word in [
            'what is', 'define', 'definition', 'meaning', 'explain'
        ])
        
        is_how_query = any(word in query_lower for word in [
            'how', 'steps', 'process', 'method', 'procedure'
        ])

        # Build adaptive instructions
        if is_structured and is_list_query:
            instruction = """Extract and present the requested information in a clear, organized list format. Use numbering or bullet points as appropriate."""
        elif is_definition_query:
            instruction = """Provide a clear, concise definition or explanation based on the information available."""
        elif is_how_query:
            instruction = """Explain the process or method step-by-step, organizing the information logically."""
        elif is_structured:
            instruction = """Present the information in a well-organized, structured format that matches the document's style."""
        else:
            instruction = """Answer the question directly and concisely, organizing the information clearly."""

        prompt = f"""You are an intelligent document assistant. Your task is to provide accurate, well-structured answers based on the provided context.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {query}

TASK: {instruction}

FORMATTING GUIDELINES:
1. Answer using ONLY the information provided in the context
2. Structure your response clearly and logically
3. Be concise but complete
4. Use clean, professional formatting:
   - For headings, use simple text without excessive formatting
   - For lists, use numbered lists (1. 2. 3.) or bullet points (•)
   - For emphasis, use **bold** sparingly and only for key terms
   - Avoid excessive markdown formatting like multiple asterisks
5. If information is incomplete, state what's missing
6. Don't invent or assume information not in the context
7. Present information in a readable, organized manner that flows naturally

ANSWER:"""
        
        return prompt
    
    async def _call_openrouter_api(
        self,
        model: str,
        prompt: str,
        max_tokens: int,
        temperature: float
    ) -> Dict[str, Any]:
        """Make API call to OpenRouter."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",  # Optional: your app URL
            "X-Title": "RAGinDocs"  # Optional: your app name
        }
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code != 200:
                error_detail = response.text
                logger.error(f"OpenRouter API error: {response.status_code} - {error_detail}")
                raise Exception(f"API request failed: {response.status_code} - {error_detail}")
            
            data = response.json()
            
            # Extract response content and usage info
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            
            return {
                "content": content.strip(),
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
    
    def _create_fallback_response(self, query: str, context: str) -> Dict[str, Any]:
        """Create an intelligent fallback response when LLM is not available."""
        if not context.strip():
            response = "I couldn't find relevant information in the documents to answer your question."
        else:
            # Intelligent parsing of context for structured response
            query_lower = query.lower()
            
            # Check if this looks like a list request
            if any(word in query_lower for word in ['list', 'what are', 'outcomes', 'objectives', 'features', 'steps']):
                # Try to extract numbered/bulleted items
                lines = context.split('\n')
                items = []
                for line in lines:
                    line = line.strip()
                    # Look for numbered items (1., 2., etc.) or bullet points
                    if any(line.startswith(str(i) + '.') for i in range(1, 20)) or line.startswith('•') or line.startswith('-'):
                        items.append(line)
                
                if items:
                    response = "Based on the documents:\n\n" + "\n".join(items)
                else:
                    # Fall back to showing first 300 chars in a structured way
                    response = f"Based on the retrieved documents:\n\n{context[:300]}"
                    if len(context) > 300:
                        response += "..."
            else:
                # For other questions, provide a more focused excerpt
                response = f"Based on the documents: {context[:400]}"
                if len(context) > 400:
                    response += "..."
        
        return {
            "status": "fallback",
            "response": response,
            "model_used": "Intelligent fallback (LLM unavailable)",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "timestamp": datetime.now().isoformat(),
            "note": "Using intelligent fallback. For AI-powered responses, ensure LLM service is available."
        }
    
    def _create_error_response(self, error: str, query: str, context: str) -> Dict[str, Any]:
        """Create an error response with fallback content."""
        fallback = self._create_fallback_response(query, context)
        return {
            "status": "error",
            "error": error,
            "response": fallback["response"],
            "model_used": "Error fallback",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_available_models(self) -> Dict[str, Any]:
        """Get list of available models."""
        return {
            "models": self.models,
            "default": "llama-3.2-3b",
            "configured": self.is_configured()
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get LLM service status."""
        return {
            "service": "llm",
            "configured": self.is_configured(),
            "default_model": self.default_model,
            "available_models": len(self.models),
            "api_base": self.base_url
        }

# Global instance
llm_service = LLMService()
