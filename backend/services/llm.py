"""
LLM service for generating intelligent responses using OpenRouter API.
Supports various models including free options like Llama 3.1 8B Instruct.
"""

import os
import logging
from typing import Dict, Any, Optional
import httpx
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)

class LLMService:
    """Service for generating responses using OpenRouter LLM API."""
    
    def __init__(self):
        """Initialize the LLM service."""
        from dotenv import load_dotenv
        load_dotenv()  # Ensure env vars are loaded
        
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1"
        self.default_model = "qwen/qwen-2-7b-instruct:free"  # Reliable free model
        self.max_tokens = 1000
        self.temperature = 0.7
        
        # Debug logging
        logger.info(f"LLM Service initialized. API key configured: {bool(self.api_key)}")
        if self.api_key:
            logger.info(f"API key preview: {self.api_key[:20]}...")
        
        # Updated Model configurations with actually available free models
        self.models = {
            "qwen-7b": {
                "name": "qwen/qwen-2-7b-instruct:free",
                "max_tokens": 1000,
                "temperature": 0.7,
                "description": "Qwen 2 7B Instruct (Free) - Reliable and fast"
            },
            "gemma-7b": {
                "name": "google/gemma-2-9b-it:free",
                "max_tokens": 1000,
                "temperature": 0.7,
                "description": "Google Gemma 2 9B IT (Free) - Good performance"
            },
            "llama-3.2-3b": {
                "name": "meta-llama/llama-3.2-3b-instruct:free",
                "max_tokens": 1000,
                "temperature": 0.7,
                "description": "Meta Llama 3.2 3B Instruct (Free) - Limited rate"
            },
            "phi-3-mini": {
                "name": "microsoft/phi-3-mini-128k-instruct:free",
                "max_tokens": 1000,
                "temperature": 0.7,
                "description": "Microsoft Phi-3 Mini (Free) - Backup option"
            }
        }
        
        # Model fallback order for rate limiting - start with most reliable
        self.fallback_models = ["qwen-7b", "gemma-7b", "phi-3-mini", "llama-3.2-3b"]
        
        # Rate limiting tracking
        self.rate_limit_tracker = {}  # {model_key: reset_timestamp}
        self.current_model_index = 0  # Track which model we're currently using
        
    def is_configured(self) -> bool:
        """Check if the LLM service is properly configured."""
        return bool(self.api_key)
    
    def _is_model_available(self, model_key: str) -> bool:
        """Check if a model is available (not rate limited)."""
        if model_key not in self.rate_limit_tracker:
            return True
        
        reset_time = self.rate_limit_tracker[model_key]
        current_time = time.time()
        
        if current_time >= reset_time:
            # Rate limit has been reset, remove from tracker
            del self.rate_limit_tracker[model_key]
            return True
        
        return False
    
    def _mark_model_rate_limited(self, model_key: str, reset_timestamp: Optional[float] = None):
        """Mark a model as rate limited with reset time."""
        if reset_timestamp is None:
            # Default to 60 seconds if we don't have the exact reset time
            reset_timestamp = time.time() + 60
        
        self.rate_limit_tracker[model_key] = reset_timestamp
        logger.info(f"Model {model_key} rate limited until {datetime.fromtimestamp(reset_timestamp)}")
    
    def _get_best_available_model(self) -> Optional[str]:
        """Get the best available model that's not rate limited."""
        # First, try models in order of preference
        for model_key in self.fallback_models:
            if self._is_model_available(model_key):
                return model_key
        
        # If all models are rate limited, return the one with the earliest reset time
        if self.rate_limit_tracker:
            earliest_reset = min(self.rate_limit_tracker.items(), key=lambda x: x[1])
            logger.warning(f"All models rate limited. Using {earliest_reset[0]} (resets in {int(earliest_reset[1] - time.time())}s)")
            return earliest_reset[0]
        
        # Fallback to first model
        return self.fallback_models[0]
    
    def _extract_reset_time(self, error_str: str) -> Optional[float]:
        """Extract rate limit reset time from error message."""
        try:
            # Look for reset timestamp in OpenRouter error format
            if "X-RateLimit-Reset" in error_str:
                import re
                match = re.search(r'"X-RateLimit-Reset":"(\d+)"', error_str)
                if match:
                    reset_timestamp = int(match.group(1)) / 1000  # Convert from milliseconds
                    return reset_timestamp
            
            # If we can't parse the exact time, default to 60 seconds from now
            return time.time() + 60
            
        except Exception:
            # Fallback to 60 seconds
            return time.time() + 60
    
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
        Uses smart model rotation with rate limit tracking.
        
        Args:
            query: User's original question
            context: Retrieved context from documents
            model: Model to use (if None, uses smart rotation)
            max_tokens: Maximum tokens in response
            temperature: Response creativity (0.0-1.0)
            
        Returns:
            Dictionary with generated response and metadata
        """
        if not self.is_configured():
            logger.warning("LLM service not configured - API key missing")
            return self._create_fallback_response(query, context)
        
        # Smart model selection
        if model:
            # User specified a model - try it first, then fallback
            models_to_try = [model] if self._is_model_available(model) else []
            models_to_try.extend([m for m in self.fallback_models if m != model and self._is_model_available(m)])
            if not models_to_try:
                # All models rate limited, use the best one anyway
                models_to_try = [self._get_best_available_model()]
        else:
            # Use smart rotation - get the best available model
            best_model = self._get_best_available_model()
            models_to_try = [best_model] if best_model else self.fallback_models
            
        last_error = None
        
        for model_key in models_to_try:
            if model_key not in self.models:
                continue
                
            try:
                model_config = self.models[model_key]
                
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
                
                # Success! This model is working
                logger.info(f"Successfully used model: {model_key}")
                
                return {
                    "status": "success",
                    "response": response_data["content"],
                    "model_used": model_config["description"],
                    "prompt_tokens": response_data.get("prompt_tokens", 0),
                    "completion_tokens": response_data.get("completion_tokens", 0),
                    "total_tokens": response_data.get("total_tokens", 0),
                    "timestamp": datetime.now().isoformat(),
                    "fallback_used": model_key != (model or self.fallback_models[0]),
                    "rate_limited_models": list(self.rate_limit_tracker.keys())
                }
                
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                
                # Check if it's a rate limit error
                if "429" in error_str or "rate limit" in error_str.lower():
                    # Try to extract reset time from error message
                    reset_time = self._extract_reset_time(error_str)
                    self._mark_model_rate_limited(model_key, reset_time)
                    logger.warning(f"Rate limit hit for {model_key}, trying next model...")
                    continue
                elif "404" in error_str or "no endpoints found" in error_str.lower():
                    logger.warning(f"Model {model_key} not available, trying next model...")
                    continue
                else:
                    # For other errors, don't try other models
                    logger.error(f"Error generating LLM response with {model_key}: {error_str}")
                    break
        
        # All models failed
        logger.error(f"All models failed. Last error: {last_error}")
        return self._create_error_response(last_error or "Unknown error", query, context)
    
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
        if is_list_query:
            instruction = """Extract and present the requested information in a clear, organized list format. Use numbering or bullet points as appropriate."""
        elif is_definition_query:
            instruction = """Provide a clear, concise definition or explanation based on the information available."""
        elif is_how_query:
            instruction = """Explain the process or method step-by-step, organizing the information logically."""
        elif is_structured:
            instruction = """Present the information in a well-organized, structured format."""
        else:
            instruction = """Answer the question directly and concisely, organizing the information clearly."""

        prompt = f"""You are an intelligent document assistant. Your task is to provide accurate, well-structured answers based on the provided context.

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION: {query}

TASK: {instruction}

INSTRUCTIONS:
1. Answer using ONLY the information provided in the context
2. Structure your response clearly and logically
3. Be concise but complete
4. Use clean, professional formatting:
   - For headings, use simple text without excessive formatting
   - For lists, use numbered lists (1. 2. 3.) or bullet points (•)
   - For emphasis, use **bold** sparingly and only for key terms
5. If information is incomplete, state what's missing
6. Don't invent or assume information not in the context
7. Present information in a readable, organized manner that flows naturally
8. Focus on the specific part of the question (e.g., if asked about a specific section, chapter, or module, extract only that relevant information)

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
            "default": "qwen-7b",
            "configured": self.is_configured()
        }
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get LLM service status including rate limit information."""
        current_time = time.time()
        rate_limit_status = {}
        
        for model_key in self.fallback_models:
            if model_key in self.rate_limit_tracker:
                reset_time = self.rate_limit_tracker[model_key]
                seconds_until_reset = max(0, int(reset_time - current_time))
                rate_limit_status[model_key] = {
                    "rate_limited": True,
                    "resets_in_seconds": seconds_until_reset,
                    "reset_time": datetime.fromtimestamp(reset_time).isoformat()
                }
            else:
                rate_limit_status[model_key] = {
                    "rate_limited": False,
                    "resets_in_seconds": 0,
                    "reset_time": None
                }
        
        return {
            "service": "llm",
            "configured": self.is_configured(),
            "default_model": self.default_model,
            "available_models": len(self.models),
            "api_base": self.base_url,
            "best_available_model": self._get_best_available_model(),
            "rate_limits": rate_limit_status
        }

# Global instance
llm_service = LLMService()
