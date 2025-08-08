"""
Text Chunking Service

This service handles:
1. Splitting text into fixed-size chunks with overlap
2. Maintaining metadata for each chunk
3. Smart boundary detection for better semantic chunks
4. Optimizing chunk boundaries at sentence/paragraph breaks
"""

from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

def chunk_text(
    text_pages: List[Dict[str, Any]], 
    chunk_size: int = 1000, 
    overlap: int = 200,
    respect_boundaries: bool = True
) -> List[Dict[str, Any]]:
    """
    Split text into overlapping chunks with smart boundary detection.
    
    Args:
        text_pages: List of page texts with metadata
        chunk_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks
        respect_boundaries: Whether to respect sentence boundaries
        
    Returns:
        List of chunk dictionaries with text and metadata
    """
    chunks = []
    chunk_id = 0
    
    for page in text_pages:
        page_text = page["text"]
        page_chunks = _split_page_into_chunks(
            page_text, 
            chunk_size, 
            overlap, 
            respect_boundaries,
            page["page_number"],
            page["source_file"],
            page["file_type"]
        )
        
        # Add chunk IDs and update metadata
        for chunk in page_chunks:
            chunk["chunk_id"] = chunk_id
            chunk["global_chunk_index"] = chunk_id
            chunks.append(chunk)
            chunk_id += 1
    
    logger.info(f"Created {len(chunks)} chunks from {len(text_pages)} pages")
    return chunks

def _split_page_into_chunks(
    text: str, 
    chunk_size: int, 
    overlap: int,
    respect_boundaries: bool,
    page_number: int,
    source_file: str,
    file_type: str
) -> List[Dict[str, Any]]:
    """
    Split a single page into chunks.
    """
    if len(text) <= chunk_size:
        # Text fits in one chunk
        return [{
            "text": text,
            "page_number": page_number,
            "source_file": source_file,
            "file_type": file_type,
            "char_count": len(text),
            "word_count": len(text.split()),
            "chunk_index": 0,
            "start_char": 0,
            "end_char": len(text)
        }]
    
    chunks = []
    start = 0
    chunk_index = 0
    
    while start < len(text):
        # Calculate end position
        end = min(start + chunk_size, len(text))
        
        # If we're not at the end and respect_boundaries is True,
        # try to find a better breaking point
        if end < len(text) and respect_boundaries:
            end = _find_best_boundary(text, start, end)
        
        chunk_text = text[start:end].strip()
        
        if chunk_text:  # Only add non-empty chunks
            chunks.append({
                "text": chunk_text,
                "page_number": page_number,
                "source_file": source_file,
                "file_type": file_type,
                "char_count": len(chunk_text),
                "word_count": len(chunk_text.split()),
                "chunk_index": chunk_index,
                "start_char": start,
                "end_char": end
            })
            chunk_index += 1
        
        # Move start position with overlap
        start = max(start + chunk_size - overlap, end)
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks

def _find_best_boundary(text: str, start: int, max_end: int) -> int:
    """
    Find the best boundary for chunking within the given range.
    Priority: paragraph > sentence > word boundary
    """
    # Look for paragraph breaks (double newlines)
    search_text = text[start:max_end]
    
    # Try to find paragraph boundary (double newline)
    paragraph_match = None
    for match in re.finditer(r'\n\s*\n', search_text):
        paragraph_match = match
    
    if paragraph_match:
        return start + paragraph_match.end()
    
    # Try to find sentence boundary
    sentence_matches = list(re.finditer(r'[.!?]\s+', search_text))
    if sentence_matches:
        # Take the last sentence boundary
        last_sentence = sentence_matches[-1]
        return start + last_sentence.end()
    
    # Try to find word boundary
    # Look backwards from max_end to find last space
    for i in range(max_end - start - 1, 0, -1):
        if search_text[i] == ' ':
            return start + i + 1
    
    # If no good boundary found, use max_end
    return max_end

def get_chunk_stats(chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about the chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        Statistics dictionary
    """
    if not chunks:
        return {
            "total_chunks": 0,
            "total_chars": 0,
            "total_words": 0,
            "avg_chunk_size": 0,
            "min_chunk_size": 0,
            "max_chunk_size": 0
        }
    
    total_chars = sum(chunk["char_count"] for chunk in chunks)
    total_words = sum(chunk["word_count"] for chunk in chunks)
    chunk_sizes = [chunk["char_count"] for chunk in chunks]
    
    return {
        "total_chunks": len(chunks),
        "total_chars": total_chars,
        "total_words": total_words,
        "avg_chunk_size": total_chars // len(chunks),
        "min_chunk_size": min(chunk_sizes),
        "max_chunk_size": max(chunk_sizes),
        "source_files": list(set(chunk["source_file"] for chunk in chunks)),
        "file_types": list(set(chunk["file_type"] for chunk in chunks))
    }

def optimize_chunk_size(
    text_pages: List[Dict[str, Any]], 
    target_chunk_count: Optional[int] = None,
    max_chunk_size: int = 2000
) -> int:
    """
    Optimize chunk size based on document length and target chunk count.
    
    Args:
        text_pages: List of page texts
        target_chunk_count: Desired number of chunks (optional)
        max_chunk_size: Maximum allowed chunk size
        
    Returns:
        Optimized chunk size
    """
    total_chars = sum(len(page["text"]) for page in text_pages)
    
    if target_chunk_count:
        # Calculate chunk size to achieve target count
        optimal_size = total_chars // target_chunk_count
        return min(optimal_size, max_chunk_size)
    else:
        # Use default logic based on document size
        if total_chars < 5000:
            return 500  # Small documents: smaller chunks
        elif total_chars < 20000:
            return 1000  # Medium documents: standard chunks
        else:
            return 1500  # Large documents: larger chunks

def preview_chunks(
    text_pages: List[Dict[str, Any]], 
    chunk_size: int = 1000, 
    overlap: int = 200,
    preview_length: int = 100
) -> List[Dict[str, Any]]:
    """
    Create a preview of how the text would be chunked.
    
    Args:
        text_pages: List of page texts
        chunk_size: Chunk size to use
        overlap: Overlap size
        preview_length: Number of characters to show in preview
        
    Returns:
        List of chunk previews
    """
    chunks = chunk_text(text_pages, chunk_size, overlap)
    
    previews = []
    for chunk in chunks:
        text_preview = chunk["text"][:preview_length]
        if len(chunk["text"]) > preview_length:
            text_preview += "..."
        
        previews.append({
            "chunk_id": chunk["chunk_id"],
            "preview": text_preview,
            "char_count": chunk["char_count"],
            "page_number": chunk["page_number"],
            "source_file": chunk["source_file"]
        })
    
    return previews
