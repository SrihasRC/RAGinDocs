"""
PDF Processing Service

This service handles:
1. Reading PDF files using PyMuPDF
2. Extracting text page by page
3. Cleaning and preprocessing text
4. Reading TXT and DOCX files
"""

from typing import List, Dict, Any
import fitz  # PyMuPDF
import logging
import os
from docx import Document

logger = logging.getLogger(__name__)

def extract_text_from_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from PDF file page by page.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing page text and metadata
    """
    try:
        doc = fitz.open(file_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Extract text from the page
            text = page.get_text()  # type: ignore
            
            # Clean the text
            text = clean_text(text)
            
            if text.strip():  # Only add pages with content
                pages.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "source_file": os.path.basename(file_path),
                    "file_type": "pdf",
                    "char_count": len(text)
                })
        
        doc.close()
        logger.info(f"Extracted text from {len(pages)} pages in {file_path}")
        return pages
        
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {str(e)}")
        raise

def extract_text_from_txt(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from TXT file.
    
    Args:
        file_path: Path to the TXT file
        
    Returns:
        List containing text and metadata
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Clean the text
        text = clean_text(text)
        
        return [{
            "page_number": 1,
            "text": text,
            "source_file": os.path.basename(file_path),
            "file_type": "txt",
            "char_count": len(text)
        }]
        
    except Exception as e:
        logger.error(f"Error extracting text from TXT {file_path}: {str(e)}")
        raise

def extract_text_from_docx(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from DOCX file.
    
    Args:
        file_path: Path to the DOCX file
        
    Returns:
        List containing text and metadata
    """
    try:
        doc = Document(file_path)
        text = ""
        
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        # Clean the text
        text = clean_text(text)
        
        return [{
            "page_number": 1,
            "text": text,
            "source_file": os.path.basename(file_path),
            "file_type": "docx",
            "char_count": len(text)
        }]
        
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
        raise

def clean_text(text: str) -> str:
    """
    Clean and preprocess extracted text.
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Remove special characters that might interfere with processing
    # Keep basic punctuation for semantic meaning
    import re
    text = re.sub(r'[^\w\s\.\,\?\!\;\:\-\(\)]', ' ', text)
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def extract_text_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract text from any supported file type.
    
    Args:
        file_path: Path to the file
        
    Returns:
        List of dictionaries containing text and metadata
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def get_file_stats(pages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get statistics about the extracted text.
    
    Args:
        pages: List of page dictionaries
        
    Returns:
        Statistics dictionary
    """
    if not pages:
        return {"total_pages": 0, "total_chars": 0, "total_words": 0}
    
    total_chars = sum(page["char_count"] for page in pages)
    total_words = sum(len(page["text"].split()) for page in pages)
    
    return {
        "total_pages": len(pages),
        "total_chars": total_chars,
        "total_words": total_words,
        "file_type": pages[0]["file_type"],
        "source_file": pages[0]["source_file"]
    }
