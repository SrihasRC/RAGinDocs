import uuid
from typing import Dict, List, Any, Optional
import asyncio

# LangChain imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

from config.settings import config

class LangChainVectorStore:
    """LangChain-based vector store using ChromaDB with multi-vector retrieval"""
    
    def __init__(self):
        # Initialize embeddings (using local model from your packages)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create separate vector stores for different content types
        self.text_vectorstore = Chroma(
            collection_name=config.get_collection_name(config.text_collection),
            embedding_function=self.embeddings,
            persist_directory=str(config.chroma_dir)
        )
        
        self.table_vectorstore = Chroma(
            collection_name=config.get_collection_name(config.table_collection),
            embedding_function=self.embeddings,
            persist_directory=str(config.chroma_dir)
        )
        
        self.image_vectorstore = Chroma(
            collection_name=config.get_collection_name(config.image_collection),
            embedding_function=self.embeddings,
            persist_directory=str(config.chroma_dir)
        )
        
        # Document stores for original content (following the reference notebook pattern)
        self.text_docstore = InMemoryStore()
        self.table_docstore = InMemoryStore()
        self.image_docstore = InMemoryStore()
        
        # Multi-vector retrievers (this is the key pattern from the reference)
        self.text_retriever = MultiVectorRetriever(
            vectorstore=self.text_vectorstore,
            docstore=self.text_docstore,
            id_key="doc_id"
        )
        
        self.table_retriever = MultiVectorRetriever(
            vectorstore=self.table_vectorstore,
            docstore=self.table_docstore,
            id_key="doc_id"
        )
        
        self.image_retriever = MultiVectorRetriever(
            vectorstore=self.image_vectorstore,
            docstore=self.image_docstore,
            id_key="doc_id"
        )
    
    def _filter_metadata(self, metadata: dict) -> dict:
        """Filter out complex metadata that ChromaDB can't handle"""
        filtered = {}
        for key, value in metadata.items():
            # Only keep simple types that ChromaDB accepts
            if isinstance(value, (str, int, float, bool)) or value is None:
                filtered[key] = value
            elif isinstance(value, (list, dict)):
                # Convert complex types to strings
                filtered[key] = str(value)
        return filtered
    
    async def store_document_content(self, processed_content: Dict[str, Any]) -> bool:
        """Store processed document content in vector stores"""
        try:
            # Get document-level metadata
            doc_metadata = processed_content.get("metadata", {})
            
            # Store text documents
            await self._store_text_documents(processed_content["text_documents"], doc_metadata)
            
            # Store table documents
            await self._store_table_documents(processed_content["table_documents"], doc_metadata)
            
            # Store image documents
            await self._store_image_documents(processed_content["image_documents"], doc_metadata)
            
            return True
        except Exception as e:
            print(f"Error storing document content: {e}")
            return False
    
    async def _store_text_documents(self, documents: List[Document], doc_metadata: Optional[Dict[str, Any]] = None):
        """Store text documents using multi-vector pattern following reference implementation"""
        if not documents:
            return
        
        # Generate unique chunk IDs for ChromaDB storage, but preserve original doc_id
        chunk_ids = [str(uuid.uuid4()) for _ in documents]
        
        # Create summary documents for vector search (what gets embedded)
        summary_docs = []
        original_docs = []
        
        for i, doc in enumerate(documents):
            # Merge document-level metadata with chunk metadata
            chunk_metadata = doc.metadata.copy()
            if doc_metadata:
                chunk_metadata.update(doc_metadata)
            
            # Filter complex metadata to avoid ChromaDB errors
            clean_metadata = self._filter_metadata(chunk_metadata)
            # Keep original doc_id for document-level grouping, but add unique chunk_id for storage  
            clean_metadata["chunk_id"] = chunk_ids[i]
            # doc_id stays the same for all chunks of the same document
            
            # Summary document for vector store (uses the AI-generated summary)
            summary_doc = Document(
                page_content=doc.page_content,  # This is the AI summary
                metadata=clean_metadata
            )
            summary_docs.append(summary_doc)
            
            # Original document for docstore (includes original content)
            original_doc = Document(
                page_content=doc.metadata.get("original_content", doc.page_content),
                metadata=clean_metadata
            )
            original_docs.append(original_doc)
        
        # Add summaries to vector store for similarity search
        await asyncio.to_thread(
            self.text_vectorstore.add_documents, summary_docs
        )
        
        # Store original documents in docstore for retrieval (following reference pattern)
        self.text_docstore.mset(list(zip(chunk_ids, original_docs)))
    
    async def _store_table_documents(self, documents: List[Document], doc_metadata: Optional[Dict[str, Any]] = None):
        """Store table documents using multi-vector pattern"""
        if not documents:
            return
        
        # Generate unique chunk IDs for ChromaDB storage, but preserve original doc_id
        chunk_ids = [str(uuid.uuid4()) for _ in documents]
        
        summary_docs = []
        original_docs = []
        
        for i, doc in enumerate(documents):
            # Merge document-level metadata with chunk metadata
            chunk_metadata = doc.metadata.copy()
            if doc_metadata:
                chunk_metadata.update(doc_metadata)
            
            # Filter complex metadata to avoid ChromaDB errors
            clean_metadata = self._filter_metadata(chunk_metadata)
            # Keep original doc_id for document-level grouping, but add unique chunk_id for storage
            clean_metadata["chunk_id"] = chunk_ids[i]
            # doc_id stays the same for all chunks of the same document
            
            # Summary document for vector store
            summary_doc = Document(
                page_content=doc.page_content,  # AI summary
                metadata=clean_metadata
            )
            summary_docs.append(summary_doc)
            
            # Original document for docstore
            original_doc = Document(
                page_content=doc.metadata.get("original_content", doc.page_content),
                metadata=clean_metadata
            )
            original_docs.append(original_doc)
        
        await asyncio.to_thread(
            self.table_vectorstore.add_documents, summary_docs
        )
        
        self.table_docstore.mset(list(zip(chunk_ids, original_docs)))
    
    async def _store_image_documents(self, documents: List[Document], doc_metadata: Optional[Dict[str, Any]] = None):
        """Store image documents using multi-vector pattern"""
        if not documents:
            return
        
        # Generate unique chunk IDs for ChromaDB storage, but preserve original doc_id
        chunk_ids = [str(uuid.uuid4()) for _ in documents]
        
        summary_docs = []
        original_docs = []
        
        for i, doc in enumerate(documents):
            # Merge document-level metadata with chunk metadata
            chunk_metadata = doc.metadata.copy()
            if doc_metadata:
                chunk_metadata.update(doc_metadata)
            
            # Filter complex metadata to avoid ChromaDB errors
            clean_metadata = self._filter_metadata(chunk_metadata)
            # Keep original doc_id for document-level grouping, but add unique chunk_id for storage
            clean_metadata["chunk_id"] = chunk_ids[i]
            # doc_id stays the same for all chunks of the same document
            
            # Summary document for vector store (description)
            summary_doc = Document(
                page_content=doc.page_content,  # AI description
                metadata=clean_metadata
            )
            summary_docs.append(summary_doc)
            
            # Original document for docstore (includes base64 image)
            original_doc = Document(
                page_content=doc.page_content,  # Keep description as content
                metadata=clean_metadata  # Metadata includes image_base64
            )
            original_docs.append(original_doc)
        
        await asyncio.to_thread(
            self.image_vectorstore.add_documents, summary_docs
        )
        
        self.image_docstore.mset(list(zip(chunk_ids, original_docs)))
    
    async def search_multimodal(self, query: str, content_types: Optional[List[str]] = None, k: int = 5) -> Dict[str, List[Document]]:
        """Search across multiple content types"""
        if content_types is None:
            content_types = ["text", "tables", "images"]
        
        results = {}
        
        # Search text if requested
        if "text" in content_types:
            try:
                text_docs = await asyncio.to_thread(
                    self.text_retriever.invoke, query
                )
                results["text"] = text_docs[:k]
            except Exception as e:
                print(f"Error searching text: {e}")
                results["text"] = []
        
        # Search tables if requested
        if "tables" in content_types:
            try:
                table_docs = await asyncio.to_thread(
                    self.table_retriever.invoke, query
                )
                results["tables"] = table_docs[:k]
            except Exception as e:
                print(f"Error searching tables: {e}")
                results["tables"] = []
        
        # Search images if requested
        if "images" in content_types:
            try:
                image_docs = await asyncio.to_thread(
                    self.image_retriever.invoke, query
                )
                results["images"] = image_docs[:k]
            except Exception as e:
                print(f"Error searching images: {e}")
                results["images"] = []
        
        return results
    
    async def get_document_metadata(self, doc_id: str) -> Optional[Dict]:
        """Get document metadata"""
        # Search across all collections for this document
        for vectorstore in [self.text_vectorstore, self.table_vectorstore, self.image_vectorstore]:
            try:
                results = await asyncio.to_thread(
                    vectorstore.get,
                    where={"doc_id": doc_id}
                )
                if results["documents"]:
                    return results["metadatas"][0]
            except Exception as e:
                print(f"Error getting metadata from collection: {e}")
                continue
        return None
    
    async def list_all_documents(self) -> List[Dict[str, Any]]:
        """List all unique documents across all collections"""
        try:
            doc_metadata_map = {}  # Use doc_id as key to store best metadata for each document
            
            # Check all vector stores
            for vectorstore in [self.text_vectorstore, self.table_vectorstore, self.image_vectorstore]:
                try:
                    results = await asyncio.to_thread(vectorstore.get)
                    if results["metadatas"]:
                        for metadata in results["metadatas"]:
                            doc_id = metadata.get("doc_id")
                            if doc_id:
                                # If we haven't seen this doc_id, or current metadata is more complete, use it
                                if (doc_id not in doc_metadata_map or 
                                    self._is_better_metadata(metadata, doc_metadata_map[doc_id])):
                                    doc_metadata_map[doc_id] = metadata
                except Exception as e:
                    print(f"Error listing from collection: {e}")
                    continue
            
            return list(doc_metadata_map.values())
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def _is_better_metadata(self, new_meta: Dict, existing_meta: Dict) -> bool:
        """Check if new metadata is more complete than existing metadata"""
        # Prefer metadata that has proper filename (not "Unknown" or missing)
        new_filename = new_meta.get("filename", new_meta.get("file_name", "Unknown"))
        existing_filename = existing_meta.get("filename", existing_meta.get("file_name", "Unknown"))
        
        if new_filename != "Unknown" and existing_filename == "Unknown":
            return True
        elif new_filename == "Unknown" and existing_filename != "Unknown":
            return False
        
        # Prefer metadata with proper file_size (not 0)
        new_size = new_meta.get("file_size", 0)
        existing_size = existing_meta.get("file_size", 0)
        
        if new_size > 0 and existing_size == 0:
            return True
        elif new_size == 0 and existing_size > 0:
            return False
            
        # If both are similar quality, keep existing (first one wins)
        return False
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete all content for a document"""
        try:
            deleted_any = False
            
            # Delete from all vector stores
            for vectorstore in [self.text_vectorstore, self.table_vectorstore, self.image_vectorstore]:
                try:
                    # Get IDs to delete
                    results = await asyncio.to_thread(
                        vectorstore.get,
                        where={"doc_id": doc_id}
                    )
                    
                    if results["ids"]:
                        await asyncio.to_thread(
                            vectorstore.delete,
                            ids=results["ids"]
                        )
                        deleted_any = True
                except Exception as e:
                    print(f"Error deleting from vector store: {e}")
                    continue
            
            # Clean up docstores
            # Note: InMemoryStore doesn't have direct delete by criteria,
            # so we'd need to track the mapping IDs separately for complete cleanup
            
            return deleted_any
        except Exception as e:
            print(f"Error deleting document {doc_id}: {e}")
            return False
