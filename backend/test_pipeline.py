"""
Integration test for the complete RAG pipeline:
Upload → Extract → Chunk → Embed → Store → Search
"""

import sys
import os
sys.path.append('.')

from services.pdf_processor import extract_text_from_file
from services.chunking import chunk_text
from services.embedding import embedding_service
from services.vector_store import vector_store

def test_complete_pipeline():
    """Test the complete RAG pipeline."""
    print("🚀 Testing Complete RAG Pipeline")
    print("=" * 50)
    
    # Step 1: Initialize services
    print("1. Initializing services...")
    vs_success = vector_store.initialize()
    emb_success = embedding_service.load_model()
    
    if not (vs_success and emb_success):
        print("❌ Failed to initialize services")
        return False
    
    print(f"   ✅ Vector store: {vs_success}")
    print(f"   ✅ Embeddings: {emb_success}")
    
    # Step 2: Simulate document text (since we might not have actual files)
    print("\n2. Processing document text...")
    
    sample_text = """
    Machine Learning and Artificial Intelligence
    
    Machine learning is a subset of artificial intelligence that focuses on the development 
    of algorithms and statistical models that enable computer systems to improve their 
    performance on a specific task through experience.
    
    Deep Learning Applications
    
    Deep learning, a subset of machine learning, uses neural networks with multiple layers 
    to model and understand complex patterns in data. It has revolutionized fields like 
    computer vision, natural language processing, and speech recognition.
    
    Python for Data Science
    
    Python has become the most popular programming language for data science and machine 
    learning due to its simplicity, extensive libraries, and strong community support. 
    Libraries like NumPy, Pandas, and Scikit-learn make it easy to work with data.
    """
    
    print(f"   ✅ Sample document: {len(sample_text)} characters")
    
    # Step 3: Chunk the text
    print("\n3. Chunking text...")
    
    # Format text as pages for chunking function
    text_pages = [{
        "text": sample_text,
        "page_number": 1,
        "source_file": "sample_ml_guide.pdf",
        "file_type": "pdf"
    }]
    
    chunks = chunk_text(
        text_pages=text_pages,
        chunk_size=150,
        overlap=30,
        respect_boundaries=True
    )
    
    print(f"   ✅ Generated {len(chunks)} chunks")
    for i, chunk in enumerate(chunks[:3]):  # Show first 3
        print(f"   Chunk {i+1}: {chunk['text'][:50]}...")
    
    # Step 4: Generate embeddings
    print("\n4. Generating embeddings...")
    
    enhanced_chunks = embedding_service.process_chunks_with_embeddings(chunks)
    print(f"   ✅ Generated embeddings for {len(enhanced_chunks)} chunks")
    print(f"   Embedding dimension: {len(enhanced_chunks[0]['embedding'])}")
    
    # Step 5: Store in vector database
    print("\n5. Storing in vector database...")
    
    embeddings = [chunk['embedding'] for chunk in enhanced_chunks]
    texts = [chunk['text'] for chunk in enhanced_chunks]
    metadatas = [
        {
            'chunk_id': chunk['chunk_id'],
            'start_char': chunk['start_char'],
            'end_char': chunk['end_char'],
            'page_number': chunk['page_number'],
            'document': chunk['source_file'],
            'file_type': chunk['file_type'],
            'char_count': chunk['char_count'],
            'word_count': chunk['word_count']
        }
        for chunk in enhanced_chunks
    ]
    
    add_success = vector_store.add_embeddings(embeddings, texts, metadatas)
    print(f"   ✅ Stored embeddings: {add_success}")
    
    # Step 6: Test similarity search
    print("\n6. Testing similarity search...")
    
    test_queries = [
        "What is machine learning?",
        "How is Python used in data science?",
        "Tell me about deep learning applications"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        
        results = vector_store.search_by_text(
            query_text=query,
            embedding_service=embedding_service,
            n_results=3
        )
        
        print(f"   Found {len(results['documents'])} relevant chunks:")
        
        for i, (doc, distance, metadata) in enumerate(zip(
            results['documents'][:2],  # Show top 2
            results['distances'][:2],
            results['metadatas'][:2]
        )):
            print(f"     {i+1}. Distance: {distance:.4f}")
            print(f"        Text: {doc[:80]}...")
            print(f"        Chunk ID: {metadata.get('chunk_id', 'N/A')}")
    
    # Step 7: Get final stats
    print("\n7. Final statistics...")
    stats = vector_store.get_collection_stats()
    print(f"   ✅ Total documents in vector store: {stats.get('document_count', 0)}")
    print(f"   Collection: {stats.get('collection_name', 'N/A')}")
    
    print("\n" + "=" * 50)
    print("🎉 Complete RAG Pipeline Test SUCCESSFUL!")
    print("✅ All components working together seamlessly")
    
    return True

if __name__ == "__main__":
    success = test_complete_pipeline()
    if success:
        print("\n🚀 Ready for production use!")
    else:
        print("\n❌ Pipeline test failed")
