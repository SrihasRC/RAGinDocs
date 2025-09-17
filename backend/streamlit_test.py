#!/usr/bin/env python3
"""
Streamlit Test Interface for RAGinDocs Backend
Simple UI to test document upload and query functionality
"""

import streamlit as st
import requests
import json
import time
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"

def main():
    st.set_page_config(
        page_title="RAGinDocs Test Interface",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAGinDocs Test Interface")
    st.markdown("Simple interface to test the multimodal RAG backend")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["🏠 Dashboard", "📤 Upload Documents", "❓ Ask Questions", "🔍 Search Documents", "⚙️ System Status"]
    )
    
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📤 Upload Documents":
        show_upload_page()
    elif page == "❓ Ask Questions":
        show_query_page()
    elif page == "🔍 Search Documents":
        show_search_page()
    elif page == "⚙️ System Status":
        show_system_status()

def show_dashboard():
    st.header("📊 Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    # Backend Status
    with col1:
        st.subheader("🔧 Backend Status")
        try:
            response = requests.get(f"{BACKEND_URL}/")
            if response.status_code == 200:
                st.success("✅ Backend Online")
            else:
                st.error("❌ Backend Error")
        except:
            st.error("❌ Backend Offline")
    
    # Document Count
    with col2:
        st.subheader("📚 Documents")
        try:
            response = requests.get(f"{BACKEND_URL}/documents/list")
            if response.status_code == 200:
                data = response.json()
                count = data.get("total", 0)
                st.metric("Total Documents", count)
            else:
                st.error("Error fetching documents")
        except:
            st.error("Cannot connect to backend")
    
    # Query Health
    with col3:
        st.subheader("🤖 AI Service")
        try:
            response = requests.get(f"{BACKEND_URL}/query/health")
            if response.status_code == 200:
                data = response.json()
                status = data.get("status", "unknown")
                if status == "healthy":
                    st.success("✅ AI Ready")
                else:
                    st.warning(f"⚠️ {status}")
            else:
                st.error("❌ AI Service Error")
        except:
            st.error("❌ AI Service Offline")

def show_upload_page():
    st.header("📤 Upload Documents")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF or DOCX file",
        type=['pdf', 'docx'],
        help="Upload a document to process and add to the knowledge base"
    )
    
    if uploaded_file is not None:
        st.write("📄 File Details:")
        st.write(f"- **Name**: {uploaded_file.name}")
        st.write(f"- **Size**: {uploaded_file.size / 1024:.2f} KB")
        st.write(f"- **Type**: {uploaded_file.type}")
        
        if st.button("🚀 Upload and Process", type="primary"):
            with st.spinner("Processing document... This may take a few moments."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BACKEND_URL}/documents/upload", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Document uploaded and processed successfully!")
                        
                        # Display processing results
                        if result.get("success"):
                            doc = result.get("document", {})
                            st.write("📊 Processing Results:")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"- **Document ID**: {doc.get('id', 'N/A')}")
                                st.write(f"- **Pages**: {doc.get('page_count', 'N/A')}")
                            with col2:
                                st.write(f"- **File Size**: {doc.get('file_size', 0)} bytes")
                                st.write(f"- **Status**: {doc.get('processing_status', 'N/A')}")
                        
                        # Show success message
                        st.balloons()
                    else:
                        st.error(f"❌ Upload failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

def show_query_page():
    st.header("❓ Ask Questions")
    
    # Query input
    query = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., What is the main topic of the uploaded documents?"
    )
    
    # Content type selection
    content_types = st.multiselect(
        "Search in:",
        ["text", "tables", "images"],
        default=["text", "tables", "images"],
        help="Select which types of content to search"
    )
    
    # Max results
    max_results = st.slider("Maximum results", 1, 10, 5)
    
    if st.button("🔍 Ask Question", type="primary") and query:
        with st.spinner("Searching and generating answer..."):
            try:
                payload = {
                    "question": query,
                    "content_types": content_types,
                    "max_results": max_results
                }
                
                response = requests.post(
                    f"{BACKEND_URL}/query/ask",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display answer
                    st.subheader("🤖 Answer")
                    st.write(result.get("answer", "No answer generated"))
                    
                    # Display sources
                    sources = result.get("sources", [])
                    if sources:
                        st.subheader("📚 Sources")
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"Source {i}: {source.get('type', 'unknown').title()}"):
                                st.write(source.get("content", "No content"))
                                if source.get("metadata"):
                                    st.json(source["metadata"])
                    
                    # Display metadata
                    metadata = result.get("metadata", {})
                    if metadata:
                        st.subheader("ℹ️ Query Info")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Total Sources**: {metadata.get('total_sources', 0)}")
                        with col2:
                            st.write(f"**Model Used**: {metadata.get('model_used', 'N/A')}")
                else:
                    st.error(f"❌ Query failed: {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def show_search_page():
    st.header("🔍 Search Documents")
    
    search_query = st.text_input(
        "Search for similar content:",
        placeholder="e.g., machine learning algorithms"
    )
    
    search_types = st.multiselect(
        "Content types:",
        ["text", "tables", "images"],
        default=["text"]
    )
    
    k = st.slider("Number of results", 1, 20, 5)
    
    if st.button("🔍 Search", type="primary") and search_query:
        with st.spinner("Searching..."):
            try:
                params = {
                    "query": search_query,
                    "content_types": ",".join(search_types),
                    "k": k
                }
                
                response = requests.get(f"{BACKEND_URL}/query/similar", params=params)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.subheader("🎯 Search Results")
                    st.write(f"Found {result.get('total_results', 0)} results")
                    
                    results = result.get("results", {})
                    for content_type, docs in results.items():
                        if docs:
                            st.write(f"**{content_type.title()} Results:**")
                            for i, doc in enumerate(docs, 1):
                                with st.expander(f"{content_type.title()} Result {i}"):
                                    st.write(doc.get("content", "No content"))
                                    if doc.get("metadata"):
                                        st.json(doc["metadata"])
                else:
                    st.error(f"❌ Search failed: {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def show_system_status():
    st.header("⚙️ System Status")
    
    # Backend connectivity
    st.subheader("🔧 Backend Connectivity")
    try:
        response = requests.get(f"{BACKEND_URL}/")
        if response.status_code == 200:
            st.success("✅ Backend is reachable")
            data = response.json()
            st.json(data)
        else:
            st.error(f"❌ Backend error: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Cannot connect to backend: {e}")
    
    # Query service health
    st.subheader("🤖 Query Service Health")
    try:
        response = requests.get(f"{BACKEND_URL}/query/health")
        if response.status_code == 200:
            data = response.json()
            st.success("✅ Query service is healthy")
            st.json(data)
        else:
            st.error(f"❌ Query service error: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Query service unreachable: {e}")
    
    # Document list
    st.subheader("📚 Document Storage")
    try:
        response = requests.get(f"{BACKEND_URL}/documents/list")
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ Document storage accessible ({data.get('total', 0)} documents)")
            
            if data.get("documents"):
                st.write("**Stored Documents:**")
                for doc in data["documents"]:
                    st.write(f"- {doc.get('file_name', 'Unknown')} ({doc.get('file_type', 'unknown')})")
            else:
                st.info("No documents uploaded yet")
        else:
            st.error(f"❌ Document storage error: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Document storage unreachable: {e}")

if __name__ == "__main__":
    main()
