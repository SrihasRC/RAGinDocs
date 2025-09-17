# RAGinDocs - Multimodal RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system that processes multimodal documents (PDF, DOCX) and enables intelligent querying across text, tables, and images.

## 🌟 Features

### **Multimodal Document Processing**
- **PDF & DOCX Support**: Extract and process text, tables, and images
- **Intelligent Chunking**: Break content into meaningful segments
- **AI Summarization**: Generate summaries using hybrid AI models
- **Content Separation**: Organize content by type for optimized retrieval

### **Advanced Vector Storage**
- **ChromaDB Integration**: Efficient vector storage with separate collections
- **Multi-Vector Retrieval**: LangChain's MultiVectorRetriever pattern
- **HuggingFace Embeddings**: Semantic similarity using sentence-transformers
- **Metadata Filtering**: Complex document metadata handling

### **Hybrid AI Architecture**
- **Groq for Text Processing**: Fast, efficient text summarization
- **Google Gemini for Complex Content**: Tables and images processing
- **Rate Limit Protection**: Intelligent fallback mechanisms
- **Cost Optimization**: Minimize API usage while maintaining quality

### **Production-Ready API**
- **FastAPI Backend**: High-performance async API
- **Auto-documentation**: Swagger UI at `/docs`
- **CORS Support**: Cross-origin requests enabled
- **Health Monitoring**: Comprehensive health checks

### **Testing Interface**
- **Streamlit Dashboard**: Full-featured testing interface
- **Document Upload**: Drag-and-drop file uploads
- **Query Testing**: Real-time query processing
- **System Monitoring**: Backend health and status

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   AI Services   │
│                 │    │                 │    │                 │
│ • Streamlit UI  │◄──►│ • FastAPI       │◄──►│ • Groq (Text)   │
│ • File Upload   │    │ • Route Mgmt    │    │ • Gemini (Multi)│
│ • Query Interface│   │ • Validation    │    │ • HF Embeddings │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Vector Storage │
                       │                 │
                       │ • ChromaDB      │
                       │ • Multi-Vector  │
                       │ • InMemoryStore │
                       └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Google AI API Key (Gemini)
- Groq API Key (optional but recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/SrihasRC/RAGinDocs.git
cd RAGinDocs
```

2. **Setup backend environment**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env and add your API keys:
# GOOGLE_API_KEY=your_gemini_api_key
# GROQ_API_KEY=your_groq_api_key
```

4. **Start the backend server**
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

5. **Start the Streamlit interface** (in new terminal)
```bash
cd ..
streamlit run streamlit_test.py --server.port 8501
```

6. **Access the application**
- API Documentation: http://localhost:8000/docs
- Streamlit Interface: http://localhost:8501

## 📊 Usage

### Document Upload
1. Open the Streamlit interface
2. Navigate to "📤 Upload Documents"
3. Upload PDF or DOCX files
4. Wait for processing completion

### Querying Documents
1. Navigate to "❓ Ask Questions"
2. Enter your question
3. Select content types (text, tables, images)
4. Get AI-powered answers with sources

### API Usage
```python
import requests

# Upload document
files = {'file': open('document.pdf', 'rb')}
response = requests.post('http://localhost:8000/documents/upload', files=files)

# Query documents
query = {
    "question": "What is the main topic?",
    "content_types": ["text"],
    "max_results": 5
}
response = requests.post('http://localhost:8000/query/ask', json=query)
```

## 🔧 Configuration

### Environment Variables
```bash
# Required
GOOGLE_API_KEY=your_gemini_api_key

# Optional (for text processing optimization)
GROQ_API_KEY=your_groq_api_key

# Optional (for debugging)
GRPC_VERBOSITY=ERROR
```

### Model Configuration
- **Text Summarization**: Groq Llama-3.1-8B-Instant (fast, efficient)
- **Table/Image Processing**: Google Gemini-2.0-Flash (multimodal)
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (local)
- **Vector Store**: ChromaDB with persistent storage

## 📁 Project Structure

```
RAGinDocs/
├── backend/
│   ├── api/routes/          # FastAPI route handlers
│   ├── config/              # Configuration management
│   ├── models/              # Pydantic data models
│   ├── services/            # Core business logic
│   │   ├── document_processor.py    # Document processing
│   │   ├── vector_store.py          # Vector storage
│   │   ├── rag_service.py           # RAG orchestration
│   │   └── shared_instances.py     # Singleton management
│   ├── utils/               # Utility functions
│   ├── main.py              # FastAPI application
│   └── requirements.txt     # Python dependencies
├── data/                    # Data storage
├── streamlit_test.py        # Testing interface
└── README.md               # This file
```

## 🔬 Technical Details

### Multi-Vector Retrieval Pattern
- **Summary Storage**: AI-generated summaries stored in vector database
- **Original Content**: Full content stored in document store
- **Retrieval Process**: Search summaries → retrieve original content
- **Benefits**: Improved search relevance while preserving full context

### Rate Limit Protection
- **Hybrid Models**: Use Groq for bulk text processing
- **Graceful Fallback**: Text truncation when API limits hit
- **Smart Health Checks**: No AI calls in health endpoints
- **Batch Processing**: Minimize API calls per document

### Performance Optimizations
- **Async Processing**: Non-blocking document processing
- **Shared Instances**: Singleton pattern for vector stores
- **Efficient Embeddings**: Local HuggingFace models
- **Persistent Storage**: ChromaDB with disk persistence

## 🐛 Troubleshooting

### Common Issues

**Rate Limit Errors**
- Check API key quotas
- Temporarily disable AI processing
- Use hybrid model setup

**Document Processing Fails**
- Verify file format (PDF/DOCX)
- Check file size limits
- Review error logs

**Vector Search Returns No Results**
- Ensure documents are uploaded
- Check vector store instances
- Verify embedding generation

### Debug Mode
```bash
# Enable detailed logging
export PYTHONPATH=$PYTHONPATH:/path/to/backend
python -c "from services.vector_store import LangChainVectorStore; vs = LangChainVectorStore(); print('Vector store healthy')"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For the multimodal RAG framework
- **ChromaDB**: For vector storage capabilities
- **HuggingFace**: For embedding models
- **Google AI**: For Gemini API
- **Groq**: For fast inference capabilities

---

**Built with ❤️ for intelligent document processing**
