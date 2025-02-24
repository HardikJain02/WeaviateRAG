# Weaviate RAG System

A powerful Retrieval Augmented Generation (RAG) system built with Weaviate, FastAPI, Docling, and OpenAI. This system provides both a web UI (using Gradio) and REST API endpoints for document processing, semantic search, and conversational AI capabilities.

## Features

### Document Processing
- Support for multiple file formats:
  - PDF (with OCR support)
  - DOCX
  - JSON
  - TXT
- Intelligent document chunking using hybrid strategies
- Automatic metadata extraction (page numbers, headings, etc.)
- Document replacement (if same filename exists already in weaviate)
- Batch processing support

### Search & Retrieval
- Hybrid search combining semantic and keyword matching
- Configurable result limits
- Document-specific search filtering based on documentId
- Relevance scoring

### Chat Interface
- Conversational AI powered by OpenAI GPT models
- Context-aware responses
- Chat history management
- Source attribution for responses
- Multi-document context support

### API Endpoints
- FAST API for all operations
- Document management endpoints
- Search and chat endpoints
- Batch processing support
- Detailed response formats

## Prerequisites

- Python 3.10 or higher
- Docker (for running Weaviate)
- CUDA-compatible GPU (optional, for improved performance)
- OpenAI API key

## Installation

1. Create and activate a conda environment:
```bash
conda create -n weaviaterag python=3.10
conda activate weaviaterag
```

2. Clone the repository:
```bash
git clone https://github.com/HardikJain02/WeaviateRAG.git
cd weaviaterag
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```
Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
WEAVIATE_URL=http://localhost:8080
```

## Setting up Weaviate

1. Start Weaviate using Docker:
```bash
docker run -d \
  -p 8080:8080 \
  -p 50051:50051 \
  -e QUERY_DEFAULTS_LIMIT=25 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
  -e DEFAULT_VECTORIZER_MODULE=none \
  -e ENABLE_MODULES="" \
  -e CLUSTER_HOSTNAME="node1" \
  --name weaviate \
  semitechnologies/weaviate:latest
```

2. Verify Weaviate is running:
```bash
curl http://localhost:8080/v1/meta
```

## Running the Application

1. Start the application:
```bash
python app.py
```

2. Access the interfaces:
- Web UI: http://localhost:7860/ui
- API Documentation: http://localhost:7860/docs

## Usage

### Web Interface

The web interface provides several tabs:

1. **Upload Files**
   - Upload documents (multi-select: Shift + Click)
   - View processing results

2. **Upload Folder**
   - Upload multiple documents at once
   - Batch processing status

3. **Manage Documents**
   - View all ingested documents
   - Delete individual or all documents or multiselect multiple files at once
   - Refresh document list

4. **Search Documents**
   - Semantic search across documents
   - Filter by document IDs
   - Configure result limits

5. **Chat**
   - Interactive chat interface
   - Context-aware responses
   - Source attribution

### API Endpoints [Test with Postman]

1. Document Management:
   - `GET /api/documents` - List all documents
   - `POST /api/ingest/files` - Upload files
   - `POST /api/ingest/folder` - Upload folder
   - `DELETE /api/del_docs` - Delete documents
   - `DELETE /api/documents/all` - Delete all documents

2. Search and Chat:
   - `POST /api/search` - Search documents
   - `POST /api/chat/start` - Start chat conversation
   - `POST /api/chat/continue` - Continue chat
   - `GET /api/chat/{conversation_id}/history` - Get chat history
   - `DELETE /api/chat/{conversation_id}` - Delete chat

### Testing with Postman
For easy API testing, a complete Postman collection is provided:

1. Start the application:
```bash
python app.py
```

2. Import the Postman collection:
   - Open Postman
   - Click "Import" button
   - Select `weaviate_rag_postman_collection.json` from the project root
   - The collection includes all API endpoints with example requests and responses


The collection is maintained and updated with all API changes.

## Dependencies

Core dependencies:
- `openai` - OpenAI API client
- `weaviate-client>=4.0.0` - Weaviate vector database client
- `gradio` - Web UI framework
- `docling~=2.7.0` - Document processing
- `python-dotenv` - Environment management
- `langchain` - LLM framework
- `tiktoken` - OpenAI tokenizer
- `torch` - PyTorch for ML operations
- `rich` - Enhanced terminal output
- `fastapi` - API framework
- `uvicorn` - ASGI server

## Project Structure

```
weaviaterag/
├── app.py                 # Main application file
├── document_processor.py  # Document processing logic
├── conversation_store.py  # Chat history management
├── requirements.txt      # Project dependencies
├── .env                 # Environment variables
└── documents/           # Document storage directory
```

## Performance Considerations

- GPU acceleration is automatically used if available for docling
- Batch processing for multiple documents
- Used docling hybrid chunking
- Connection fallback mechanisms for Weaviate
- Automatic handling of document versioning

## Troubleshooting

1. Weaviate Connection Issues:
   - Ensure both ports (8080 and 50051) are available
   - Check Docker container status
   - System will fall back to HTTP-only mode if gRPC fails

2. Document Processing:
   - Check file permissions
   - Verify supported file formats
   - Monitor chunks directory for processing results

3. Memory Issues:
   - Adjust chunk sizes in configuration
   - Monitor GPU memory usage
   - Consider CPU-only mode if needed

## Document Processing & Chunking Strategies

### Why Docling?
The project specifically uses Docling for document processing due to its powerful OCR capabilities:
- Built-in OCR support for scanned PDFs
- Handles complex document layouts
- Extracts text from images within PDFs
- Maintains document structure and formatting
- Processes tables and figures with OCR when needed
- Supports multiple languages
- GPU acceleration for faster OCR processing


### Chunking Overview
The system uses different chunking strategies based on file formats to ensure optimal text segmentation for embedding and retrieval:

1. **PDF & DOCX Documents (Using Docling HybridChunker)**
   - Implements a sophisticated two-pass chunking strategy:
     - First Pass: Document-based hierarchical chunking preserving document structure
     - Second Pass: Tokenization-aware refinements
   - Preserves document metadata:
     - Page numbers
     - Headers and captions
     - Element types (paragraphs, lists, tables)
   - Uses BAAI/bge-small-en-v1.5 tokenizer for chunk size optimization
   - Automatically merges undersized chunks with same headings
   - Maintains document hierarchy and structure

2. **JSON Files**
   - Uses CharacterTextSplitter with custom settings:
     - Chunk size: 1000 characters
     - Overlap: 200 characters
     - Preserves JSON structure through indentation
     - Adds chunk index for traceability

3. **Text Files**
   - Primary Strategy:
     - Uses newline separator
     - Chunk size: 1000 characters
     - Overlap: 200 characters
   - Fallback Strategy (if no chunks created):
     - Uses paragraph separator ("\n\n")
     - Same chunk size and overlap
   - Final Fallback:
     - Treats entire file as single chunk if no natural breaks found

### Chunk Debugging
The system maintains a `chunks` directory for debugging and verification:
```
chunks/
├── document1_chunks.txt
├── document2_chunks.txt
└── ...
```

Each chunks file contains:
- Original document name
- Chunk index
- Chunk length
- Metadata (if available):
  - Page numbers
  - Element types
  - Headers
  - Source type
- Actual chunk content

Example chunk file format:
```
Chunks from example.pdf
====================

Chunk 0:
----------------------------------------
Length: 856 characters

Metadata:
Pages: [1, 2]
Element Types: [PARAGRAPH, LIST]
Headings: ["Introduction", "Section 1"]

Content:
[actual chunk content]

====================
```

### Performance Optimizations
- GPU acceleration for document processing and OCR (if available)
- Efficient batch processing for multiple documents
- Smart chunk size adjustments based on token count
- Metadata preservation for context-aware retrieval
- Automatic handling of document updates and versioning
- OCR processing optimized for speed and accuracy

## Notable Points

- I can easily host it on any virtual machine- but i do not have availability.
- Automation: Monitors for new document uploads- this can be easily done but in my case when user hits process files i know i gotta upload right away- but when we are using s3 bucket/gcs bucket, we can setup a watcher for new folder/file or can directly trigger just like i did.
-Reranker can be added to the retrieved chunks to improve the quality of the final results.
