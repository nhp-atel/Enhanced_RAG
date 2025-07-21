# RAG System with FAISS and LangChain

## Overview
This project implements a Retrieval-Augmented Generation (RAG) system using FAISS for vector storage, LangChain for orchestration, and OpenAI's models for embeddings and generation. The system is designed to answer questions about research papers by retrieving relevant context and generating accurate responses.

## Key Features
- **PDF Processing**: Extracts text and metadata from research papers
- **FAISS Vector Store**: Efficient similarity search for document retrieval
- **Metadata Handling**: Special processing for paper title, authors, and publication date
- **LangSmith Integration**: Tracing enabled for debugging and monitoring
- **Environment Variables**: Secure API key management via .env file

## Technical Implementation

### Document Processing
1. Downloads PDF from arXiv
2. Extracts metadata (title, authors, publication date) from the first page
3. Creates a special metadata document to ensure accurate responses about paper details
4. Splits the document into chunks with 800 character size and 150 character overlap

### Vector Store
- Uses FAISS with OpenAI's text-embedding-3-small model
- Metadata document is inserted at multiple positions to ensure retrieval
- Enhanced retrieval logic prioritizes metadata for author/title/date queries

### Query Pipeline
1. **Retrieval**: Searches for relevant chunks based on the question
2. **Generation**: Uses GPT-4o-mini to generate answers from retrieved context
3. **Special handling**: Metadata questions trigger enhanced search patterns

## Known Issues and Solutions

### Issue: FAISS numpy array error
**Solution**: Use `FAISS.from_documents()` method instead of manual index creation

### Issue: Incorrect publication year from references
**Solution**: Created dedicated metadata document with correct information extracted from title page

### Issue: Missing author information
**Solution**: Enhanced retrieval strategy that searches for multiple metadata-related keywords

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your_key_here
LANGSMITH_API_KEY=your_key_here
```

3. Run the notebook:
```bash
jupyter notebook RAG.ipynb
```

## Testing
When testing, always verify:
- Title retrieval
- Author identification  
- Publication year (should be 2025 for the example paper, not dates from references)
- Institution information
- Main contributions

## Future Improvements
- Add support for multiple PDF formats
- Implement caching for processed documents
- Add more sophisticated chunk ranking algorithms
- Support for multi-modal content (figures, tables)