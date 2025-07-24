# 🚀 Enhanced RAG System - Production Ready

A production-ready Retrieval-Augmented Generation (RAG) system with modular architecture, built according to enterprise requirements.

## ✨ Key Features

### 🏗️ **Modular Architecture**
- **Dependency Injection**: All components are properly abstracted with interfaces
- **Separate Modules**: Ingest, Split, Embed, Retrieve, and Synthesize in dedicated classes
- **Pipeline Orchestrator**: Coordinates all components with proper error handling

### 🔧 **Abstracted Interfaces**
- **LLM & Embedding Clients**: Support multiple providers (OpenAI, Anthropic, HuggingFace)
- **External Prompt Templates**: Stored in YAML files for versioning and A/B testing
- **Pluggable Components**: Easy to swap implementations

### ⚙️ **Configuration Management**
- **YAML Configuration**: Comprehensive config files with validation
- **CLI Arguments**: Override any config value from command line
- **Environment Variables**: Secure API key management
- **Auto-computed Limits**: Dynamic chunk sizes based on model context windows

### 🛡️ **Robust Error Handling**
- **Retry Logic**: Exponential backoff with circuit breaker patterns
- **Structured Logging**: JSON logs with request IDs and performance metrics
- **Graceful Degradation**: System continues working when non-critical components fail

### 💾 **Caching & Persistence**
- **FAISS Index Persistence**: Save/load indices to/from disk or S3
- **Multi-backend Caching**: File, memory, and Redis support with LRU eviction
- **Content-based Caching**: Avoid redundant LLM calls using content hashing

### 🎯 **Dynamic Domains**
- **Runtime Domain Loading**: Add new domains via YAML configuration
- **Few-shot Examples**: Domain-specific prompts with examples
- **Auto-classification**: Automatically detect document domains

### 🧪 **Testing & Evaluation**
- **Ground Truth Suite**: Automated evaluation with recall@k, precision@k, MRR
- **Integration Tests**: Full pipeline testing with mocked components
- **Performance Benchmarks**: Track improvements and regressions

### 📊 **Performance & Cost Controls**
- **Rate Limiting**: Configurable API rate limits with budget controls
- **Cost Monitoring**: Track token usage and API costs per request
- **Lite vs Full Modes**: Balance between speed and comprehensiveness

### 🌐 **REST API Deployment**
- **FastAPI Server**: Production-ready API with automatic documentation
- **Health Checks**: Monitor system status and component health
- **Background Processing**: Async document processing with status tracking

## 🚦 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd Enhanced_RAG

# Install dependencies
pip install -r requirements_enhanced.txt

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### 2. Configuration

```bash
# Initialize default configuration
python main.py config init

# Validate configuration
python main.py config validate

# View current configuration
python main.py config show
```

### 3. Process Your First Document

```bash
# Process a research paper
python main.py process https://arxiv.org/pdf/2101.00001.pdf

# Process with custom settings
python main.py process paper.pdf --chunk-size 1000 --split-strategy adaptive
```

### 4. Query the System

```bash
# Interactive mode
python main.py query --interactive

# Single query
python main.py query "What is the main contribution of this paper?"

# JSON output for scripting
python main.py query "Who are the authors?" --output-format json
```

### 5. Start API Server

```bash
# Start development server
python main.py serve --port 8000 --reload

# Production deployment
uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📁 Project Structure

```
Enhanced_RAG/
├── main.py                 # CLI interface
├── config/
│   ├── config.yaml        # Main configuration
│   ├── domains.yaml       # Domain definitions
│   └── prompts.yaml       # Prompt templates
├── src/
│   ├── core/              # Core components
│   │   ├── ingest.py      # Document ingestion
│   │   ├── splitter.py    # Document splitting
│   │   └── pipeline.py    # Main orchestrator
│   ├── interfaces.py      # Abstract interfaces
│   ├── providers/         # Provider implementations
│   │   └── openai_provider.py
│   ├── stores/            # Storage implementations
│   │   ├── faiss_store.py # FAISS vector store
│   │   └── cache.py       # Caching system
│   ├── utils/             # Utilities
│   │   ├── config.py      # Configuration management
│   │   ├── logging.py     # Structured logging
│   │   ├── errors.py      # Error handling
│   │   ├── retry.py       # Retry logic
│   │   └── prompts.py     # Prompt management
│   └── api/               # REST API
│       └── server.py      # FastAPI server
├── tests/                 # Test suite
│   └── test_integration.py
└── data/                  # Data storage
    ├── indices/           # Vector indices
    ├── cache/             # Cache files
    └── logs/              # Log files
```

## 🔧 Configuration

### Main Configuration (`config/config.yaml`)

```yaml
# LLM Configuration
llm:
  provider: "openai"
  model: "gpt-4o-mini" 
  temperature: 0.1
  max_tokens: 2048
  rate_limit:
    requests_per_minute: 60

# Document Processing
document_processing:
  chunk_size: 800
  chunk_overlap: 150
  split_strategy: "adaptive"
  
# Vector Store  
vector_store:
  provider: "faiss"
  persistence:
    enabled: true
    directory: "./data/indices"

# Cost Controls
cost_control:
  daily_budget_usd: 50.0
  per_request_budget_usd: 1.0
```

### Domain Configuration (`config/domains.yaml`)

```yaml
domains:
  ai_ml:
    name: "Artificial Intelligence & Machine Learning"
    keywords: ["neural network", "deep learning", "machine learning"]
    embedding_categories:
      - "algorithms"
      - "datasets" 
      - "evaluation_metrics"
    few_shot_examples:
      - question: "What is a transformer architecture?"
        answer: "A transformer is a neural network architecture..."
```

### Prompt Templates (`config/prompts.yaml`)

```yaml
prompts:
  enhanced_rag:
    system: |
      You are an expert research assistant with access to multiple knowledge sources.
      Use the provided CONTEXT to answer questions accurately.
    human: |
      CONTEXT:
      {context}
      
      QUESTION: {question}
```

## 💻 CLI Usage

### Document Processing

```bash
# Basic processing
python main.py process document.pdf

# Advanced options
python main.py process https://arxiv.org/pdf/2101.00001.pdf \
  --split-strategy adaptive \
  --chunk-size 1000 \
  --save-index ./my_index

# Process with custom domain
python main.py process paper.pdf --domain ai_ml
```

### Querying

```bash
# Interactive mode
python main.py query --interactive

# Batch queries
python main.py query "What is machine learning?" --k 5 --strategy enhanced

# Load specific index
python main.py query "Explain the methodology" --load-index ./saved_index
```

### System Management

```bash
# Show system statistics
python main.py stats

# Health check
python main.py stats --load-index ./data/indices/auto_save

# Configuration management
python main.py config show
python main.py config validate
```

## 🌐 API Usage

### Start Server

```bash
python main.py serve --port 8000
```

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Process document
curl -X POST http://localhost:8000/process \
  -H "Content-Type: application/json" \
  -d '{"source": "https://arxiv.org/pdf/2101.00001.pdf"}'

# Query system  
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution?", "k": 5}'

# Get statistics
curl http://localhost:8000/stats
```

### API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test
pytest tests/test_integration.py::TestRAGIntegration::test_complete_workflow -v
```

### Ground Truth Evaluation

```bash
# Run evaluation suite (when implemented)
python -m src.evaluation.evaluate --ground-truth tests/data/ground_truth.json

# Custom evaluation
python -m src.evaluation.evaluate \
  --papers tests/data/sample_papers/ \
  --metrics recall@k,precision@k,mrr
```

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_enhanced.txt .
RUN pip install -r requirements_enhanced.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_api_key_here

# Optional
RAG_ENVIRONMENT=production
RAG_LOG_LEVEL=INFO
RAG_DAILY_BUDGET=100.0
RAG_API_PORT=8000
```

### Production Configuration

```yaml
# config/production.yaml
system:
  environment: "production"

logging:
  level: "INFO"
  output: "file"
  file_path: "/var/log/rag_system.log"

monitoring:
  enabled: true
  metrics_port: 8080
  
api:
  authentication:
    enabled: true
    type: "api_key"
```

## 🔍 Monitoring & Observability

### Structured Logging

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "INFO",
  "message": "Document processing completed",
  "request_id": "req_123",
  "document_id": "doc_456",
  "processing_time_ms": 2340,
  "tokens_used": 1250,
  "cost_usd": 0.045
}
```

### Metrics Collection

```python
# Custom metrics
from src.utils.logging import get_logger, PerformanceLogger

logger = get_logger("my_component")
perf_logger = PerformanceLogger(logger)

# Log performance metrics
perf_logger.log_operation("document_processing", duration_ms=1500, 
                         doc_count=1, chunk_count=15)

perf_logger.log_api_call("openai", "gpt-4o-mini", tokens=1200, 
                        cost=0.024, duration_ms=800)
```

## 🛠️ Extending the System

### Add New LLM Provider

```python
# src/providers/anthropic_provider.py
from ..interfaces import LLMClient, LLMResponse

class AnthropicLLMClient(LLMClient):
    def generate(self, messages, **kwargs) -> LLMResponse:
        # Implementation
        pass
```

### Custom Vector Store

```python
# src/stores/pinecone_store.py  
from ..interfaces import VectorStore

class PineconeVectorStore(VectorStore):
    def similarity_search(self, query, k=6, **kwargs):
        # Implementation
        pass
```

### Domain-Specific Processing

```yaml
# config/domains.yaml - Add new domain
domains:
  medical:
    name: "Medical Research"
    keywords: ["clinical", "patient", "diagnosis", "treatment"]
    embedding_categories:
      - "conditions"
      - "treatments"
      - "outcomes"
    few_shot_examples:
      - question: "What is the treatment protocol?"
        answer: "The treatment protocol involves..."
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Make changes following the architecture patterns
4. Add tests for new functionality
5. Update documentation
6. Submit pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements_enhanced.txt
pip install black isort flake8 mypy

# Set up pre-commit hooks
pre-commit install

# Run linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of LangChain and FAISS
- Inspired by production RAG systems at scale
- Follows enterprise software architecture patterns

---

## 📞 Support

For issues and questions:
- 🐛 **Bug Reports**: [GitHub Issues](issues)
- 💬 **Discussions**: [GitHub Discussions](discussions)  
- 📚 **Documentation**: [Wiki](wiki)

**Built for Production. Ready for Scale. 🚀**