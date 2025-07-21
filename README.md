# RAG System for Research Paper Analysis

A Retrieval-Augmented Generation (RAG) system built with FAISS, LangChain, and OpenAI for analyzing research papers and answering questions about their content.

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key
- LangSmith API key (optional, for tracing)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/RAG.git
   cd RAG
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   LANGSMITH_API_KEY=your_langsmith_api_key_here  # Optional
   ```

### Running the Application

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open `RAG.ipynb`** in your browser

3. **Run all cells** in order (Cell → Run All)

## 📋 Features

- **PDF Processing**: Automatically downloads and processes research papers from arXiv
- **Metadata Extraction**: Accurately extracts paper title, authors, publication date
- **FAISS Vector Store**: Efficient similarity search for document retrieval
- **Question Answering**: Ask questions about the paper and get accurate, context-based answers
- **LangSmith Tracing**: Optional debugging and monitoring support

## 💡 Usage Examples

The notebook includes test questions such as:
- "What is the title of this paper?"
- "Who are the authors of this paper?"
- "In which year was this paper published?"
- "What are the main contributions?"

You can also enter custom questions in the interactive cell at the bottom of the notebook.

## 🛠️ Troubleshooting

### Common Issues

1. **FAISS Installation Error**
   ```bash
   # On Mac with M1/M2:
   pip install faiss-cpu --no-cache-dir
   ```

2. **API Key Errors**
   - Ensure your `.env` file is in the project root
   - Check that API keys are valid and have proper permissions

3. **Memory Issues**
   - The default paper is ~150 pages. For larger documents, consider increasing chunk size or reducing k value in similarity search

## 📁 Project Structure

```
RAG/
├── RAG.ipynb           # Main notebook
├── requirements.txt    # Python dependencies
├── .env               # API keys (create this file)
├── .gitignore         # Git ignore rules
├── README.md          # This file
└── CLAUDE.md          # Technical documentation
```

## 📚 Documentation

For technical details about the implementation, see [CLAUDE.md](CLAUDE.md).

## 🔒 Security

- Never commit your `.env` file
- API keys are automatically excluded via `.gitignore`
- Consider using environment-specific `.env` files for different deployments

## 📝 License

This project is for educational purposes. Please ensure you have the right to process any PDFs you use with this system.