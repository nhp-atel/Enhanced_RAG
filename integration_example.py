"""
Integration example showing how to use domain_analyzer.py with existing RAG system
"""

from domain_analyzer import DomainAnalyzer, DomainAwareVectorStore
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS

def integrate_domain_analysis(documents, vector_store, llm):
    """
    Example integration of domain analysis into existing RAG pipeline
    """
    
    print("=== DOMAIN-SPECIFIC RAG ENHANCEMENT ===\n")
    
    # Step 1: Initialize domain analyzer
    print("1. Initializing domain analyzer...")
    analyzer = DomainAnalyzer(llm)
    
    # Step 2: Analyze paper domain
    print("2. Analyzing paper domain with ReAct protocol...")
    analysis_result = analyzer.analyze_paper_domain(documents)
    
    # Print analysis results
    print(f"\n--- DOMAIN ANALYSIS RESULTS ---")
    print(f"Domain: {analysis_result['classification']['domain']}")
    print(f"Confidence: {analysis_result['classification']['confidence']}")
    print(f"ReAct Reasoning:")
    print(f"  Thought: {analysis_result['classification'].get('thought', 'N/A')}")
    print(f"  Action: {analysis_result['classification'].get('action', 'N/A')}")
    print(f"  Observation: {analysis_result['classification'].get('observation', 'N/A')}")
    print(f"\nEmbedding Categories: {analysis_result['classification']['embedding_categories']}")
    print(f"Generated Domain Embeddings: {list(analysis_result['domain_embeddings'].keys())}")
    
    # Step 3: Create enhanced vector store
    print("\n3. Creating domain-aware vector store...")
    enhanced_store = DomainAwareVectorStore(
        base_vector_store=vector_store,
        domain_embeddings=analysis_result['domain_embeddings']
    )
    
    print("✅ Domain-specific RAG system ready!")
    
    return enhanced_store, analysis_result

def test_domain_queries(enhanced_store):
    """
    Test the enhanced system with domain-specific queries
    """
    
    print("\n=== TESTING DOMAIN-AWARE QUERIES ===\n")
    
    test_queries = [
        "What algorithms are discussed in this paper?",
        "What datasets were used for evaluation?", 
        "What are the key technical concepts?",
        "What evaluation metrics were employed?",
        "What are the practical applications?"
    ]
    
    for query in test_queries:
        print(f"Query: {query}")
        results = enhanced_store.enhanced_similarity_search(query, k=4)
        
        print("Retrieved content types:")
        for i, doc in enumerate(results):
            doc_type = doc.metadata.get('type', 'content')
            category = doc.metadata.get('category', 'N/A')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  [{i+1}] Type: {doc_type}, Category: {category}")
            print(f"      Preview: {preview}...")
        print("-" * 60)

# Usage example in notebook:
"""
# Add this to your RAG notebook after creating vector_store and before the RAG pipeline

# Import the domain analyzer
from domain_analyzer import DomainAnalyzer, DomainAwareVectorStore
from integration_example import integrate_domain_analysis, test_domain_queries

# Integrate domain analysis
enhanced_store, analysis = integrate_domain_analysis(all_splits, vector_store, llm)

# Test domain-aware queries  
test_domain_queries(enhanced_store)

# Update your retrieve function to use enhanced store:
def enhanced_retrieve(state: State):
    question_lower = state["question"].lower()
    
    # Check for metadata questions first
    metadata_keywords = ["author", "title", "year", "published", "wrote", "when", "date"]
    if any(keyword in question_lower for keyword in metadata_keywords):
        docs = vector_store.similarity_search("PAPER METADATA authors title publication date", k=15)
        metadata_docs = [doc for doc in docs if "PAPER METADATA" in doc.page_content]
        other_docs = [doc for doc in docs if "PAPER METADATA" not in doc.page_content]
        docs = metadata_docs + other_docs[:5]
    else:
        # Use domain-aware search
        docs = enhanced_store.enhanced_similarity_search(state["question"], k=6)
    
    return {"context": docs[:8]}

# Replace the retrieve function in your graph
graph_builder = StateGraph(State).add_sequence([enhanced_retrieve, generate])
"""