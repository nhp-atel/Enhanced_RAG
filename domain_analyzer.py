"""
Domain-Specific Analysis with ReAct Protocol for RAG Systems

This module implements intelligent domain classification and embedding strategy
generation using the ReAct (Reasoning and Acting) protocol.
"""

import json
from typing import Dict, List, Any
from langchain.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


class DomainAnalyzer:
    """
    Analyzes academic papers to determine domain and suggest optimal embedding strategies
    using ReAct protocol for reasoning.
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Summarization prompt
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an expert research analyst. Analyze this academic paper content "
             "and provide a comprehensive summary focusing on domain identification."
            ),
            ("human", 
             "Analyze this paper and provide:\n"
             "1. Main research domain/field\n"
             "2. Key technical concepts and terminology\n"
             "3. Research methodology approach\n"
             "4. Primary contribution/findings\n"
             "5. Target application area\n\n"
             "Paper content:\n{content}"
            )
        ])
        
        # ReAct protocol prompt for domain classification
        self.react_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a domain classification expert using ReAct protocol.\n\n"
             "Follow this reasoning pattern:\n"
             "Thought: [Analyze the summary to understand the research domain]\n"
             "Action: [Classify domain and determine embedding strategy]\n"
             "Observation: [Based on classification, recommend specific embedding types]\n\n"
             "Domain-specific embedding strategies:\n"
             "- AI/ML: ['algorithms', 'datasets', 'evaluation_metrics', 'architectures', 'applications', 'technical_concepts']\n"
             "- NLP: ['language_models', 'tasks', 'evaluation', 'datasets', 'techniques', 'applications']\n"
             "- Computer Vision: ['models', 'datasets', 'metrics', 'applications', 'techniques', 'benchmarks']\n"
             "- Healthcare: ['conditions', 'treatments', 'outcomes', 'methodologies', 'populations', 'biomarkers']\n"
             "- Finance: ['models', 'markets', 'instruments', 'risk_factors', 'strategies', 'regulations']\n"
             "- General Research: ['concepts', 'methodology', 'findings', 'related_work', 'applications']\n\n"
             "Return ONLY valid JSON with these exact fields:\n"
             "{\n"
             "  \"thought\": \"reasoning about domain\",\n"
             "  \"action\": \"classification decision\",\n"
             "  \"observation\": \"embedding strategy rationale\",\n"
             "  \"domain\": \"specific domain name\",\n"
             "  \"confidence\": 0.95,\n"
             "  \"embedding_categories\": [\"category1\", \"category2\", ...]\n"
             "}"
            ),
            ("human", 
             "Paper Summary:\n{summary}\n\n"
             "Use ReAct protocol to classify domain and suggest embedding strategies."
            )
        ])
        
        # Domain-specific embedding generation prompts
        self.embedding_generation_prompts = {
            "algorithms": "Extract and list all algorithms, techniques, and computational methods mentioned",
            "datasets": "Identify all datasets, data sources, and data collection methods discussed", 
            "evaluation_metrics": "List all evaluation metrics, performance measures, and assessment criteria",
            "technical_concepts": "Extract key technical concepts, terminology, and theoretical frameworks",
            "applications": "Identify practical applications, use cases, and real-world implementations",
            "architectures": "List system architectures, model structures, and design patterns",
            "methodologies": "Extract research methodologies, experimental approaches, and procedures",
            "findings": "Summarize key findings, results, and conclusions",
            "related_work": "Identify related research, citations, and comparative studies"
        }

    def summarize_paper(self, content: str) -> str:
        """Generate comprehensive paper summary for domain analysis"""
        # Limit content size to avoid token limits
        truncated_content = content[:12000] if len(content) > 12000 else content
        
        messages = self.summarization_prompt.invoke({"content": truncated_content})
        response = self.llm.invoke(messages)
        return response.content

    def classify_domain_with_react(self, summary: str) -> Dict[str, Any]:
        """Use ReAct protocol to classify domain and suggest embedding strategies"""
        messages = self.react_prompt.invoke({"summary": summary})
        response = self.llm.invoke(messages)
        
        try:
            # Parse JSON response
            result = json.loads(response.content)
            
            # Validate required fields
            required_fields = ["domain", "confidence", "embedding_categories"]
            if not all(field in result for field in required_fields):
                raise ValueError("Missing required fields in response")
                
            return result
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing ReAct response: {e}")
            print(f"Raw response: {response.content}")
            
            # Fallback classification
            return {
                "thought": "Unable to parse detailed analysis",
                "action": "Default classification applied", 
                "observation": "Using general embedding strategy",
                "domain": "General Research",
                "confidence": 0.6,
                "embedding_categories": ["concepts", "methodology", "findings", "applications"]
            }

    def generate_domain_specific_embeddings(self, documents: List[Document], 
                                          embedding_categories: List[str]) -> Dict[str, Any]:
        """Generate domain-specific embeddings based on classification"""
        
        domain_embeddings = {}
        
        for category in embedding_categories:
            if category in self.embedding_generation_prompts:
                # Extract category-specific content
                extraction_prompt = ChatPromptTemplate.from_messages([
                    ("system", f"You are extracting {category} from research content. {self.embedding_generation_prompts[category]}"),
                    ("human", "Extract relevant {category} information from this content:\n\n{content}")
                ])
                
                category_content = []
                for doc in documents[:5]:  # Limit to first 5 docs for efficiency
                    messages = extraction_prompt.invoke({
                        "category": category,
                        "content": doc.page_content[:2000]
                    })
                    response = self.llm.invoke(messages)
                    if response.content.strip():
                        category_content.append(response.content)
                
                # Create embeddings for this category
                if category_content:
                    category_text = "\n".join(category_content)
                    embedding_vector = self.embeddings.embed_query(category_text)
                    domain_embeddings[category] = {
                        "content": category_text,
                        "embedding": embedding_vector,
                        "document_count": len(category_content)
                    }
        
        return domain_embeddings

    def analyze_paper_domain(self, documents: List[Document]) -> Dict[str, Any]:
        """Complete domain analysis pipeline"""
        
        # Step 1: Summarize paper content
        full_content = "\n\n".join([doc.page_content for doc in documents[:10]])
        summary = self.summarize_paper(full_content)
        
        # Step 2: Classify domain using ReAct protocol  
        classification = self.classify_domain_with_react(summary)
        
        # Step 3: Generate domain-specific embeddings
        domain_embeddings = self.generate_domain_specific_embeddings(
            documents, 
            classification["embedding_categories"]
        )
        
        return {
            "summary": summary,
            "classification": classification,
            "domain_embeddings": domain_embeddings,
            "total_categories": len(domain_embeddings)
        }


class DomainAwareVectorStore:
    """Enhanced vector store that uses domain-specific embeddings"""
    
    def __init__(self, base_vector_store, domain_embeddings: Dict[str, Any]):
        self.base_store = base_vector_store
        self.domain_embeddings = domain_embeddings
        
    def enhanced_similarity_search(self, query: str, k: int = 6, 
                                 use_domain_boost: bool = True) -> List[Document]:
        """Enhanced search using both base embeddings and domain-specific ones"""
        
        # Get base results
        base_results = self.base_store.similarity_search(query, k=k)
        
        if not use_domain_boost or not self.domain_embeddings:
            return base_results
            
        # Check if query matches any domain categories
        query_lower = query.lower()
        relevant_categories = []
        
        for category in self.domain_embeddings.keys():
            if category in query_lower or any(word in query_lower for word in category.split('_')):
                relevant_categories.append(category)
        
        # If domain match found, boost those results
        if relevant_categories:
            domain_context = []
            for category in relevant_categories:
                domain_info = self.domain_embeddings[category]
                domain_doc = Document(
                    page_content=f"[{category.upper()}]\n{domain_info['content']}",
                    metadata={"type": "domain_specific", "category": category}
                )
                domain_context.append(domain_doc)
            
            # Combine domain context with base results
            return domain_context + base_results[:k-len(domain_context)]
        
        return base_results