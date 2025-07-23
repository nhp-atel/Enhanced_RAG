"""
Advanced prompting strategies for improved RAG performance.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

from langchain.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Classification of query types for targeted prompting."""
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    SUMMARIZATION = "summarization"


@dataclass
class PromptExample:
    """Container for few-shot prompt examples."""
    context: str
    question: str
    answer: str
    explanation: Optional[str] = None


class QueryClassifier:
    """Classify queries to select appropriate prompting strategy."""
    
    def __init__(self, llm):
        self.llm = llm
        self.classification_prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a query classifier for a research paper Q&A system.
            
            Classify the query into one of these categories:
            - FACTUAL: Questions about specific facts (who, what, when, where)
            - CONCEPTUAL: Questions about definitions, explanations, concepts
            - PROCEDURAL: Questions about how something works or is done
            - COMPARATIVE: Questions comparing different approaches/methods
            - ANALYTICAL: Questions requiring analysis or interpretation
            - SUMMARIZATION: Questions asking for summaries or overviews
            
            Return only the category name.
            """),
            ("human", "Query: {query}")
        ])
    
    def classify(self, query: str) -> QueryType:
        """Classify query type."""
        try:
            messages = self.classification_prompt.invoke({"query": query})
            response = self.llm.invoke(messages)
            
            category = response.content.strip().upper()
            if category in [qt.name for qt in QueryType]:
                return QueryType[category]
            else:
                logger.warning(f"Unknown category: {category}, defaulting to FACTUAL")
                return QueryType.FACTUAL
                
        except Exception as e:
            logger.error(f"Query classification failed: {e}")
            return QueryType.FACTUAL


class FewShotRAGPrompts:
    """Few-shot prompting templates for different query types."""
    
    # Few-shot examples for different query types
    FACTUAL_EXAMPLES = [
        PromptExample(
            context="Paper: 'Attention Is All You Need' by Vaswani et al., published in 2017 at NIPS.",
            question="Who are the authors of the Transformer paper?",
            answer="The authors of 'Attention Is All You Need' are Vaswani et al.",
            explanation="Direct factual information extracted from metadata."
        ),
        PromptExample(
            context="The paper introduces the Transformer architecture, which achieved BLEU scores of 28.4 on WMT 2014 English-to-German translation.",
            question="What BLEU score did the Transformer achieve?",
            answer="The Transformer achieved a BLEU score of 28.4 on WMT 2014 English-to-German translation.",
            explanation="Specific numerical fact from the results."
        )
    ]
    
    CONCEPTUAL_EXAMPLES = [
        PromptExample(
            context="Self-attention allows each position in a sequence to attend to all positions in the previous layer, enabling the model to capture long-range dependencies.",
            question="What is self-attention?",
            answer="Self-attention is a mechanism that allows each position in a sequence to attend to all positions in the previous layer, enabling the model to capture long-range dependencies between elements regardless of their distance.",
            explanation="Conceptual explanation with context and implications."
        )
    ]
    
    PROCEDURAL_EXAMPLES = [
        PromptExample(
            context="The multi-head attention mechanism splits the input into multiple heads, applies scaled dot-product attention to each head, then concatenates the results.",
            question="How does multi-head attention work?",
            answer="Multi-head attention works by: 1) Splitting the input into multiple heads, 2) Applying scaled dot-product attention to each head independently, 3) Concatenating the results from all heads, and 4) Applying a final linear transformation.",
            explanation="Step-by-step procedural explanation."
        )
    ]
    
    @classmethod
    def get_examples(cls, query_type: QueryType) -> List[PromptExample]:
        """Get appropriate examples for query type."""
        examples_map = {
            QueryType.FACTUAL: cls.FACTUAL_EXAMPLES,
            QueryType.CONCEPTUAL: cls.CONCEPTUAL_EXAMPLES,
            QueryType.PROCEDURAL: cls.PROCEDURAL_EXAMPLES,
            # Add more as needed
        }
        return examples_map.get(query_type, cls.FACTUAL_EXAMPLES)


class AdaptiveRAGPrompts:
    """Adaptive prompting system that selects strategies based on query type."""
    
    def __init__(self, llm):
        self.llm = llm
        self.classifier = QueryClassifier(llm)
        self.few_shot_examples = FewShotRAGPrompts()
    
    def create_prompt(
        self, 
        query: str, 
        context: List[Document],
        query_type: Optional[QueryType] = None
    ) -> ChatPromptTemplate:
        """
        Create adaptive prompt based on query type.
        
        Args:
            query: User question
            context: Retrieved documents
            query_type: Optional query type (will classify if not provided)
            
        Returns:
            Optimized prompt template
        """
        if query_type is None:
            query_type = self.classifier.classify(query)
        
        logger.info(f"Creating {query_type.value} prompt for query")
        
        # Get appropriate examples
        examples = self.few_shot_examples.get_examples(query_type)
        
        # Create few-shot template
        example_template = ChatPromptTemplate.from_messages([
            ("human", "Context: {context}\nQuestion: {question}"),
            ("assistant", "{answer}")
        ])
        
        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_template,
            examples=[
                {
                    "context": ex.context,
                    "question": ex.question,
                    "answer": ex.answer
                }
                for ex in examples
            ]
        )
        
        # Create main prompt based on query type
        system_message = self._get_system_message(query_type)
        
        final_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message),
            few_shot_prompt,
            ("human", "Context: {context}\nQuestion: {question}")
        ])
        
        return final_prompt
    
    def _get_system_message(self, query_type: QueryType) -> str:
        """Get system message tailored to query type."""
        base_message = """You are an expert research assistant analyzing academic papers. 
Use the provided context to answer questions accurately and comprehensively."""
        
        type_specific = {
            QueryType.FACTUAL: """
Focus on extracting specific facts, numbers, names, and dates. 
Be precise and cite the exact information from the context.
If the information isn't in the context, clearly state this.""",
            
            QueryType.CONCEPTUAL: """
Provide clear, comprehensive explanations of concepts and ideas.
Define technical terms and explain their significance.
Connect concepts to broader theoretical frameworks when relevant.""",
            
            QueryType.PROCEDURAL: """
Break down processes into clear, logical steps.
Explain not just what happens, but how and why it works.
Use numbered steps or structured explanations when appropriate.""",
            
            QueryType.COMPARATIVE: """
Highlight similarities and differences clearly.
Organize comparisons in a structured way (e.g., advantages/disadvantages).
Provide balanced analysis of different approaches.""",
            
            QueryType.ANALYTICAL: """
Provide thoughtful analysis and interpretation.
Connect evidence to conclusions.
Consider multiple perspectives and implications.""",
            
            QueryType.SUMMARIZATION: """
Create comprehensive yet concise summaries.
Capture the most important points and key insights.
Maintain logical flow and organization."""
        }
        
        return base_message + "\n\n" + type_specific.get(query_type, "")


class ChainOfThoughtPrompting:
    """Chain-of-thought prompting for complex reasoning."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def create_cot_prompt(self, query: str, context: List[Document]) -> ChatPromptTemplate:
        """Create chain-of-thought prompt for complex reasoning."""
        return ChatPromptTemplate.from_messages([
            ("system", """
You are an expert research assistant. For complex questions, work through your reasoning step by step.

Follow this process:
1. UNDERSTAND: What is the question asking?
2. ANALYZE: What relevant information is in the context?
3. REASON: How does this information connect to answer the question?
4. CONCLUDE: What is the final answer based on your reasoning?

Use this format:
**Understanding:** [What the question is asking]
**Analysis:** [Key information from context]
**Reasoning:** [How the information connects]
**Conclusion:** [Final answer]
            """),
            ("human", """
Context: {context}

Question: {question}

Please work through this step by step.
            """)
        ])


class ReflectivePrompting:
    """Self-reflection prompting to improve answer quality."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def create_reflective_prompt(
        self, 
        initial_answer: str, 
        context: str, 
        question: str
    ) -> ChatPromptTemplate:
        """Create prompt for answer reflection and improvement."""
        return ChatPromptTemplate.from_messages([
            ("system", """
You are reviewing an answer to improve its quality. 

Evaluate the answer on these criteria:
1. Accuracy: Is the information correct based on the context?
2. Completeness: Does it fully address the question?
3. Clarity: Is it well-organized and easy to understand?
4. Relevance: Does it focus on what's asked?

If the answer needs improvement, provide a better version.
If it's already good, confirm it's accurate.
            """),
            ("human", """
Original Context: {context}

Question: {question}

Initial Answer: {initial_answer}

Please evaluate and improve this answer if needed.
            """)
        ])


# Usage example
def main():
    """Example usage of adaptive prompting."""
    from langchain_openai import ChatOpenAI
    from langchain_core.documents import Document
    
    # Initialize
    llm = ChatOpenAI()
    adaptive_prompts = AdaptiveRAGPrompts(llm)
    
    # Sample context and query
    context = [
        Document(page_content="The Transformer uses self-attention mechanisms...")
    ]
    query = "What is self-attention?"
    
    # Create adaptive prompt
    prompt = adaptive_prompts.create_prompt(query, context)
    
    # Use prompt
    messages = prompt.invoke({
        "context": "\n".join([doc.page_content for doc in context]),
        "question": query
    })
    
    response = llm.invoke(messages)
    print(f"Response: {response.content}")


if __name__ == "__main__":
    main()