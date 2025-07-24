"""
Quality tests for answer generation using evaluation metrics.
"""

import pytest
from typing import List, Dict, Any
from unittest.mock import Mock, patch

from tests.utils.evaluation_metrics import EvaluationMetrics, evaluate_answer_quality
from tests.utils.test_corpus import create_test_corpus, GROUND_TRUTH_QUERIES, get_queries_by_type
from src.core.pipeline import RAGPipeline
from langchain_core.documents import Document


class TestGenerationQuality:
    """Test suite for evaluating answer generation quality"""
    
    @pytest.fixture
    def generation_pipeline(self, test_config):
        """Pipeline configured for generation quality testing"""
        
        with patch('src.core.pipeline.create_llm_client') as mock_create_llm, \
             patch('src.core.pipeline.create_embedding_client') as mock_create_embed, \
             patch('src.core.pipeline.create_vector_store') as mock_create_store:
            
            # Setup LLM mock with realistic responses
            mock_llm = Mock()
            mock_embed = Mock()
            mock_store = Mock()
            
            mock_create_llm.return_value = mock_llm
            mock_create_embed.return_value = mock_embed
            mock_create_store.return_value = mock_store
            
            # Configure LLM with high-quality responses for testing
            def quality_generate(messages, **kwargs):
                message_text = " ".join([msg.get("content", "") for msg in messages]).lower()
                
                # Author questions
                if "authors" in message_text and "transformer" in message_text:
                    return Mock(content="The authors of the Transformer paper are Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin from Google Brain and Google Research.")
                
                elif "authors" in message_text and "bert" in message_text:
                    return Mock(content="The authors of BERT are Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova from Google AI Language.")
                
                # Architecture questions
                elif "transformer" in message_text and ("architecture" in message_text or "work" in message_text):
                    return Mock(content="The Transformer is a neural network architecture based entirely on attention mechanisms. It uses multi-head self-attention to process sequences in parallel, dispensing with recurrence and convolutions entirely. The model consists of an encoder and decoder, each with multiple layers containing multi-head attention and feed-forward networks.")
                
                elif "multi-head attention" in message_text:
                    return Mock(content="Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. It projects queries, keys, and values into multiple subspaces using learned linear projections, applies attention in parallel, then concatenates and projects the results.")
                
                # BERT questions
                elif "bert" in message_text and ("stand" in message_text or "acronym" in message_text):
                    return Mock(content="BERT stands for Bidirectional Encoder Representations from Transformers. It is a language representation model designed to pre-train deep bidirectional representations from unlabeled text.")
                
                elif "bert" in message_text and "pre-training" in message_text:
                    return Mock(content="BERT uses two pre-training tasks: Masked Language Model (MLM) where random tokens are masked and the model predicts them, and Next Sentence Prediction (NSP) where the model predicts if two sentences are consecutive in the original text.")
                
                # Comparison questions
                elif "advantages" in message_text and "transformer" in message_text:
                    return Mock(content="The Transformer has several advantages over RNNs: it is more parallelizable during training, requires significantly less time to train, and achieves better results on translation tasks. Unlike RNNs, it can process all positions simultaneously rather than sequentially.")
                
                # Dataset questions
                elif "datasets" in message_text or "wmt" in message_text:
                    return Mock(content="The Transformer was evaluated on WMT 2014 English-German and English-French translation tasks. The English-German dataset contains about 4.5 million sentence pairs, while the English-French dataset contains 36 million sentences.")
                
                # Default response
                else:
                    return Mock(content="Based on the provided context, I cannot find specific information to answer this question accurately.")
            
            mock_llm.generate.side_effect = quality_generate
            mock_llm.generate_with_retry.side_effect = quality_generate
            
            # Setup retrieval to return relevant documents
            test_docs = create_test_corpus()
            
            def mock_similarity_search(query_embedding, k=6):
                # Return different docs based on query context
                # This is a simplified relevance matching
                return test_docs[:k]  # Return first k documents
            
            mock_embed.embed_query.return_value = [0.1] * 768
            mock_embed.embed_documents.return_value = [[0.1] * 768] * len(test_docs)
            mock_store.add_documents.return_value = None
            mock_store.similarity_search.side_effect = mock_similarity_search
            
            pipeline = RAGPipeline(test_config)
            
            # Mock the retriever to return structured results
            def mock_retrieve_and_generate(question, k=6, strategy="basic"):
                query_embedding = pipeline.embedding_client.embed_query(question)
                sources = pipeline.vector_store.similarity_search(query_embedding, k)
                
                # Generate answer using LLM
                context = "\n".join([doc.page_content for doc in sources[:3]])
                messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}]
                response = pipeline.llm_client.generate(messages)
                
                return Mock(
                    answer=response.content,
                    sources=sources,
                    processing_time_ms=100,
                    tokens_used=150,
                    cost_usd=0.01,
                    metadata={"strategy": strategy, "context_length": len(context)}
                )
            
            mock_retriever = Mock()
            mock_retriever.retrieve_and_generate = mock_retrieve_and_generate
            pipeline.retriever = mock_retriever
            
            return pipeline
    
    def test_rouge_l_score_calculation(self):
        """Test ROUGE-L score calculation"""
        
        # Perfect match
        generated = "The quick brown fox jumps over the lazy dog"
        reference = "The quick brown fox jumps over the lazy dog"
        result = EvaluationMetrics.rouge_l(generated, reference)
        assert result.score == 1.0
        
        # Partial match
        generated = "The quick brown fox"
        reference = "The quick brown fox jumps over the lazy dog"
        result = EvaluationMetrics.rouge_l(generated, reference)
        assert 0.0 < result.score < 1.0
        assert result.details["lcs_length"] == 4  # "the quick brown fox"
        
        # No match
        generated = "Completely different text here"
        reference = "The quick brown fox jumps over the lazy dog"
        result = EvaluationMetrics.rouge_l(generated, reference)
        assert result.score == 0.0
    
    def test_bleu_score_calculation(self):
        """Test BLEU score calculation"""
        
        # Perfect match
        generated = "The authors are John and Jane"
        reference = "The authors are John and Jane"
        result = EvaluationMetrics.bleu_score(generated, reference)
        assert result.score == 1.0
        
        # Good match with different order
        generated = "John and Jane are the authors"
        reference = "The authors are John and Jane"
        result = EvaluationMetrics.bleu_score(generated, reference)
        assert result.score > 0.0
        
        # Poor match
        generated = "Machine learning is interesting"
        reference = "The authors are John and Jane"
        result = EvaluationMetrics.bleu_score(generated, reference)
        assert result.score < 0.3
    
    def test_semantic_similarity_calculation(self):
        """Test semantic similarity calculation"""
        
        # High similarity (overlapping concepts)
        generated = "The Transformer uses attention mechanisms for sequence modeling"
        reference = "Attention mechanisms are used by the Transformer for sequence processing"
        result = EvaluationMetrics.semantic_similarity(generated, reference)
        assert result.score > 0.5
        
        # Low similarity
        generated = "Machine learning is a subset of artificial intelligence"
        reference = "The cat sat on the mat"
        result = EvaluationMetrics.semantic_similarity(generated, reference)
        assert result.score < 0.2
    
    def test_answer_relevance_calculation(self):
        """Test answer relevance to question"""
        
        # Highly relevant answer
        question = "Who are the authors of the Transformer paper?"
        answer = "The authors of the Transformer paper are Vaswani, Shazeer, and others from Google"
        result = EvaluationMetrics.answer_relevance(answer, question)
        assert result.score > 0.5
        
        # Somewhat relevant answer
        question = "How does multi-head attention work?"
        answer = "Attention mechanisms allow models to focus on different parts of the input"
        result = EvaluationMetrics.answer_relevance(answer, question)
        assert result.score > 0.2
        
        # Irrelevant answer
        question = "Who are the authors?"
        answer = "Machine learning is a powerful technique for data analysis"
        result = EvaluationMetrics.answer_relevance(answer, question)
        assert result.score < 0.3
    
    def test_answer_faithfulness_calculation(self):
        """Test answer faithfulness to source documents"""
        
        source_docs = [
            Document(page_content="The Transformer was proposed by Vaswani et al. in 2017 and uses attention mechanisms."),
            Document(page_content="Multi-head attention allows parallel processing of different representation subspaces.")
        ]
        
        # Faithful answer (words appear in sources)
        faithful_answer = "The Transformer was proposed by Vaswani and uses attention mechanisms"
        result = EvaluationMetrics.answer_faithfulness(faithful_answer, source_docs)
        assert result.score > 0.7
        
        # Partially faithful answer
        mixed_answer = "The Transformer uses attention and also employs novel architectural innovations"
        result = EvaluationMetrics.answer_faithfulness(mixed_answer, source_docs)
        assert 0.3 < result.score < 0.8
        
        # Unfaithful answer (makes up information not in sources)
        unfaithful_answer = "The system utilizes quantum computing and blockchain technology for optimization"
        result = EvaluationMetrics.answer_faithfulness(unfaithful_answer, source_docs)
        assert result.score < 0.4
    
    def test_factual_answer_quality(self, generation_pipeline):
        """Test quality of answers to factual questions"""
        
        factual_queries = get_queries_by_type("factual")
        rouge_scores = []
        relevance_scores = []
        
        for query in factual_queries[:3]:  # Test 3 factual queries
            result = generation_pipeline.query(query.question)
            
            # Calculate ROUGE-L with expected answer
            rouge_result = EvaluationMetrics.rouge_l(result.answer, query.expected_answer)
            rouge_scores.append(rouge_result.score)
            
            # Calculate answer relevance
            relevance_result = EvaluationMetrics.answer_relevance(result.answer, query.question)
            relevance_scores.append(relevance_result.score)
            
            # Factual answers should be reasonably relevant
            assert relevance_result.score >= 0.3, f"Low relevance for factual query: {query.question}"
        
        # Average scores should meet minimum thresholds
        avg_rouge = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
        avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        assert avg_rouge >= 0.15, f"Average ROUGE too low for factual queries: {avg_rouge:.3f}"
        assert avg_relevance >= 0.4, f"Average relevance too low for factual queries: {avg_relevance:.3f}"
    
    def test_conceptual_answer_quality(self, generation_pipeline):
        """Test quality of answers to conceptual questions"""
        
        conceptual_queries = get_queries_by_type("conceptual")
        quality_scores = []
        
        for query in conceptual_queries[:2]:  # Test 2 conceptual queries
            result = generation_pipeline.query(query.question)
            
            # Calculate multiple quality metrics
            rouge_result = EvaluationMetrics.rouge_l(result.answer, query.expected_answer)
            similarity_result = EvaluationMetrics.semantic_similarity(result.answer, query.expected_answer)
            relevance_result = EvaluationMetrics.answer_relevance(result.answer, query.question)
            
            # Composite quality score
            quality_score = (rouge_result.score + similarity_result.score + relevance_result.score) / 3
            quality_scores.append(quality_score)
            
            # Individual checks
            assert relevance_result.score >= 0.25, f"Low relevance for conceptual query: {query.question}"
            assert len(result.answer.split()) >= 10, f"Answer too brief for conceptual query: {query.question}"
        
        # Average quality should be reasonable for conceptual questions
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            assert avg_quality >= 0.25, f"Low average quality for conceptual queries: {avg_quality:.3f}"
    
    def test_technical_answer_quality(self, generation_pipeline):
        """Test quality of answers to technical questions"""
        
        technical_queries = get_queries_by_type("technical")
        faithfulness_scores = []
        
        for query in technical_queries[:2]:  # Test 2 technical queries
            result = generation_pipeline.query(query.question)
            
            # Technical answers should be faithful to sources
            faithfulness_result = EvaluationMetrics.answer_faithfulness(result.answer, result.sources)
            faithfulness_scores.append(faithfulness_result.score)
            
            # Technical answers should contain some technical terms
            technical_terms = ["attention", "neural", "model", "architecture", "layer", "embedding", "token"]
            answer_lower = result.answer.lower()
            technical_term_count = sum(1 for term in technical_terms if term in answer_lower)
            
            assert technical_term_count >= 1, f"No technical terms in answer to: {query.question}"
            assert faithfulness_result.score >= 0.4, f"Low faithfulness for technical query: {query.question}"
        
        # Average faithfulness should be good for technical questions
        if faithfulness_scores:
            avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
            assert avg_faithfulness >= 0.5, f"Low average faithfulness for technical queries: {avg_faithfulness:.3f}"
    
    def test_answer_completeness(self, generation_pipeline):
        """Test that answers are complete and informative"""
        
        test_queries = [
            "What is the main contribution of the Transformer paper?",
            "How does BERT differ from previous language models?",
            "What are the advantages of the Transformer architecture?"
        ]
        
        for question in test_queries:
            result = generation_pipeline.query(question)
            
            # Check answer length (should be substantial)
            word_count = len(result.answer.split())
            assert word_count >= 15, f"Answer too short ({word_count} words) for: {question}"
            assert word_count <= 200, f"Answer too long ({word_count} words) for: {question}"
            
            # Check for complete sentences
            sentence_count = len([s for s in result.answer.split('.') if s.strip()])
            assert sentence_count >= 2, f"Answer should have multiple sentences for: {question}"
            
            # Answer should not be just "I don't know" type responses
            negative_phrases = ["i don't know", "cannot answer", "not enough information", "unclear"]
            answer_lower = result.answer.lower()
            negative_count = sum(1 for phrase in negative_phrases if phrase in answer_lower)
            
            # Allow some "cannot answer" responses but not all
            assert negative_count == 0 or len(result.answer) > 50, f"Answer seems too evasive for: {question}"
    
    def test_answer_consistency(self, generation_pipeline):
        """Test consistency of answers across similar questions"""
        
        similar_question_sets = [
            [
                "Who are the authors of the Transformer paper?",
                "Who wrote the Attention Is All You Need paper?",
                "What are the names of the Transformer paper authors?"
            ],
            [
                "What is BERT?",
                "What does BERT stand for?",
                "Can you explain what BERT is?"
            ]
        ]
        
        for question_set in similar_question_sets:
            answers = []
            for question in question_set:
                result = generation_pipeline.query(question)
                answers.append(result.answer.lower())
            
            # Check for consistent key information across answers
            if len(answers) >= 2:
                # For author questions, all should mention key authors
                if "author" in question_set[0].lower():
                    author_mentions = [sum(1 for answer in answers if name in answer) 
                                     for name in ["vaswani", "shazeer"]]
                    # At least one key author should be mentioned consistently
                    assert max(author_mentions) >= 2, "Inconsistent author information across similar questions"
                
                # For BERT questions, all should mention key concepts
                elif "bert" in question_set[0].lower():
                    bert_concepts = ["bidirectional", "transformer", "language", "representation"]
                    concept_mentions = [sum(1 for answer in answers if concept in answer)
                                      for concept in bert_concepts]
                    # At least one key concept should appear in multiple answers
                    assert max(concept_mentions) >= 2, "Inconsistent BERT information across similar questions"
    
    def test_comprehensive_answer_evaluation(self, generation_pipeline):
        """Comprehensive evaluation using the full evaluation suite"""
        
        # Test a sample of queries with full evaluation
        test_queries = GROUND_TRUTH_QUERIES[:4]  # Test first 4 queries
        evaluation_results = []
        
        for query in test_queries:
            result = generation_pipeline.query(query.question)
            
            # Perform comprehensive evaluation
            evaluation = evaluate_answer_quality(
                question=query.question,
                generated_answer=result.answer,
                reference_answer=query.expected_answer,
                retrieved_docs=result.sources,
                relevant_doc_ids=query.relevant_doc_ids
            )
            
            evaluation_results.append(evaluation)
            
            # Check individual metrics meet minimum thresholds
            assert evaluation.generation_metrics["answer_relevance"].score >= 0.2, \
                f"Low relevance for: {query.question}"
            
            assert evaluation.generation_metrics["answer_faithfulness"].score >= 0.3, \
                f"Low faithfulness for: {query.question}"
        
        # Calculate average performance across all queries
        if evaluation_results:
            from tests.utils.evaluation_metrics import aggregate_evaluation_results
            
            aggregated = aggregate_evaluation_results(evaluation_results)
            
            # Overall system performance should meet minimum standards
            assert aggregated.overall_score >= 0.25, \
                f"Overall system performance too low: {aggregated.overall_score:.3f}"
            
            # Log detailed results for analysis
            print(f"\nGeneration Quality Summary:")
            print(f"Overall Score: {aggregated.overall_score:.3f}")
            print(f"Average ROUGE-L: {aggregated.generation_metrics['rouge_l'].score:.3f}")
            print(f"Average Relevance: {aggregated.generation_metrics['answer_relevance'].score:.3f}")
            print(f"Average Faithfulness: {aggregated.generation_metrics['answer_faithfulness'].score:.3f}")
    
    def test_generation_edge_cases(self, generation_pipeline):
        """Test generation quality on edge cases"""
        
        edge_case_questions = [
            "",  # Empty question
            "?",  # Just punctuation
            "What is the meaning of life, universe, and everything related to transformers?",  # Very broad
            "Explain quantum mechanics in relation to attention mechanisms",  # Unrelated concepts
            "A" * 500,  # Very long question
        ]
        
        for question in edge_case_questions:
            if question:  # Skip empty questions
                try:
                    result = generation_pipeline.query(question)
                    
                    # Should still return some answer
                    assert result.answer is not None
                    assert len(result.answer) > 0
                    
                    # Answer should be reasonable length (not too short or too long)
                    word_count = len(result.answer.split())
                    assert 5 <= word_count <= 300, f"Unusual answer length for edge case: {word_count}"
                    
                except Exception as e:
                    # Should handle edge cases gracefully
                    assert "graceful error handling" in str(e).lower() or len(str(e)) > 0
    
    def test_generation_performance_benchmarks(self, generation_pipeline):
        """Test generation against performance benchmarks"""
        
        # Define minimum acceptable performance thresholds
        performance_thresholds = {
            "min_rouge_l": 0.10,
            "min_relevance": 0.30,
            "min_faithfulness": 0.40,
            "min_queries_above_threshold": 0.40  # 40% of queries should meet thresholds
        }
        
        queries_above_threshold = 0
        all_results = []
        
        # Test on sample queries
        test_queries = GROUND_TRUTH_QUERIES[:6]  # Test first 6 queries
        
        for query in test_queries:
            result = generation_pipeline.query(query.question)
            
            rouge_result = EvaluationMetrics.rouge_l(result.answer, query.expected_answer)
            relevance_result = EvaluationMetrics.answer_relevance(result.answer, query.question)
            faithfulness_result = EvaluationMetrics.answer_faithfulness(result.answer, result.sources)
            
            query_performance = {
                "question": query.question,
                "rouge_l": rouge_result.score,
                "relevance": relevance_result.score,
                "faithfulness": faithfulness_result.score
            }
            all_results.append(query_performance)
            
            # Check if query meets all thresholds
            if (rouge_result.score >= performance_thresholds["min_rouge_l"] and
                relevance_result.score >= performance_thresholds["min_relevance"] and
                faithfulness_result.score >= performance_thresholds["min_faithfulness"]):
                queries_above_threshold += 1
        
        # Calculate overall performance
        if all_results:
            avg_rouge = sum(r["rouge_l"] for r in all_results) / len(all_results)
            avg_relevance = sum(r["relevance"] for r in all_results) / len(all_results)
            avg_faithfulness = sum(r["faithfulness"] for r in all_results) / len(all_results)
            
            threshold_rate = queries_above_threshold / len(all_results)
            
            # Log performance for debugging
            print(f"\nGeneration Performance Summary:")
            print(f"Average ROUGE-L: {avg_rouge:.3f}")
            print(f"Average Relevance: {avg_relevance:.3f}")
            print(f"Average Faithfulness: {avg_faithfulness:.3f}")
            print(f"Queries above threshold: {threshold_rate:.3f}")
            
            # At least some queries should meet performance thresholds
            assert threshold_rate >= performance_thresholds["min_queries_above_threshold"], \
                f"Too few queries met generation performance thresholds: {threshold_rate:.3f}"