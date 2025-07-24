"""
Evaluation metrics for RAG system quality assessment.
"""

import re
import math
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import Counter
import numpy as np


@dataclass
class EvaluationResult:
    """Results from evaluation metrics"""
    metric_name: str
    score: float
    max_score: float
    details: Dict[str, Any]
    
    @property
    def normalized_score(self) -> float:
        """Get score normalized to 0-1 range"""
        return self.score / self.max_score if self.max_score > 0 else 0.0


@dataclass
class RAGEvaluationSuite:
    """Complete evaluation results for RAG system"""
    retrieval_metrics: Dict[str, EvaluationResult]
    generation_metrics: Dict[str, EvaluationResult]
    overall_score: float
    num_queries: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "overall_score": self.overall_score,
            "num_queries": self.num_queries,
            "retrieval_metrics": {
                name: {
                    "score": result.score,
                    "normalized_score": result.normalized_score,
                    "details": result.details
                }
                for name, result in self.retrieval_metrics.items()
            },
            "generation_metrics": {
                name: {
                    "score": result.score,
                    "normalized_score": result.normalized_score,
                    "details": result.details
                }
                for name, result in self.generation_metrics.items()
            }
        }


class EvaluationMetrics:
    """Collection of evaluation metrics for RAG systems"""
    
    @staticmethod
    def precision_at_k(retrieved_docs: List[Any], relevant_doc_ids: List[str], k: int = None) -> EvaluationResult:
        """
        Calculate Precision@K for document retrieval
        
        Args:
            retrieved_docs: List of retrieved documents with metadata
            relevant_doc_ids: List of IDs for relevant documents
            k: Number of top documents to consider (default: all)
        """
        if not retrieved_docs:
            return EvaluationResult("precision_at_k", 0.0, 1.0, {"k": k, "retrieved": 0, "relevant": 0})
        
        k = k or len(retrieved_docs)
        top_k_docs = retrieved_docs[:k]
        
        # Count relevant documents in top-k
        relevant_retrieved = 0
        for doc in top_k_docs:
            doc_id = doc.metadata.get("document_id") or doc.metadata.get("source", "")
            if doc_id in relevant_doc_ids:
                relevant_retrieved += 1
        
        precision = relevant_retrieved / k if k > 0 else 0.0
        
        return EvaluationResult(
            metric_name="precision_at_k",
            score=precision,
            max_score=1.0,
            details={
                "k": k,
                "retrieved": len(top_k_docs),
                "relevant_retrieved": relevant_retrieved,
                "total_relevant": len(relevant_doc_ids)
            }
        )
    
    @staticmethod
    def recall_at_k(retrieved_docs: List[Any], relevant_doc_ids: List[str], k: int = None) -> EvaluationResult:
        """
        Calculate Recall@K for document retrieval
        """
        if not relevant_doc_ids:
            return EvaluationResult("recall_at_k", 1.0, 1.0, {"k": k, "no_relevant_docs": True})
        
        if not retrieved_docs:
            return EvaluationResult("recall_at_k", 0.0, 1.0, {"k": k, "retrieved": 0, "relevant": len(relevant_doc_ids)})
        
        k = k or len(retrieved_docs)
        top_k_docs = retrieved_docs[:k]
        
        # Count relevant documents found in top-k
        found_relevant = set()
        for doc in top_k_docs:
            doc_id = doc.metadata.get("document_id") or doc.metadata.get("source", "")
            if doc_id in relevant_doc_ids:
                found_relevant.add(doc_id)
        
        recall = len(found_relevant) / len(relevant_doc_ids)
        
        return EvaluationResult(
            metric_name="recall_at_k",
            score=recall,
            max_score=1.0,
            details={
                "k": k,
                "retrieved": len(top_k_docs),
                "relevant_found": len(found_relevant),
                "total_relevant": len(relevant_doc_ids)
            }
        )
    
    @staticmethod
    def f1_at_k(retrieved_docs: List[Any], relevant_doc_ids: List[str], k: int = None) -> EvaluationResult:
        """
        Calculate F1@K combining precision and recall
        """
        precision_result = EvaluationMetrics.precision_at_k(retrieved_docs, relevant_doc_ids, k)
        recall_result = EvaluationMetrics.recall_at_k(retrieved_docs, relevant_doc_ids, k)
        
        precision = precision_result.score
        recall = recall_result.score
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return EvaluationResult(
            metric_name="f1_at_k",
            score=f1,
            max_score=1.0,
            details={
                "k": k,
                "precision": precision,
                "recall": recall,
                "precision_details": precision_result.details,
                "recall_details": recall_result.details
            }
        )
    
    @staticmethod
    def mean_reciprocal_rank(retrieved_docs: List[Any], relevant_doc_ids: List[str]) -> EvaluationResult:
        """
        Calculate Mean Reciprocal Rank (MRR) for document retrieval
        """
        if not retrieved_docs or not relevant_doc_ids:
            return EvaluationResult("mrr", 0.0, 1.0, {"first_relevant_rank": None})
        
        # Find rank of first relevant document
        for rank, doc in enumerate(retrieved_docs, 1):
            doc_id = doc.metadata.get("document_id") or doc.metadata.get("source", "")
            if doc_id in relevant_doc_ids:
                mrr = 1.0 / rank
                return EvaluationResult(
                    metric_name="mrr",
                    score=mrr,
                    max_score=1.0,
                    details={"first_relevant_rank": rank}
                )
        
        # No relevant documents found
        return EvaluationResult("mrr", 0.0, 1.0, {"first_relevant_rank": None})
    
    @staticmethod
    def rouge_l(generated_text: str, reference_text: str) -> EvaluationResult:
        """
        Calculate ROUGE-L score for generated text quality
        """
        def _lcs_length(x: List[str], y: List[str]) -> int:
            """Calculate length of longest common subsequence"""
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        # Tokenize texts
        gen_tokens = generated_text.lower().split()
        ref_tokens = reference_text.lower().split()
        
        if not gen_tokens or not ref_tokens:
            return EvaluationResult("rouge_l", 0.0, 1.0, {"lcs_length": 0})
        
        # Calculate LCS
        lcs_len = _lcs_length(gen_tokens, ref_tokens)
        
        # Calculate precision and recall
        precision = lcs_len / len(gen_tokens) if gen_tokens else 0
        recall = lcs_len / len(ref_tokens) if ref_tokens else 0
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return EvaluationResult(
            metric_name="rouge_l",
            score=f1,
            max_score=1.0,
            details={
                "lcs_length": lcs_len,
                "precision": precision,
                "recall": recall,
                "gen_tokens": len(gen_tokens),
                "ref_tokens": len(ref_tokens)
            }
        )
    
    @staticmethod
    def bleu_score(generated_text: str, reference_text: str, n: int = 4) -> EvaluationResult:
        """
        Calculate BLEU score for generated text quality
        """
        def _get_ngrams(tokens: List[str], n: int) -> Counter:
            """Get n-grams from tokens"""
            ngrams = []
            for i in range(len(tokens) - n + 1):
                ngrams.append(tuple(tokens[i:i+n]))
            return Counter(ngrams)
        
        # Tokenize
        gen_tokens = generated_text.lower().split()
        ref_tokens = reference_text.lower().split()
        
        if not gen_tokens or not ref_tokens:
            return EvaluationResult("bleu", 0.0, 1.0, {"brevity_penalty": 0})
        
        # Calculate brevity penalty
        gen_len = len(gen_tokens)
        ref_len = len(ref_tokens)
        
        if gen_len > ref_len:
            brevity_penalty = 1.0
        else:
            brevity_penalty = math.exp(1 - ref_len / gen_len) if gen_len > 0 else 0
        
        # Calculate n-gram precisions
        precisions = []
        for i in range(1, n + 1):
            gen_ngrams = _get_ngrams(gen_tokens, i)
            ref_ngrams = _get_ngrams(ref_tokens, i)
            
            if not gen_ngrams:
                precisions.append(0.0)
                continue
                
            # Count matching n-grams
            matches = 0
            for ngram, count in gen_ngrams.items():
                matches += min(count, ref_ngrams.get(ngram, 0))
            
            precision = matches / sum(gen_ngrams.values())
            precisions.append(precision)
        
        # Calculate geometric mean of precisions
        if any(p == 0 for p in precisions):
            bleu = 0.0
        else:
            log_precisions = [math.log(p) for p in precisions]
            bleu = brevity_penalty * math.exp(sum(log_precisions) / len(log_precisions))
        
        return EvaluationResult(
            metric_name="bleu",
            score=bleu,
            max_score=1.0,
            details={
                "brevity_penalty": brevity_penalty,
                "precisions": precisions,
                "gen_length": gen_len,
                "ref_length": ref_len
            }
        )
    
    @staticmethod
    def semantic_similarity(generated_text: str, reference_text: str) -> EvaluationResult:
        """
        Calculate semantic similarity using simple token overlap
        (In production, you'd use sentence embeddings)
        """
        # Simple token-based similarity for testing
        gen_tokens = set(generated_text.lower().split())
        ref_tokens = set(reference_text.lower().split())
        
        if not gen_tokens or not ref_tokens:
            return EvaluationResult("semantic_similarity", 0.0, 1.0, {"common_tokens": 0})
        
        # Jaccard similarity
        intersection = len(gen_tokens.intersection(ref_tokens))
        union = len(gen_tokens.union(ref_tokens))
        
        similarity = intersection / union if union > 0 else 0.0
        
        return EvaluationResult(
            metric_name="semantic_similarity",
            score=similarity,
            max_score=1.0,
            details={
                "common_tokens": intersection,
                "gen_unique_tokens": len(gen_tokens),
                "ref_unique_tokens": len(ref_tokens),
                "total_unique_tokens": union
            }
        )
    
    @staticmethod
    def answer_relevance(generated_answer: str, question: str) -> EvaluationResult:
        """
        Evaluate how well the answer addresses the question
        """
        # Simple keyword-based relevance
        question_words = set(question.lower().split())
        answer_words = set(generated_answer.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'what', 'how', 'when', 'where', 'why', 'who'}
        
        question_content_words = question_words - stop_words
        answer_content_words = answer_words - stop_words
        
        if not question_content_words:
            return EvaluationResult("answer_relevance", 1.0, 1.0, {"no_content_words": True})
        
        # Calculate overlap of content words
        overlap = len(question_content_words.intersection(answer_content_words))
        relevance = overlap / len(question_content_words)
        
        return EvaluationResult(
            metric_name="answer_relevance",
            score=relevance,
            max_score=1.0,
            details={
                "question_content_words": len(question_content_words),
                "answer_content_words": len(answer_content_words),
                "overlapping_words": overlap
            }
        )
    
    @staticmethod
    def answer_faithfulness(generated_answer: str, source_documents: List[Any]) -> EvaluationResult:
        """
        Evaluate how faithful the answer is to source documents
        """
        if not source_documents:
            return EvaluationResult("answer_faithfulness", 0.0, 1.0, {"no_sources": True})
        
        # Combine all source text
        source_text = " ".join([doc.page_content for doc in source_documents])
        source_words = set(source_text.lower().split())
        
        answer_words = set(generated_answer.lower().split())
        
        if not answer_words:
            return EvaluationResult("answer_faithfulness", 1.0, 1.0, {"empty_answer": True})
        
        # Calculate what fraction of answer words appear in sources
        supported_words = len(answer_words.intersection(source_words))
        faithfulness = supported_words / len(answer_words)
        
        return EvaluationResult(
            metric_name="answer_faithfulness",
            score=faithfulness,
            max_score=1.0,
            details={
                "answer_words": len(answer_words),
                "source_words": len(source_words),
                "supported_words": supported_words
            }
        )


def evaluate_answer_quality(
    question: str,
    generated_answer: str,
    reference_answer: str,
    retrieved_docs: List[Any],
    relevant_doc_ids: List[str]
) -> RAGEvaluationSuite:
    """
    Comprehensive evaluation of RAG system quality
    
    Args:
        question: The input question
        generated_answer: The system's generated answer
        reference_answer: The ground truth answer
        retrieved_docs: Documents retrieved by the system
        relevant_doc_ids: IDs of documents that are actually relevant
    """
    
    # Retrieval metrics
    retrieval_metrics = {
        "precision_at_3": EvaluationMetrics.precision_at_k(retrieved_docs, relevant_doc_ids, k=3),
        "precision_at_5": EvaluationMetrics.precision_at_k(retrieved_docs, relevant_doc_ids, k=5),
        "recall_at_5": EvaluationMetrics.recall_at_k(retrieved_docs, relevant_doc_ids, k=5),
        "f1_at_5": EvaluationMetrics.f1_at_k(retrieved_docs, relevant_doc_ids, k=5),
        "mrr": EvaluationMetrics.mean_reciprocal_rank(retrieved_docs, relevant_doc_ids)
    }
    
    # Generation metrics
    generation_metrics = {
        "rouge_l": EvaluationMetrics.rouge_l(generated_answer, reference_answer),
        "bleu": EvaluationMetrics.bleu_score(generated_answer, reference_answer),
        "semantic_similarity": EvaluationMetrics.semantic_similarity(generated_answer, reference_answer),
        "answer_relevance": EvaluationMetrics.answer_relevance(generated_answer, question),
        "answer_faithfulness": EvaluationMetrics.answer_faithfulness(generated_answer, retrieved_docs)
    }
    
    # Calculate overall score (weighted average)
    retrieval_weights = {"precision_at_5": 0.3, "recall_at_5": 0.3, "mrr": 0.4}
    generation_weights = {"rouge_l": 0.25, "bleu": 0.25, "semantic_similarity": 0.2, "answer_relevance": 0.15, "answer_faithfulness": 0.15}
    
    retrieval_score = sum(
        retrieval_metrics[metric].normalized_score * weight
        for metric, weight in retrieval_weights.items()
    )
    
    generation_score = sum(
        generation_metrics[metric].normalized_score * weight
        for metric, weight in generation_weights.items()
    )
    
    # Overall score (60% generation, 40% retrieval)
    overall_score = 0.6 * generation_score + 0.4 * retrieval_score
    
    return RAGEvaluationSuite(
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        overall_score=overall_score,
        num_queries=1
    )


def aggregate_evaluation_results(evaluation_results: List[RAGEvaluationSuite]) -> RAGEvaluationSuite:
    """
    Aggregate multiple evaluation results into summary statistics
    """
    if not evaluation_results:
        return RAGEvaluationSuite({}, {}, 0.0, 0)
    
    num_queries = len(evaluation_results)
    
    # Aggregate retrieval metrics
    retrieval_metrics = {}
    all_retrieval_metric_names = set()
    for result in evaluation_results:
        all_retrieval_metric_names.update(result.retrieval_metrics.keys())
    
    for metric_name in all_retrieval_metric_names:
        scores = [result.retrieval_metrics.get(metric_name, EvaluationResult(metric_name, 0.0, 1.0, {})).score 
                 for result in evaluation_results]
        avg_score = sum(scores) / len(scores)
        
        retrieval_metrics[metric_name] = EvaluationResult(
            metric_name=metric_name,
            score=avg_score,
            max_score=1.0,
            details={"mean": avg_score, "std": np.std(scores), "num_queries": num_queries}
        )
    
    # Aggregate generation metrics
    generation_metrics = {}
    all_generation_metric_names = set()
    for result in evaluation_results:
        all_generation_metric_names.update(result.generation_metrics.keys())
    
    for metric_name in all_generation_metric_names:
        scores = [result.generation_metrics.get(metric_name, EvaluationResult(metric_name, 0.0, 1.0, {})).score
                 for result in evaluation_results]
        avg_score = sum(scores) / len(scores)
        
        generation_metrics[metric_name] = EvaluationResult(
            metric_name=metric_name,
            score=avg_score,
            max_score=1.0,
            details={"mean": avg_score, "std": np.std(scores), "num_queries": num_queries}
        )
    
    # Overall score
    overall_scores = [result.overall_score for result in evaluation_results]
    overall_score = sum(overall_scores) / len(overall_scores)
    
    return RAGEvaluationSuite(
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        overall_score=overall_score,
        num_queries=num_queries
    )