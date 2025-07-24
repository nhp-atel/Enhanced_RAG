"""
Regression tests for prompt engineering changes.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import Mock, patch
import hashlib
import time

from tests.utils.evaluation_metrics import evaluate_answer_quality, aggregate_evaluation_results
from tests.utils.test_corpus import GROUND_TRUTH_QUERIES, create_test_corpus
from src.core.pipeline import RAGPipeline
from src.utils.prompts import PromptManager


class PromptRegressionTester:
    """Framework for testing prompt changes against baselines"""
    
    def __init__(self, baseline_file: str = "tests/baselines/prompt_baselines.json"):
        self.baseline_file = Path(baseline_file)
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        self.current_results = {}
        self.baseline_results = self._load_baseline()
    
    def _load_baseline(self) -> Dict[str, Any]:
        """Load baseline results from file"""
        if self.baseline_file.exists():
            try:
                with open(self.baseline_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
    
    def save_baseline(self) -> None:
        """Save current results as new baseline"""
        with open(self.baseline_file, 'w') as f:
            json.dump(self.current_results, f, indent=2)
    
    def run_prompt_evaluation(self, pipeline: RAGPipeline, prompt_version: str) -> Dict[str, Any]:
        """Run evaluation with current prompts"""
        
        # Test queries across different types
        test_queries = GROUND_TRUTH_QUERIES[:8]  # Use first 8 queries for regression testing
        evaluation_results = []
        
        for query in test_queries:
            try:
                # Get answer using current prompts
                result = pipeline.query(query.question)
                
                # Evaluate quality
                evaluation = evaluate_answer_quality(
                    question=query.question,
                    generated_answer=result.answer,
                    reference_answer=query.expected_answer,
                    retrieved_docs=result.sources,
                    relevant_doc_ids=query.relevant_doc_ids
                )
                
                evaluation_results.append(evaluation)
                
            except Exception as e:
                print(f"Error evaluating query '{query.question}': {e}")
                continue
        
        # Aggregate results
        if evaluation_results:
            aggregated = aggregate_evaluation_results(evaluation_results)
            prompt_results = {
                "version": prompt_version,
                "timestamp": time.time(),
                "overall_score": aggregated.overall_score,
                "num_queries": aggregated.num_queries,
                "retrieval_metrics": {
                    name: {
                        "score": result.score,
                        "details": result.details
                    } for name, result in aggregated.retrieval_metrics.items()
                },
                "generation_metrics": {
                    name: {
                        "score": result.score,
                        "details": result.details
                    } for name, result in aggregated.generation_metrics.items()
                }
            }
        else:
            prompt_results = {
                "version": prompt_version,
                "timestamp": time.time(),
                "overall_score": 0.0,
                "num_queries": 0,
                "error": "No successful evaluations"
            }
        
        self.current_results[prompt_version] = prompt_results
        return prompt_results
    
    def compare_with_baseline(self, current_version: str, baseline_version: str = None) -> Dict[str, Any]:
        """Compare current results with baseline"""
        
        if current_version not in self.current_results:
            raise ValueError(f"No results found for version {current_version}")
        
        if baseline_version is None:
            # Find most recent baseline version
            baseline_versions = [v for v in self.baseline_results.keys() if v != current_version]
            if not baseline_versions:
                return {"error": "No baseline version found"}
            baseline_version = max(baseline_versions, key=lambda v: self.baseline_results[v].get("timestamp", 0))
        
        if baseline_version not in self.baseline_results:
            return {"error": f"Baseline version {baseline_version} not found"}
        
        current = self.current_results[current_version]
        baseline = self.baseline_results[baseline_version]
        
        # Compare overall scores
        score_change = current["overall_score"] - baseline["overall_score"]
        
        # Compare individual metrics
        metric_changes = {}
        
        for metric_type in ["retrieval_metrics", "generation_metrics"]:
            if metric_type in current and metric_type in baseline:
                for metric_name in current[metric_type]:
                    if metric_name in baseline[metric_type]:
                        current_score = current[metric_type][metric_name]["score"]
                        baseline_score = baseline[metric_type][metric_name]["score"]
                        change = current_score - baseline_score
                        metric_changes[f"{metric_type}.{metric_name}"] = {
                            "current": current_score,
                            "baseline": baseline_score,
                            "change": change,
                            "relative_change": change / baseline_score if baseline_score > 0 else 0
                        }
        
        return {
            "current_version": current_version,
            "baseline_version": baseline_version,
            "overall_score_change": score_change,
            "overall_relative_change": score_change / baseline["overall_score"] if baseline["overall_score"] > 0 else 0,
            "metric_changes": metric_changes,
            "regression_detected": self._detect_regression(score_change, metric_changes),
            "comparison_timestamp": time.time()
        }
    
    def _detect_regression(self, score_change: float, metric_changes: Dict[str, Any]) -> Dict[str, Any]:
        """Detect if there's a significant regression"""
        
        # Define regression thresholds
        thresholds = {
            "overall_score": -0.05,  # 5% decrease in overall score
            "retrieval_metrics.precision_at_5": -0.03,  # 3% decrease in precision
            "generation_metrics.rouge_l": -0.02,  # 2% decrease in ROUGE
            "generation_metrics.answer_faithfulness": -0.04,  # 4% decrease in faithfulness
        }
        
        regressions = []
        
        # Check overall score regression
        if score_change < thresholds["overall_score"]:
            regressions.append({
                "metric": "overall_score",
                "change": score_change,
                "threshold": thresholds["overall_score"],
                "severity": "high" if score_change < thresholds["overall_score"] * 2 else "medium"
            })
        
        # Check individual metric regressions
        for metric_name, change_data in metric_changes.items():
            if metric_name in thresholds:
                if change_data["change"] < thresholds[metric_name]:
                    regressions.append({
                        "metric": metric_name,
                        "change": change_data["change"],
                        "threshold": thresholds[metric_name],
                        "severity": "high" if change_data["change"] < thresholds[metric_name] * 2 else "medium"
                    })
        
        return {
            "has_regression": len(regressions) > 0,
            "regressions": regressions,
            "total_regressions": len(regressions),
            "high_severity_count": len([r for r in regressions if r["severity"] == "high"])
        }


class TestPromptRegression:
    """Test suite for prompt regression detection"""
    
    @pytest.fixture
    def regression_tester(self):
        """Regression tester instance"""
        return PromptRegressionTester()
    
    @pytest.fixture
    def mock_pipeline(self, test_config):
        """Mock pipeline for prompt testing"""
        
        with patch('src.core.pipeline.create_llm_client') as mock_create_llm, \
             patch('src.core.pipeline.create_embedding_client') as mock_create_embed, \
             patch('src.core.pipeline.create_vector_store') as mock_create_store:
            
            # Setup deterministic mock responses
            mock_llm = Mock()
            mock_embed = Mock()
            mock_store = Mock()
            
            mock_create_llm.return_value = mock_llm
            mock_create_embed.return_value = mock_embed
            mock_create_store.return_value = mock_store
            
            # Configure embedding mock
            def create_embedding(text: str) -> List[float]:
                import hashlib
                import numpy as np
                text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
                np.random.seed(text_hash % (2**31))
                return np.random.normal(0, 1, 768).tolist()
            
            mock_embed.embed_documents.side_effect = lambda texts: [create_embedding(text) for text in texts]
            mock_embed.embed_query.side_effect = lambda text: create_embedding(text)
            
            # Configure LLM with prompt-sensitive responses
            def prompt_sensitive_generate(messages, **kwargs):
                message_text = " ".join([msg.get("content", "") for msg in messages])
                
                # Different responses based on prompt structure
                if "context:" in message_text.lower() and "question:" in message_text.lower():
                    # Structured prompt format
                    if "authors" in message_text.lower():
                        return Mock(content="The authors are Ashish Vaswani, Noam Shazeer, and colleagues from Google Brain and Google Research.")
                    elif "transformer" in message_text.lower() and "architecture" in message_text.lower():
                        return Mock(content="The Transformer is a neural network architecture based entirely on attention mechanisms, eliminating the need for recurrence and convolutions.")
                    elif "bert" in message_text.lower():
                        return Mock(content="BERT stands for Bidirectional Encoder Representations from Transformers and uses bidirectional training.")
                    else:
                        return Mock(content="Based on the provided context, this paper discusses advanced neural network architectures.")
                else:
                    # Unstructured prompt format - slightly lower quality responses
                    if "authors" in message_text.lower():
                        return Mock(content="Authors include Vaswani and others.")
                    elif "transformer" in message_text.lower():
                        return Mock(content="The Transformer uses attention.")
                    else:
                        return Mock(content="The paper discusses neural networks.")
            
            mock_llm.generate.side_effect = prompt_sensitive_generate
            mock_llm.generate_with_retry.side_effect = prompt_sensitive_generate
            
            # Setup retrieval
            test_docs = create_test_corpus()
            mock_store.similarity_search.return_value = test_docs[:6]
            
            pipeline = RAGPipeline(test_config) 
            
            # Mock retriever
            def mock_retrieve_and_generate(question, k=6, strategy="basic"):
                sources = test_docs[:k]
                context = "\n".join([doc.page_content for doc in sources[:3]])
                messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"}]
                response = pipeline.llm_client.generate(messages)
                
                return Mock(
                    answer=response.content,
                    sources=sources,
                    processing_time_ms=100,
                    tokens_used=150,
                    cost_usd=0.01,
                    metadata={"strategy": strategy}
                )
            
            mock_retriever = Mock()
            mock_retriever.retrieve_and_generate = mock_retrieve_and_generate
            pipeline.retriever = mock_retriever
            
            return pipeline
    
    def test_baseline_creation(self, regression_tester, mock_pipeline):
        """Test creating a new baseline"""
        
        # Run evaluation to create baseline
        results = regression_tester.run_prompt_evaluation(mock_pipeline, "baseline_v1.0")
        
        assert "baseline_v1.0" in regression_tester.current_results
        assert results["overall_score"] >= 0.0
        assert results["num_queries"] > 0
        
        # Save as baseline
        regression_tester.save_baseline()
        
        # Verify baseline was saved
        assert regression_tester.baseline_file.exists()
    
    def test_prompt_improvement_detection(self, regression_tester, mock_pipeline):
        """Test detection of prompt improvements"""
        
        # Create a baseline with lower performance
        baseline_results = {
            "baseline_v1.0": {
                "version": "baseline_v1.0",
                "timestamp": time.time() - 3600,  # 1 hour ago
                "overall_score": 0.40,
                "num_queries": 8,
                "retrieval_metrics": {
                    "precision_at_5": {"score": 0.20},
                    "recall_at_5": {"score": 0.25}
                },
                "generation_metrics": {
                    "rouge_l": {"score": 0.12},
                    "answer_faithfulness": {"score": 0.35}
                }
            }
        }
        
        regression_tester.baseline_results = baseline_results
        
        # Run evaluation with "improved" prompts (mock should give better responses)
        current_results = regression_tester.run_prompt_evaluation(mock_pipeline, "improved_v1.1")
        
        # Compare with baseline
        comparison = regression_tester.compare_with_baseline("improved_v1.1", "baseline_v1.0")
        
        assert comparison["overall_score_change"] >= 0, "Expected improvement in overall score"
        assert not comparison["regression_detected"]["has_regression"], "Should not detect regression for improvement"
    
    def test_prompt_regression_detection(self, regression_tester, mock_pipeline):
        """Test detection of prompt regressions"""
        
        # Create a high-performance baseline
        baseline_results = {
            "baseline_v1.0": {
                "version": "baseline_v1.0", 
                "timestamp": time.time() - 3600,
                "overall_score": 0.75,  # High baseline score
                "num_queries": 8,
                "retrieval_metrics": {
                    "precision_at_5": {"score": 0.60},
                    "recall_at_5": {"score": 0.65}
                },
                "generation_metrics": {
                    "rouge_l": {"score": 0.45},
                    "answer_faithfulness": {"score": 0.70}
                }
            }
        }
        
        regression_tester.baseline_results = baseline_results
        
        # Current results should be lower (since mock gives simpler responses)
        current_results = regression_tester.run_prompt_evaluation(mock_pipeline, "regressed_v1.1")
        
        # Compare with baseline
        comparison = regression_tester.compare_with_baseline("regressed_v1.1", "baseline_v1.0")
        
        # Should detect regression if current performance is significantly lower
        if comparison["overall_score_change"] < -0.05:  # More than 5% decrease
            assert comparison["regression_detected"]["has_regression"], "Should detect regression"
            assert len(comparison["regression_detected"]["regressions"]) > 0
    
    def test_prompt_version_tracking(self, regression_tester, mock_pipeline):
        """Test tracking multiple prompt versions"""
        
        # Create multiple versions
        versions = ["v1.0", "v1.1", "v1.2"]
        
        for version in versions:
            results = regression_tester.run_prompt_evaluation(mock_pipeline, version)
            assert version in regression_tester.current_results
            assert results["version"] == version
        
        # Verify all versions are tracked
        assert len(regression_tester.current_results) == len(versions)
        
        # Test comparison between non-baseline versions
        if "v1.1" in regression_tester.current_results and "v1.0" in regression_tester.current_results:
            regression_tester.baseline_results = {"v1.0": regression_tester.current_results["v1.0"]}
            comparison = regression_tester.compare_with_baseline("v1.1", "v1.0")
            
            assert "current_version" in comparison
            assert "baseline_version" in comparison
            assert comparison["current_version"] == "v1.1"
            assert comparison["baseline_version"] == "v1.0"
    
    def test_prompt_change_sensitivity(self, regression_tester, test_config):
        """Test that system detects changes in prompts"""
        
        # Test with different prompt styles
        with patch('src.core.pipeline.create_llm_client') as mock_create_llm, \
             patch('src.core.pipeline.create_embedding_client') as mock_create_embed, \
             patch('src.core.pipeline.create_vector_store') as mock_create_store:
            
            mock_llm = Mock()
            mock_embed = Mock() 
            mock_store = Mock()
            
            mock_create_llm.return_value = mock_llm
            mock_create_embed.return_value = mock_embed
            mock_create_store.return_value = mock_store
            
            # Configure different responses for different prompt styles
            def style_sensitive_generate(messages, **kwargs):
                message_text = " ".join([msg.get("content", "") for msg in messages])
                
                if "please answer the following question" in message_text.lower():
                    # Polite prompt style
                    return Mock(content="I'd be happy to help. The authors of the Transformer paper are Ashish Vaswani and his colleagues from Google Brain.")
                elif "answer:" in message_text.lower():
                    # Direct prompt style
                    return Mock(content="Ashish Vaswani, Noam Shazeer, and others from Google.")
                else:
                    # Default style
                    return Mock(content="The paper was written by researchers at Google.")
            
            mock_llm.generate.side_effect = style_sensitive_generate
            mock_llm.generate_with_retry.side_effect = style_sensitive_generate
            
            # Setup other mocks
            mock_embed.embed_query.return_value = [0.1] * 768
            mock_store.similarity_search.return_value = create_test_corpus()[:6]
            
            pipeline = RAGPipeline(test_config)
            
            # Mock retriever with prompt-sensitive behavior
            def prompt_style_retrieve(question, k=6, strategy="basic"):
                sources = create_test_corpus()[:k]
                
                # Simulate different prompt templates
                if strategy == "polite":
                    context = "\n".join([doc.page_content for doc in sources[:3]])
                    messages = [{"role": "user", "content": f"Context: {context}\n\nPlease answer the following question: {question}"}]
                else:
                    context = "\n".join([doc.page_content for doc in sources[:3]])
                    messages = [{"role": "user", "content": f"Context: {context}\n\nQuestion: {question}\nAnswer:"}]
                
                response = pipeline.llm_client.generate(messages)
                
                return Mock(
                    answer=response.content,
                    sources=sources,
                    processing_time_ms=100,
                    tokens_used=150,
                    cost_usd=0.01,
                    metadata={"strategy": strategy}
                )
            
            mock_retriever = Mock()
            mock_retriever.retrieve_and_generate = prompt_style_retrieve
            pipeline.retriever = mock_retriever
            
            # Test different prompt styles
            original_query = pipeline.query
            
            # Style 1: Direct
            def direct_query(question, **kwargs):
                return pipeline.retriever.retrieve_and_generate(question, strategy="direct", **kwargs)
            
            # Style 2: Polite
            def polite_query(question, **kwargs):
                return pipeline.retriever.retrieve_and_generate(question, strategy="polite", **kwargs)
            
            # Test both styles
            pipeline.query = direct_query
            direct_results = regression_tester.run_prompt_evaluation(pipeline, "direct_style")
            
            pipeline.query = polite_query
            polite_results = regression_tester.run_prompt_evaluation(pipeline, "polite_style")
            
            # Verify different results for different prompt styles
            assert direct_results["overall_score"] != polite_results["overall_score"] or \
                   len(direct_results.get("generation_metrics", {})) != len(polite_results.get("generation_metrics", {}))
    
    def test_regression_threshold_configuration(self, regression_tester, mock_pipeline):
        """Test configurable regression thresholds"""
        
        # Create baseline
        baseline_results = {
            "baseline_v1.0": {
                "version": "baseline_v1.0",
                "timestamp": time.time() - 3600,
                "overall_score": 0.50,
                "num_queries": 8,
                "retrieval_metrics": {
                    "precision_at_5": {"score": 0.40}
                },
                "generation_metrics": {
                    "rouge_l": {"score": 0.30},
                    "answer_faithfulness": {"score": 0.45}
                }
            }
        }
        
        regression_tester.baseline_results = baseline_results
        
        # Run current evaluation
        current_results = regression_tester.run_prompt_evaluation(mock_pipeline, "current_v1.1")
        
        # Test comparison and regression detection
        comparison = regression_tester.compare_with_baseline("current_v1.1", "baseline_v1.0")
        
        # Verify regression detection has configurable thresholds
        regression_info = comparison["regression_detected"]
        
        assert "has_regression" in regression_info
        assert "regressions" in regression_info
        assert "total_regressions" in regression_info
        
        # Check that thresholds are applied correctly
        for regression in regression_info["regressions"]:
            assert "metric" in regression
            assert "change" in regression
            assert "threshold" in regression
            assert "severity" in regression
            assert regression["change"] < regression["threshold"]  # Change should be worse than threshold
    
    def test_prompt_regression_reporting(self, regression_tester, mock_pipeline):
        """Test detailed regression reporting"""
        
        # Setup baseline and current results
        baseline_results = {
            "baseline_v1.0": {
                "version": "baseline_v1.0",
                "timestamp": time.time() - 3600,
                "overall_score": 0.60,
                "num_queries": 8,
                "retrieval_metrics": {
                    "precision_at_5": {"score": 0.45},
                    "recall_at_5": {"score": 0.50}
                },
                "generation_metrics": {
                    "rouge_l": {"score": 0.35},
                    "answer_faithfulness": {"score": 0.55}
                }
            }
        }
        
        regression_tester.baseline_results = baseline_results
        
        # Run evaluation
        current_results = regression_tester.run_prompt_evaluation(mock_pipeline, "test_v1.1")
        comparison = regression_tester.compare_with_baseline("test_v1.1", "baseline_v1.0")
        
        # Verify comprehensive comparison data
        assert "current_version" in comparison
        assert "baseline_version" in comparison
        assert "overall_score_change" in comparison
        assert "overall_relative_change" in comparison
        assert "metric_changes" in comparison
        assert "regression_detected" in comparison
        assert "comparison_timestamp" in comparison
        
        # Verify metric changes include detailed information
        for metric_name, change_data in comparison["metric_changes"].items():
            assert "current" in change_data
            assert "baseline" in change_data
            assert "change" in change_data
            assert "relative_change" in change_data


@pytest.mark.integration
def test_end_to_end_prompt_regression_workflow():
    """End-to-end test of prompt regression detection workflow"""
    
    # This test simulates a complete prompt regression testing workflow
    # that would be run in CI/CD
    
    regression_tester = PromptRegressionTester("tests/baselines/e2e_test_baseline.json")
    
    # Mock pipeline setup would go here...
    # For brevity, using a mock evaluation result
    
    # Simulate baseline creation
    baseline_evaluation = {
        "version": "production_v1.0",
        "timestamp": time.time() - 86400,  # 1 day ago
        "overall_score": 0.65,
        "num_queries": 10,
        "retrieval_metrics": {
            "precision_at_5": {"score": 0.50, "details": {}},
            "recall_at_5": {"score": 0.55, "details": {}}
        },
        "generation_metrics": {
            "rouge_l": {"score": 0.40, "details": {}},
            "answer_faithfulness": {"score": 0.60, "details": {}}
        }
    }
    
    regression_tester.baseline_results["production_v1.0"] = baseline_evaluation
    
    # Simulate current evaluation
    current_evaluation = {
        "version": "candidate_v1.1",
        "timestamp": time.time(),
        "overall_score": 0.58,  # Slightly lower
        "num_queries": 10,
        "retrieval_metrics": {
            "precision_at_5": {"score": 0.48, "details": {}},  # Slight decrease
            "recall_at_5": {"score": 0.53, "details": {}}      # Slight decrease
        },
        "generation_metrics": {
            "rouge_l": {"score": 0.38, "details": {}},         # Slight decrease
            "answer_faithfulness": {"score": 0.55, "details": {}} # Decrease
        }
    }
    
    regression_tester.current_results["candidate_v1.1"] = current_evaluation
    
    # Run comparison
    comparison = regression_tester.compare_with_baseline("candidate_v1.1", "production_v1.0")
    
    # Verify workflow results
    assert comparison["overall_score_change"] < 0  # Performance decreased
    
    # Check if regression is detected (depends on thresholds)
    regression_detected = comparison["regression_detected"]["has_regression"]
    
    if regression_detected:
        print("❌ Regression detected - deployment should be blocked")
        print(f"Regressions found: {comparison['regression_detected']['total_regressions']}")
        for regression in comparison["regression_detected"]["regressions"]:
            print(f"  - {regression['metric']}: {regression['change']:.3f} (threshold: {regression['threshold']:.3f})")
    else:
        print("✅ No significant regression detected - deployment can proceed")
    
    # Cleanup test baseline file
    if Path("tests/baselines/e2e_test_baseline.json").exists():
        Path("tests/baselines/e2e_test_baseline.json").unlink()