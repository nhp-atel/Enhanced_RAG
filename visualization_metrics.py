"""
Visualization and metrics for RAG system evaluation.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    rouge_1_f: float
    rouge_2_f: float
    rouge_l_f: float
    bleu_score: float
    exact_match: float
    semantic_similarity: float
    retrieval_precision: float
    retrieval_recall: float
    response_time: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'ROUGE-1 F1': self.rouge_1_f,
            'ROUGE-2 F1': self.rouge_2_f,
            'ROUGE-L F1': self.rouge_l_f,
            'BLEU Score': self.bleu_score,
            'Exact Match': self.exact_match,
            'Semantic Similarity': self.semantic_similarity,
            'Retrieval Precision': self.retrieval_precision,
            'Retrieval Recall': self.retrieval_recall,
            'Response Time (s)': self.response_time
        }


class EmbeddingVisualizer:
    """Visualize embeddings using dimensionality reduction."""
    
    def __init__(self):
        self.reduction_methods = {
            'tsne': TSNE,
            'pca': PCA,
            'umap': umap.UMAP
        }
    
    def reduce_dimensions(
        self, 
        embeddings: np.ndarray, 
        method: str = 'umap',
        n_components: int = 2,
        **kwargs
    ) -> np.ndarray:
        """
        Reduce embedding dimensions for visualization.
        
        Args:
            embeddings: High-dimensional embeddings
            method: Reduction method ('tsne', 'pca', 'umap')
            n_components: Target dimensions
            **kwargs: Additional parameters for reduction method
            
        Returns:
            Reduced embeddings
        """
        if method not in self.reduction_methods:
            raise ValueError(f"Unknown method: {method}")
        
        reducer_class = self.reduction_methods[method]
        
        # Set default parameters based on method
        if method == 'tsne':
            default_params = {'perplexity': 30, 'random_state': 42}
        elif method == 'pca':
            default_params = {'random_state': 42}
        elif method == 'umap':
            default_params = {'random_state': 42, 'n_neighbors': 15}
        
        # Merge with user parameters
        params = {**default_params, **kwargs, 'n_components': n_components}
        
        logger.info(f"Reducing {embeddings.shape} embeddings using {method}")
        
        reducer = reducer_class(**params)
        reduced = reducer.fit_transform(embeddings)
        
        return reduced
    
    def plot_embeddings_2d(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        categories: Optional[List[str]] = None,
        method: str = 'umap',
        title: str = "Document Embeddings Visualization",
        save_path: Optional[str] = None
    ) -> go.Figure:
        """
        Create 2D visualization of embeddings.
        
        Args:
            embeddings: High-dimensional embeddings
            labels: Document labels/titles
            categories: Document categories for coloring
            method: Dimensionality reduction method
            title: Plot title
            save_path: Path to save plot
            
        Returns:
            Plotly figure
        """
        # Reduce dimensions
        reduced = self.reduce_dimensions(embeddings, method=method, n_components=2)
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': reduced[:, 0],
            'y': reduced[:, 1],
            'label': labels,
            'category': categories if categories else ['Document'] * len(labels)
        })
        
        # Create interactive plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='category',
            hover_data=['label'],
            title=title,
            labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'}
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(
            width=800,
            height=600,
            font=dict(size=12),
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved visualization to {save_path}")
        
        return fig
    
    def plot_embedding_clusters(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        document_labels: List[str],
        method: str = 'umap',
        title: str = "Document Clustering Visualization"
    ) -> go.Figure:
        """
        Visualize document clusters in embedding space.
        
        Args:
            embeddings: Document embeddings
            cluster_labels: Cluster assignments
            document_labels: Document titles/labels
            method: Dimensionality reduction method
            title: Plot title
            
        Returns:
            Plotly figure
        """
        # Reduce dimensions
        reduced = self.reduce_dimensions(embeddings, method=method, n_components=2)
        
        # Create DataFrame
        df = pd.DataFrame({
            'x': reduced[:, 0],
            'y': reduced[:, 1],
            'cluster': cluster_labels.astype(str),
            'document': document_labels
        })
        
        # Create plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='cluster',
            hover_data=['document'],
            title=title,
            labels={'x': f'{method.upper()} 1', 'y': f'{method.upper()} 2'}
        )
        
        # Add cluster centers if available
        unique_clusters = np.unique(cluster_labels)
        for cluster in unique_clusters:
            cluster_mask = cluster_labels == cluster
            center_x = reduced[cluster_mask, 0].mean()
            center_y = reduced[cluster_mask, 1].mean()
            
            fig.add_trace(go.Scatter(
                x=[center_x],
                y=[center_y],
                mode='markers',
                marker=dict(
                    size=15,
                    symbol='x',
                    color='black',
                    line=dict(width=2)
                ),
                name=f'Cluster {cluster} Center',
                showlegend=False
            ))
        
        fig.update_layout(width=800, height=600)
        return fig
    
    def create_similarity_heatmap(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        title: str = "Document Similarity Heatmap"
    ) -> go.Figure:
        """
        Create similarity heatmap between documents.
        
        Args:
            embeddings: Document embeddings
            labels: Document labels
            title: Plot title
            
        Returns:
            Plotly heatmap figure
        """
        # Compute cosine similarity matrix
        normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = np.dot(normalized, normalized.T)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=labels,
            y=labels,
            colorscale='Viridis',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Documents",
            yaxis_title="Documents",
            width=800,
            height=800
        )
        
        return fig


class RAGEvaluator:
    """Evaluate RAG system performance."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )
        
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def evaluate_qa_performance(
        self,
        questions: List[str],
        generated_answers: List[str],
        reference_answers: List[str],
        retrieval_results: Optional[List[List[Dict]]] = None
    ) -> EvaluationMetrics:
        """
        Comprehensive evaluation of QA performance.
        
        Args:
            questions: List of questions
            generated_answers: System-generated answers
            reference_answers: Ground truth answers
            retrieval_results: Optional retrieval results for precision/recall
            
        Returns:
            Evaluation metrics
        """
        if len(generated_answers) != len(reference_answers):
            raise ValueError("Generated and reference answers must have same length")
        
        # ROUGE scores
        rouge_scores = self._compute_rouge_scores(generated_answers, reference_answers)
        
        # BLEU scores
        bleu_score = self._compute_bleu_score(generated_answers, reference_answers)
        
        # Exact match
        exact_match = self._compute_exact_match(generated_answers, reference_answers)
        
        # Semantic similarity (mock implementation)
        semantic_similarity = self._compute_semantic_similarity(generated_answers, reference_answers)
        
        # Retrieval metrics
        if retrieval_results:
            precision, recall = self._compute_retrieval_metrics(retrieval_results)
        else:
            precision, recall = 0.0, 0.0
        
        return EvaluationMetrics(
            rouge_1_f=rouge_scores['rouge1_f'],
            rouge_2_f=rouge_scores['rouge2_f'],
            rouge_l_f=rouge_scores['rougeL_f'],
            bleu_score=bleu_score,
            exact_match=exact_match,
            semantic_similarity=semantic_similarity,
            retrieval_precision=precision,
            retrieval_recall=recall,
            response_time=0.0  # Would be measured in actual evaluation
        )
    
    def _compute_rouge_scores(
        self, 
        generated: List[str], 
        reference: List[str]
    ) -> Dict[str, float]:
        """Compute ROUGE scores."""
        rouge_scores = defaultdict(list)
        
        for gen, ref in zip(generated, reference):
            scores = self.rouge_scorer.score(ref, gen)
            rouge_scores['rouge1_f'].append(scores['rouge1'].fmeasure)
            rouge_scores['rouge2_f'].append(scores['rouge2'].fmeasure)
            rouge_scores['rougeL_f'].append(scores['rougeL'].fmeasure)
        
        return {
            metric: np.mean(scores) 
            for metric, scores in rouge_scores.items()
        }
    
    def _compute_bleu_score(
        self, 
        generated: List[str], 
        reference: List[str]
    ) -> float:
        """Compute BLEU score."""
        bleu_scores = []
        
        for gen, ref in zip(generated, reference):
            # Tokenize
            gen_tokens = nltk.word_tokenize(gen.lower())
            ref_tokens = [nltk.word_tokenize(ref.lower())]
            
            # Compute BLEU
            try:
                score = sentence_bleu(ref_tokens, gen_tokens)
                bleu_scores.append(score)
            except:
                bleu_scores.append(0.0)
        
        return np.mean(bleu_scores)
    
    def _compute_exact_match(
        self, 
        generated: List[str], 
        reference: List[str]
    ) -> float:
        """Compute exact match accuracy."""
        exact_matches = [
            gen.strip().lower() == ref.strip().lower()
            for gen, ref in zip(generated, reference)
        ]
        return np.mean(exact_matches)
    
    def _compute_semantic_similarity(
        self, 
        generated: List[str], 
        reference: List[str]
    ) -> float:
        """Compute semantic similarity (mock implementation)."""
        # In a real implementation, you'd use sentence embeddings
        # and compute cosine similarity
        return 0.75  # Mock value
    
    def _compute_retrieval_metrics(
        self, 
        retrieval_results: List[List[Dict]]
    ) -> Tuple[float, float]:
        """Compute retrieval precision and recall."""
        # Mock implementation - would need ground truth relevance judgments
        precision_scores = []
        recall_scores = []
        
        for results in retrieval_results:
            # Mock calculation
            relevant_retrieved = len([r for r in results if r.get('relevant', False)])
            total_retrieved = len(results)
            total_relevant = 5  # Mock number
            
            precision = relevant_retrieved / total_retrieved if total_retrieved > 0 else 0
            recall = relevant_retrieved / total_relevant if total_relevant > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        return np.mean(precision_scores), np.mean(recall_scores)
    
    def create_evaluation_report(
        self,
        metrics: EvaluationMetrics,
        save_path: Optional[str] = None
    ) -> go.Figure:
        """Create comprehensive evaluation report visualization."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('ROUGE Scores', 'Overall Metrics', 
                          'Retrieval Performance', 'Response Time'),
            specs=[[{"type": "bar"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "indicator"}]]
        )
        
        # ROUGE scores
        rouge_data = {
            'ROUGE-1': metrics.rouge_1_f,
            'ROUGE-2': metrics.rouge_2_f,
            'ROUGE-L': metrics.rouge_l_f
        }
        
        fig.add_trace(
            go.Bar(
                x=list(rouge_data.keys()),
                y=list(rouge_data.values()),
                name="ROUGE Scores",
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Overall performance indicator
        overall_score = np.mean([
            metrics.rouge_1_f, metrics.bleu_score, 
            metrics.exact_match, metrics.semantic_similarity
        ])
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=overall_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Performance"},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                }
            ),
            row=1, col=2
        )
        
        # Retrieval performance
        retrieval_data = {
            'Precision': metrics.retrieval_precision,
            'Recall': metrics.retrieval_recall
        }
        
        fig.add_trace(
            go.Bar(
                x=list(retrieval_data.keys()),
                y=list(retrieval_data.values()),
                name="Retrieval Metrics",
                marker_color='lightgreen'
            ),
            row=2, col=1
        )
        
        # Response time indicator
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=metrics.response_time,
                title={'text': "Avg Response Time (s)"},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="RAG System Evaluation Report",
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class PerformanceMonitor:
    """Monitor and visualize system performance over time."""
    
    def __init__(self):
        self.metrics_history = []
    
    def record_metrics(
        self,
        timestamp: str,
        metrics: Dict[str, float]
    ):
        """Record performance metrics."""
        record = {'timestamp': timestamp, **metrics}
        self.metrics_history.append(record)
    
    def plot_performance_trends(
        self,
        metrics_to_plot: List[str],
        title: str = "Performance Trends Over Time"
    ) -> go.Figure:
        """Plot performance trends over time."""
        if not self.metrics_history:
            return go.Figure()
        
        df = pd.DataFrame(self.metrics_history)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            if metric in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'],
                    y=df[metric],
                    mode='lines+markers',
                    name=metric
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Score",
            hovermode='x unified'
        )
        
        return fig
    
    def export_metrics(self, file_path: str):
        """Export metrics to CSV."""
        if self.metrics_history:
            df = pd.DataFrame(self.metrics_history)
            df.to_csv(file_path, index=False)
            logger.info(f"Exported metrics to {file_path}")


# Usage example
def main():
    """Example usage of visualization and metrics."""
    # Mock data
    embeddings = np.random.rand(50, 128)  # 50 documents, 128-dim embeddings
    labels = [f"Document {i+1}" for i in range(50)]
    categories = ['Research', 'Tutorial', 'Survey'] * 17 + ['Research']  # 50 total
    
    # Visualization
    visualizer = EmbeddingVisualizer()
    
    # 2D embedding plot
    fig = visualizer.plot_embeddings_2d(
        embeddings, 
        labels, 
        categories,
        method='umap',
        title="Research Papers Embedding Space"
    )
    fig.show()
    
    # Evaluation
    evaluator = RAGEvaluator()
    
    # Mock QA data
    questions = ["What is the main contribution?", "Who are the authors?"]
    generated = ["The main contribution is...", "The authors are John Doe and Jane Smith"]
    reference = ["The paper's main contribution is...", "Authors: John Doe, Jane Smith"]
    
    metrics = evaluator.evaluate_qa_performance(questions, generated, reference)
    
    # Create evaluation report
    report_fig = evaluator.create_evaluation_report(metrics)
    report_fig.show()
    
    print("Metrics:", metrics.to_dict())


if __name__ == "__main__":
    main()