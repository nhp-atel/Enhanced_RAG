"""
Test corpus and ground truth data for evaluation.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
from langchain_core.documents import Document


@dataclass
class GroundTruthQuery:
    """Ground truth query with expected answer and relevant documents"""
    question: str
    expected_answer: str
    answer_type: str  # "factual", "conceptual", "technical", "summary"
    relevant_doc_ids: List[str]
    keywords: List[str]
    difficulty: str  # "easy", "medium", "hard"


# Ground truth queries for evaluation
GROUND_TRUTH_QUERIES = [
    GroundTruthQuery(
        question="Who are the authors of the Transformer paper?",
        expected_answer="Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin",
        answer_type="factual",
        relevant_doc_ids=["transformer_metadata", "transformer_authors"],
        keywords=["authors", "vaswani", "shazeer", "transformer"],
        difficulty="easy"
    ),
    GroundTruthQuery(
        question="What is the main contribution of the Transformer paper?",
        expected_answer="The main contribution is proposing the Transformer architecture that relies entirely on attention mechanisms, dispensing with recurrence and convolutions entirely",
        answer_type="conceptual",
        relevant_doc_ids=["transformer_abstract", "transformer_intro", "transformer_conclusion"],
        keywords=["contribution", "attention", "architecture", "recurrence", "convolution"],
        difficulty="medium"
    ),
    GroundTruthQuery(
        question="How does multi-head attention work in the Transformer?",
        expected_answer="Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. It projects queries, keys, and values into multiple subspaces and applies attention in parallel",
        answer_type="technical",
        relevant_doc_ids=["transformer_attention", "transformer_multihead"],
        keywords=["multi-head", "attention", "subspaces", "queries", "keys", "values"],
        difficulty="hard"
    ),
    GroundTruthQuery(
        question="What are the advantages of the Transformer over RNNs?",
        expected_answer="The Transformer is more parallelizable than RNNs, requires significantly less time to train, and achieves better results on translation tasks",
        answer_type="conceptual",
        relevant_doc_ids=["transformer_comparison", "transformer_results"],
        keywords=["advantages", "parallelizable", "RNN", "training time", "results"],
        difficulty="medium"
    ),
    GroundTruthQuery(
        question="What datasets were used to evaluate the Transformer?",
        expected_answer="The Transformer was evaluated on WMT 2014 English-German and English-French translation tasks",
        answer_type="factual",
        relevant_doc_ids=["transformer_experiments", "transformer_datasets"],
        keywords=["datasets", "WMT", "English-German", "English-French", "translation"],
        difficulty="easy"
    ),
    GroundTruthQuery(
        question="What is BERT and how does it relate to the Transformer?",
        expected_answer="BERT (Bidirectional Encoder Representations from Transformers) uses the encoder part of the Transformer architecture with bidirectional training to create contextualized word representations",
        answer_type="conceptual",
        relevant_doc_ids=["bert_abstract", "bert_architecture", "bert_transformer"],
        keywords=["BERT", "bidirectional", "encoder", "transformer", "representations"],
        difficulty="medium"
    ),
    GroundTruthQuery(
        question="What pre-training tasks does BERT use?",
        expected_answer="BERT uses two pre-training tasks: Masked Language Model (MLM) where random tokens are masked and predicted, and Next Sentence Prediction (NSP) where the model predicts if two sentences are consecutive",
        answer_type="technical",
        relevant_doc_ids=["bert_pretraining", "bert_mlm", "bert_nsp"],
        keywords=["pre-training", "MLM", "masked language model", "NSP", "next sentence prediction"],
        difficulty="hard"
    ),
    GroundTruthQuery(
        question="How does BERT handle input representations?",
        expected_answer="BERT uses WordPiece embeddings combined with learned positional embeddings and segment embeddings to create the final input representation",
        answer_type="technical",
        relevant_doc_ids=["bert_input", "bert_embeddings", "bert_wordpiece"],
        keywords=["input", "WordPiece", "positional embeddings", "segment embeddings"],
        difficulty="hard"
    )
]


def create_test_corpus() -> List[Document]:
    """Create a test corpus with documents covering Transformer and BERT papers"""
    
    documents = [
        # Transformer paper documents
        Document(
            page_content="""PAPER METADATA:
Title: Attention Is All You Need
Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin
Institutions: Google Brain, Google Research, University of Toronto
Publication Date: 2017-06-12
ArXiv ID: 1706.03762
Keywords: attention mechanism, transformer, neural networks, sequence modeling
Abstract: We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
--- END OF METADATA ---""",
            metadata={
                "document_id": "transformer_metadata",
                "source": "transformer_paper.pdf",
                "type": "paper_metadata",
                "chunk_id": "metadata_0"
            }
        ),
        
        Document(
            page_content="""Abstract
The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show that these models are superior in quality while being more parallelizable and requiring significantly less time to train.""",
            metadata={
                "document_id": "transformer_abstract",
                "source": "transformer_paper.pdf",
                "type": "content",
                "section": "abstract",
                "chunk_id": "abstract_0"
            }
        ),
        
        Document(
            page_content="""1 Introduction
Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15].

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.""",
            metadata={
                "document_id": "transformer_intro",
                "source": "transformer_paper.pdf", 
                "type": "content",
                "section": "introduction",
                "chunk_id": "intro_0"
            }
        ),
        
        Document(
            page_content="""3.2.1 Scaled Dot-Product Attention
We call our particular attention "Scaled Dot-Product Attention". The input consists of queries and keys of dimension dk, and values of dimension dv. We compute the dot products of the query with all keys, divide each by √dk, and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. Similarly, the keys and values are packed together into matrices K and V. We compute the matrix of outputs as:

Attention(Q, K, V) = softmax(QKᵀ/√dk)V""",
            metadata={
                "document_id": "transformer_attention",
                "source": "transformer_paper.pdf",
                "type": "content", 
                "section": "attention",
                "chunk_id": "attention_0"
            }
        ),
        
        Document(
            page_content="""3.2.2 Multi-Head Attention
Instead of performing a single attention function with dmodel-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values h times with different, learned linear projections to dk, dk and dv dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding dv-dimensional output values. These are concatenated and once again projected, resulting in the final values.

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.""",
            metadata={
                "document_id": "transformer_multihead",
                "source": "transformer_paper.pdf",
                "type": "content",
                "section": "multihead_attention", 
                "chunk_id": "multihead_0"
            }
        ),
        
        Document(
            page_content="""6 Results
6.1 Machine Translation
On the WMT 2014 English-to-German translation task, the big Transformer model (Transformer (big)) outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU, establishing a new single-model state-of-the-art BLEU score of 28.4. On the WMT 2014 English-to-French translation task, our model achieves a BLEU score of 41.8, outperforming all of the previously published single models, at less than 1/4 the training cost of the previous state-of-the-art model.""",
            metadata={
                "document_id": "transformer_results",
                "source": "transformer_paper.pdf",
                "type": "content",
                "section": "results",
                "chunk_id": "results_0"
            }
        ),
        
        Document(
            page_content="""Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention.

The Transformer allows for significantly more parallelization than recurrent models and can reach any position in constant time, compared to O(n) for recurrent layers. While the per-layer computational complexity is higher for very long sequences, in most practical applications of sequence modeling the sequence length n is much smaller than the representation dimension d.""",
            metadata={
                "document_id": "transformer_comparison",
                "source": "transformer_paper.pdf",
                "type": "content",
                "section": "comparison",
                "chunk_id": "comparison_0" 
            }
        ),
        
        Document(
            page_content="""We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. We also trained on the larger WMT 2014 English-French dataset consisting of 36M sentences. Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens for English-German and 32000 tokens for English-French.""",
            metadata={
                "document_id": "transformer_datasets",
                "source": "transformer_paper.pdf",
                "type": "content",
                "section": "experiments",
                "chunk_id": "datasets_0"
            }
        ),
        
        # BERT paper documents
        Document(
            page_content="""PAPER METADATA:
Title: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Authors: Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova
Institutions: Google AI Language
Publication Date: 2018-10-11
ArXiv ID: 1810.04805
Keywords: BERT, bidirectional, transformer, language understanding, pre-training
Abstract: We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers.
--- END OF METADATA ---""",
            metadata={
                "document_id": "bert_metadata",
                "source": "bert_paper.pdf",
                "type": "paper_metadata",
                "chunk_id": "metadata_0"
            }
        ),
        
        Document(
            page_content="""Abstract
We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models, BERT is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT model can be finetuned with just one additional output layer to create state-of-the-art models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.""",
            metadata={
                "document_id": "bert_abstract",
                "source": "bert_paper.pdf",
                "type": "content",
                "section": "abstract",
                "chunk_id": "abstract_0"
            }
        ),
        
        Document(
            page_content="""3.1 Model Architecture
BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017) and released in the tensor2tensor library. Because the use of bidirectional self-attention makes it impossible to use a standard language modeling objective, we use different pre-training objectives, detailed in Section 3.3.

In this work, we denote the number of layers (i.e., Transformer blocks) as L, the hidden size as H, and the number of self-attention heads as A. We primarily report results on two model sizes: BERT_BASE (L=12, H=768, A=12, Total Parameters=110M) and BERT_LARGE (L=24, H=1024, A=16, Total Parameters=340M).""",
            metadata={
                "document_id": "bert_architecture",
                "source": "bert_paper.pdf",
                "type": "content",
                "section": "architecture",
                "chunk_id": "architecture_0"
            }
        ),
        
        Document(
            page_content="""3.2 Input/Output Representations
To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences (e.g., ⟨Question, Answer⟩) in one token sequence. Throughout this work, a "sentence" can be an arbitrary span of contiguous text, rather than an actual linguistic sentence. A "sequence" refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together.

We use WordPiece embeddings with a 30,000 token vocabulary. The first token of every sequence is always a special classification token ([CLS]). The final hidden state corresponding to this token is used as the aggregate sequence representation for classification tasks. Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ([SEP]). Second, we add a learned embedding to every token indicating whether it belongs to sentence A or sentence B.""",
            metadata={
                "document_id": "bert_input",
                "source": "bert_paper.pdf",
                "type": "content",
                "section": "input_representation",
                "chunk_id": "input_0"
            }
        ),
        
        Document(
            page_content="""3.3 Pre-training BERT
We pre-train BERT using two unsupervised tasks, described in this section.

Task #1: Masked LM
Standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly "see itself", and the model could trivially predict the target word in a multi-layered context. In order to train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens. We refer to this procedure as a "masked LM" (MLM), although it is often referred to as a Cloze task in the literature. In this case, the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM.""",
            metadata={
                "document_id": "bert_mlm",
                "source": "bert_paper.pdf",
                "type": "content",
                "section": "pretraining",
                "chunk_id": "mlm_0"
            }
        ),
        
        Document(
            page_content="""Task #2: Next Sentence Prediction (NSP)
Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two sentences, which is not directly captured by language modeling. In order to train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Specifically, when choosing the sentences A and B for each pre-training example, 50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext).""",
            metadata={
                "document_id": "bert_nsp", 
                "source": "bert_paper.pdf",
                "type": "content",
                "section": "pretraining",
                "chunk_id": "nsp_0"
            }
        ),
        
        Document(
            page_content="""For WordPiece embeddings, we use a 30,000 token vocabulary that was learned on the lower-cased English text. We denote split word pieces with ##. For example, "playing" would be split into "play" and "##ing".

For position embeddings, we use learned positional embeddings with supported sequence lengths up to 512 tokens. For segment embeddings, we learn embeddings for sentence A and sentence B.

The input embedding is the sum of the token embeddings, the segmentation embeddings and the position embeddings.""",
            metadata={
                "document_id": "bert_embeddings",
                "source": "bert_paper.pdf", 
                "type": "content",
                "section": "embeddings",
                "chunk_id": "embeddings_0"
            }
        )
    ]
    
    return documents


def get_ground_truth_for_query(question: str) -> GroundTruthQuery:
    """Get ground truth data for a specific question"""
    for gt_query in GROUND_TRUTH_QUERIES:
        if gt_query.question.lower() == question.lower():
            return gt_query
    return None


def get_queries_by_type(answer_type: str) -> List[GroundTruthQuery]:
    """Get all queries of a specific type"""
    return [q for q in GROUND_TRUTH_QUERIES if q.answer_type == answer_type]


def get_queries_by_difficulty(difficulty: str) -> List[GroundTruthQuery]:
    """Get all queries of a specific difficulty"""
    return [q for q in GROUND_TRUTH_QUERIES if q.difficulty == difficulty]


def create_evaluation_dataset() -> Dict[str, Any]:
    """Create a complete evaluation dataset with documents and queries"""
    return {
        "documents": create_test_corpus(),
        "queries": GROUND_TRUTH_QUERIES,
        "metadata": {
            "num_documents": len(create_test_corpus()),
            "num_queries": len(GROUND_TRUTH_QUERIES),
            "query_types": list(set(q.answer_type for q in GROUND_TRUTH_QUERIES)),
            "difficulty_levels": list(set(q.difficulty for q in GROUND_TRUTH_QUERIES)),
            "papers_covered": ["Attention Is All You Need", "BERT"]
        }
    }