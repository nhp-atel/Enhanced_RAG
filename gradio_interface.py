"""
Gradio web interface for the RAG system.
"""
import gradio as gr
import asyncio
from typing import List, Tuple, Optional, Dict, Any
import logging
from pathlib import Path
import json
import tempfile
import pandas as pd

from config_example import get_config
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class RAGWebInterface:
    """Web interface for RAG system using Gradio."""
    
    def __init__(self):
        self.config = get_config()
        self.llm = ChatOpenAI(
            model=self.config.chat_model,
            temperature=self.config.temperature
        )
        self.processed_papers = {}  # Store processed papers
        self.current_paper_id = None
    
    def process_paper_upload(self, file) -> Tuple[str, str, Dict]:
        """Process uploaded PDF file."""
        if file is None:
            return "❌ No file uploaded", "", {}
        
        try:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file.read())
                file_path = tmp_file.name
            
            # Process paper (mock implementation)
            paper_id = f"paper_{len(self.processed_papers) + 1}"
            
            # Mock processing results
            processing_result = {
                'id': paper_id,
                'title': 'Sample Research Paper',
                'authors': ['Author 1', 'Author 2'],
                'publication_date': '2024',
                'total_chunks': 142,
                'concept_count': 23,
                'processing_time': 45.2
            }
            
            self.processed_papers[paper_id] = processing_result
            self.current_paper_id = paper_id
            
            # Format success message
            success_msg = f"""
✅ **Paper processed successfully!**

📋 **Details:**
- **Title:** {processing_result['title']}
- **Authors:** {', '.join(processing_result['authors'])}
- **Publication Date:** {processing_result['publication_date']}
- **Document Chunks:** {processing_result['total_chunks']}
- **Concepts Extracted:** {processing_result['concept_count']}
- **Processing Time:** {processing_result['processing_time']}s

🎯 **Ready for questions!**
            """
            
            # Create metadata display
            metadata_json = json.dumps(processing_result, indent=2)
            
            return success_msg, metadata_json, gr.update(visible=True)
            
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return f"❌ Processing failed: {str(e)}", "", {}
    
    def process_paper_url(self, url: str) -> Tuple[str, str, Dict]:
        """Process paper from URL."""
        if not url.strip():
            return "❌ Please enter a valid URL", "", {}
        
        try:
            # Mock URL processing
            paper_id = f"paper_{len(self.processed_papers) + 1}"
            
            processing_result = {
                'id': paper_id,
                'source_url': url,
                'title': 'Neural Networks for Natural Language Processing',
                'authors': ['Research Team'],
                'publication_date': '2024',
                'total_chunks': 89,
                'concept_count': 31,
                'processing_time': 38.7
            }
            
            self.processed_papers[paper_id] = processing_result
            self.current_paper_id = paper_id
            
            success_msg = f"""
✅ **Paper processed from URL!**

🔗 **Source:** {url}

📋 **Details:**
- **Title:** {processing_result['title']}
- **Authors:** {', '.join(processing_result['authors'])}
- **Document Chunks:** {processing_result['total_chunks']}
- **Concepts Extracted:** {processing_result['concept_count']}

🎯 **Ready for questions!**
            """
            
            metadata_json = json.dumps(processing_result, indent=2)
            
            return success_msg, metadata_json, gr.update(visible=True)
            
        except Exception as e:
            logger.error(f"URL processing failed: {e}")
            return f"❌ Processing failed: {str(e)}", "", {}
    
    def answer_question(
        self, 
        question: str, 
        chat_history: List[List[str]]
    ) -> Tuple[List[List[str]], str]:
        """Process user question and return answer."""
        if not question.strip():
            return chat_history, ""
        
        if not self.current_paper_id:
            response = "❌ Please process a paper first before asking questions."
            chat_history.append([question, response])
            return chat_history, ""
        
        try:
            # Mock question answering
            mock_answers = {
                "title": f"The title of this paper is '{self.processed_papers[self.current_paper_id]['title']}'.",
                "authors": f"The authors are: {', '.join(self.processed_papers[self.current_paper_id]['authors'])}.",
                "abstract": "This paper presents novel approaches to neural network architectures...",
                "methodology": "The methodology involves training deep learning models on large datasets...",
                "results": "The results show significant improvements over baseline methods...",
                "conclusion": "The paper concludes that the proposed approach is effective..."
            }
            
            # Simple keyword matching for demo
            question_lower = question.lower()
            response = "I'd be happy to help answer that question based on the processed paper. "
            
            for keyword, answer in mock_answers.items():
                if keyword in question_lower:
                    response = answer
                    break
            else:
                response += f"Regarding '{question}', based on my analysis of the paper, this appears to be related to the research methodology and findings discussed in the document."
            
            # Add to chat history
            chat_history.append([question, response])
            
            return chat_history, ""
            
        except Exception as e:
            logger.error(f"Question answering failed: {e}")
            error_response = f"❌ Error processing question: {str(e)}"
            chat_history.append([question, error_response])
            return chat_history, ""
    
    def get_paper_stats(self) -> str:
        """Get statistics about processed papers."""
        if not self.processed_papers:
            return "📊 **No papers processed yet**"
        
        total_papers = len(self.processed_papers)
        total_chunks = sum(p['total_chunks'] for p in self.processed_papers.values())
        total_concepts = sum(p['concept_count'] for p in self.processed_papers.values())
        
        stats = f"""
📊 **Processing Statistics:**

- **Total Papers Processed:** {total_papers}
- **Total Document Chunks:** {total_chunks}
- **Total Concepts Extracted:** {total_concepts}
- **Average Chunks per Paper:** {total_chunks / total_papers:.1f}
- **Average Concepts per Paper:** {total_concepts / total_papers:.1f}
        """
        
        return stats
    
    def export_results(self) -> Optional[str]:
        """Export processing results to CSV."""
        if not self.processed_papers:
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(self.processed_papers, orient='index')
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            return tmp_file.name
    
    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface."""
        with gr.Blocks(
            title="🚀 Enhanced RAG System",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            .panel {
                border-radius: 10px;
                padding: 20px;
            }
            """
        ) as interface:
            
            # Header
            gr.Markdown("""
            # 🚀 Enhanced RAG System
            ### Universal Research Paper Analysis with Dynamic Processing
            
            Upload a PDF or provide an arXiv URL to get started!
            """)
            
            with gr.Tabs():
                # Tab 1: Paper Processing
                with gr.Tab("📄 Process Paper"):
                    gr.Markdown("### Choose your input method:")
                    
                    with gr.Row():
                        # File upload column
                        with gr.Column(scale=1):
                            gr.Markdown("#### 📁 Upload PDF File")
                            file_input = gr.File(
                                label="Select PDF file",
                                file_types=['.pdf'],
                                type="binary"
                            )
                            upload_btn = gr.Button("🔄 Process Upload", variant="primary")
                        
                        # URL input column  
                        with gr.Column(scale=1):
                            gr.Markdown("#### 🔗 Enter Paper URL")
                            url_input = gr.Textbox(
                                label="Paper URL (arXiv, etc.)",
                                placeholder="https://arxiv.org/pdf/2101.00001",
                                lines=1
                            )
                            url_btn = gr.Button("🔄 Process URL", variant="primary")
                    
                    # Processing results
                    with gr.Row():
                        processing_status = gr.Markdown("📝 Upload a file or enter a URL to begin processing...")
                    
                    # Metadata display (initially hidden)
                    with gr.Row(visible=False) as metadata_row:
                        with gr.Column():
                            gr.Markdown("#### 📋 Extracted Metadata")
                            metadata_display = gr.Code(
                                language="json",
                                label="Paper Metadata"
                            )
                
                # Tab 2: Question Answering
                with gr.Tab("❓ Ask Questions"):
                    gr.Markdown("### Chat with your processed paper")
                    
                    with gr.Row():
                        with gr.Column(scale=3):
                            chatbot = gr.Chatbot(
                                label="Q&A Chat",
                                height=400,
                                show_label=True
                            )
                            
                            with gr.Row():
                                question_input = gr.Textbox(
                                    label="Your question",
                                    placeholder="What is the main contribution of this paper?",
                                    lines=2,
                                    scale=4
                                )
                                ask_btn = gr.Button("🤔 Ask", variant="primary", scale=1)
                        
                        with gr.Column(scale=1):
                            gr.Markdown("#### 💡 Example Questions")
                            example_questions = gr.Examples(
                                examples=[
                                    ["What is the title of this paper?"],
                                    ["Who are the authors?"],
                                    ["What is the main methodology?"],
                                    ["What are the key findings?"],
                                    ["What datasets were used?"],
                                    ["What are the limitations?"]
                                ],
                                inputs=question_input,
                                label="Click to use:"
                            )
                
                # Tab 3: Statistics & Export
                with gr.Tab("📊 Statistics"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("### 📈 Processing Statistics")
                            stats_display = gr.Markdown("📊 No papers processed yet")
                            refresh_stats_btn = gr.Button("🔄 Refresh Stats")
                        
                        with gr.Column():
                            gr.Markdown("### 💾 Export Results")
                            export_btn = gr.Button("📥 Export to CSV", variant="secondary")
                            export_file = gr.File(label="Download Results", visible=False)
            
            # Event handlers
            upload_btn.click(
                fn=self.process_paper_upload,
                inputs=[file_input],
                outputs=[processing_status, metadata_display, metadata_row]
            )
            
            url_btn.click(
                fn=self.process_paper_url,
                inputs=[url_input],
                outputs=[processing_status, metadata_display, metadata_row]
            )
            
            ask_btn.click(
                fn=self.answer_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            
            question_input.submit(
                fn=self.answer_question,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            
            refresh_stats_btn.click(
                fn=self.get_paper_stats,
                outputs=[stats_display]
            )
            
            export_btn.click(
                fn=self.export_results,
                outputs=[export_file]
            ).then(
                lambda x: gr.update(visible=True) if x else gr.update(visible=False),
                inputs=[export_file],
                outputs=[export_file]
            )
        
        return interface
    
    def launch(
        self, 
        share: bool = False, 
        server_name: str = "127.0.0.1",
        server_port: int = 7860,
        auth: Optional[Tuple[str, str]] = None
    ):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        
        logger.info(f"Launching Gradio interface on {server_name}:{server_port}")
        
        interface.launch(
            share=share,
            server_name=server_name,
            server_port=server_port,
            auth=auth,
            show_error=True,
            quiet=False
        )


def main():
    """Main function to launch the web interface."""
    app = RAGWebInterface()
    app.launch(
        share=False,  # Set to True to create public link
        server_name="0.0.0.0",  # Allow external connections
        server_port=7860,
        # auth=("admin", "password")  # Optional authentication
    )


if __name__ == "__main__":
    main()